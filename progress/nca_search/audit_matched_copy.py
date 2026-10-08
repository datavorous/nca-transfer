"""Audit a fresh matched horizontal-copy block before any model training.

Run: uv run --all-packages python progress/nca_search/audit_matched_copy.py --block g
Checks every selected rule against the rule stream, arm/split hashes, quotas,
and earlier search corpora; writes a complete rule index and audit receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from sample_matched_copy import (ARMS, BLOCK_SEEDS, REPO, THRESHOLD, FREEZE,
                                 arm_for, cell_bins, frozen, gzip_bin, prior_rules, sha,
                                 split_for, world_features)
from sampleeff_nca_pretraining.packing import PackedDataset, pack_one


def rule_stream(cursor: jax.Array, length: int) -> tuple[jax.Array, np.ndarray]:
    def step(key, _):
        key, rule = jax.random.split(key)
        return key, rule
    following, rules = jax.lax.scan(step, cursor, None, length=length)
    return following, np.asarray(rules, dtype=np.uint32)


def audit(block: str) -> None:
    root = REPO / "runs/nca-data-search/matched-copy" / f"block-{block}"
    config_path = root / "sampling_config.json"
    config = json.loads(config_path.read_text())
    assert config["stream_seed"] == BLOCK_SEEDS[block]
    assert config["threshold"] == THRESHOLD
    freeze, target = frozen()
    assert config["joint_quotas_per_arm"] == target.tolist()
    assert config["arm_hash_person"] == "nca-mcopy-arm"
    assert config["split_hash_person"] == "nca-split"
    assert config["schema"] == "nca-matched-copy-v1"
    assert config["reference_sha256"] == sha(REPO / "runs/1m-nca-transfer-v2/data/nca/40_50/packed.pt")
    assert config["sampler_code_sha256"] == sha(REPO / "progress/nca_search/sample_matched_copy.py")
    assert config["freeze_sha256"] == sha(FREEZE)
    prior = prior_rules(block)
    seen: set[tuple[int, int]] = set()
    counts = np.zeros((2, 2, 5, 3, 3), dtype=np.int64)
    cursor = jax.random.PRNGKey(BLOCK_SEEDS[block])
    attempt = 0
    overlap = 0
    rejected_overlap = 0
    sampled_rechecks = 0
    all_features: dict[str, dict[str, list[float]]] = {
        arm: {key: [] for key in ("gzip_ratio", "patch_copy_excess", "lag2", "transition_mi_bits")}
        for arm in ARMS
    }
    index_path = root / "rule_index.jsonl"
    temporary = index_path.with_name(f".{index_path.name}.tmp")
    with temporary.open("w") as output:
        for path in sorted((root / "chunks").glob("chunk-*.npz")):
            with np.load(path) as data:
                start, end = int(data["start"]), int(data["end"])
                assert start == attempt and 0 < end - start <= 4096, path
                cursor, keys = rule_stream(cursor, end - start)
                assert np.array_equal(np.asarray(cursor, dtype=np.uint32), data["cursor_end"]), path
                n = len(data["rule_seed"])
                selected_attempts = set(map(int, data["attempt"]))
                assert len(selected_attempts) == n, path
                previously_selected = seen.copy()
                expected_rejections = 0
                for key_offset, key in enumerate(keys):
                    rule_id = tuple(map(int, key))
                    if rule_id in prior or rule_id in previously_selected:
                        expected_rejections += 1
                    elif start + key_offset in selected_attempts:
                        previously_selected.add(rule_id)
                assert expected_rejections == int(data["overlap_rejections"]), path
                rejected_overlap += expected_rejections
                for field in ("states", "rule_seed", "initial_seed", "gzip_ratio", "patch_copy_excess", "lag2", "transition_mi_bits", "lag2_bin", "mi_bin", "arm", "split", "bin", "attempt"):
                    assert len(data[field]) == n, (path, field)
                for i in range(n):
                    selected_attempt = int(data["attempt"][i])
                    rule_seed = data["rule_seed"][i]
                    assert start <= selected_attempt < end
                    assert np.array_equal(rule_seed, keys[selected_attempt - start])
                    rule_id = tuple(map(int, rule_seed))
                    assert rule_id not in seen, (path, rule_id)
                    seen.add(rule_id)
                    overlap += rule_id in prior
                    arm, split, bin_ = (int(data[key][i]) for key in ("arm", "split", "bin"))
                    ratio, score = float(data["gzip_ratio"][i]), float(data["patch_copy_excess"][i])
                    lag2, mi = float(data["lag2"][i]), float(data["transition_mi_bits"][i])
                    lag_bin, mi_bin = cell_bins(lag2, mi, freeze)
                    assert lag_bin == int(data["lag2_bin"][i]) and mi_bin == int(data["mi_bin"][i])
                    assert arm == arm_for(rule_seed) and split == split_for(rule_seed)
                    assert split < 2 and bin_ == gzip_bin(ratio)
                    assert arm != 0 or score >= THRESHOLD
                    assert arm != 1 or score < THRESHOLD
                    counts[arm, split, bin_, lag_bin, mi_bin] += 1
                    for key in all_features[ARMS[arm]]:
                        all_features[ARMS[arm]][key].append(float(data[key][i]))
                    if len(seen) % 256 == 1:
                        assert world_features(data["states"][i], ratio) == (ratio, score, lag2, mi)
                        sampled_rechecks += 1
                    output.write(json.dumps({"rule_seed": list(rule_id), "attempt": selected_attempt,
                                             "arm": ARMS[arm], "split": ("train" if split == 0 else "validation"),
                                             "gzip_bin": bin_, "gzip_ratio": ratio,
                                             "patch_copy_excess": score, "lag2": lag2, "transition_mi_bits": mi,
                                             "lag2_bin": lag_bin, "mi_bin": mi_bin,
                                             "states_sha256": hashlib.sha256(data["states"][i].tobytes()).hexdigest()},
                                            sort_keys=True) + "\n")
                attempt = end
        output.flush()
        os.fsync(output.fileno())
    assert np.array_equal(counts, np.broadcast_to(target, (2, 2, 5, 3, 3)))
    assert overlap == 0, f"{overlap} selected rules overlap earlier corpora; invalidate this block"
    os.replace(temporary, index_path)
    packed_hashes = {}
    for arm in ARMS:
        receipt = json.loads((root / arm / "done.json").read_text())
        assert receipt["selected"] == 20_231
        packed_hashes[arm] = sha(root / arm / "packed.pt")
        assert packed_hashes[arm] == receipt["packed_sha256"]
        assert receipt["joint_cell_counts"] == target.tolist()
        assert receipt["gzip_bin_counts"] == target.sum(axis=(2, 3)).tolist()
        packed = PackedDataset.load(root / arm / "packed.pt")
        assert packed.sequences.dtype == packed.targets.dtype == torch.int32
        assert packed.sequences.shape == packed.targets.shape == (20_231, 1026)
        assert int((packed.targets != -100).sum()) == 20_231 * 27 * 36
        frames = packed.sequences.reshape(-1, 27, 38)
        masks = packed.targets.reshape(-1, 27, 38)
        assert torch.all(frames[:, :, 0] == 10_000)
        assert torch.all(frames[:, :, -1] == 10_001)
        assert torch.all(masks[:, :, 0] == -100)
        assert torch.all(masks[:, :, -1] == -100)
        assert torch.equal(frames[:, :, 1:-1], masks[:, :, 1:-1])
        assert torch.all((frames[:, :, 1:-1] >= 0) & (frames[:, :, 1:-1] < 10_000))
        assert packed.metadata["grid_len"] == 38
        assert packed.metadata["num_colors"] == 10
        assert packed.metadata["patch_size"] == 2
        assert packed.metadata["usable_training_tokens"] == 19_975 * 987
        train = packed.metadata["split_indices"]["train"]
        validation = packed.metadata["split_indices"]["validation"]
        assert len(train) == 19_975 and len(validation) == 256
        assert len(set(train) | set(validation)) == 20_231
        assert not (set(train) & set(validation))
        for i in (0, len(packed.sequences) // 2, len(packed.sequences) - 1):
            # Locate the same selected world in the chunk stream for an independent pack_one check.
            offset = i
            for path in sorted((root / "chunks").glob("chunk-*.npz")):
                with np.load(path) as data:
                    keep = np.flatnonzero(data["arm"] == ARMS.index(arm))
                    if offset < len(keep):
                        seq, targets = pack_one(data["states"][keep[offset]], num_colors=10, patch_size=2)
                        assert np.array_equal(packed.sequences[i].numpy(), seq.numpy())
                        assert np.array_equal(packed.targets[i].numpy(), targets.numpy())
                        break
                    offset -= len(keep)
            else:
                raise AssertionError((arm, i))
    values = {arm: {key: np.asarray(rows, dtype=np.float64)
                    for key, rows in measures.items()}
              for arm, measures in all_features.items()}
    means = {arm: {key: float(rows.mean()) for key, rows in measures.items()}
             for arm, measures in values.items()}
    nuisance_smd = {}
    for key in ("lag2", "transition_mi_bits"):
        hi, lo = values["high"][key], values["low"][key]
        pooled_sd = float(np.sqrt((hi.var() + lo.var()) / 2))
        nuisance_smd[key] = float((hi.mean() - lo.mean()) / pooled_sd)
    copy_gap = means["high"]["patch_copy_excess"] - means["low"]["patch_copy_excess"]
    manipulation_passed = (copy_gap > 0 and all(abs(value) < .10 for value in nuisance_smd.values()))
    summary = {"schema": "nca-matched-copy-audit-v1", "block": block,
               "attempts": attempt, "selected_rules": len(seen), "earlier_rule_overlap": overlap,
               "rejected_rule_overlap": rejected_overlap,
               "feature_rechecks": sampled_rechecks, "counts_by_arm_split_gzip_lag2_mi": counts.tolist(),
               "quota_freeze_sha256": sha(FREEZE),
               "rule_index_sha256": sha(index_path), "sampling_config_sha256": sha(config_path),
               "packed_sha256_by_arm": packed_hashes,
               "feature_means_by_arm": means,
               "full_gzip_cdf_quantiles_0_to_100_by_arm": {
                   arm: np.quantile(values[arm]["gzip_ratio"], np.linspace(0, 1, 101)).tolist()
                   for arm in ARMS},
               "lag2_and_mi_standardized_mean_differences": nuisance_smd,
               "high_minus_low_patch_copy_excess": copy_gap,
               "manipulation_check_passed": manipulation_passed}
    receipt_path = root / "audit.json"
    tmp_receipt = receipt_path.with_name(f".{receipt_path.name}.tmp")
    tmp_receipt.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_receipt, receipt_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=BLOCK_SEEDS, required=True)
    args = parser.parse_args()
    audit(args.block)
