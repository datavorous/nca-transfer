"""Audit a fresh vertical lag-6 block before any model training.

Run: uv run --all-packages python progress/nca_search/audit_vertical_lag6.py --block d
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

from sample_vertical_lag6 import (ARMS, BLOCK_SEEDS, REPO, THRESHOLD,
                                  arm_for, gzip_bin, prior_rules, quotas, sha,
                                  split_for, vertical_excess)
from sampleeff_nca_pretraining.packing import PackedDataset, pack_one


def rule_stream(cursor: jax.Array, length: int) -> tuple[jax.Array, np.ndarray]:
    def step(key, _):
        key, rule = jax.random.split(key)
        return key, rule
    following, rules = jax.lax.scan(step, cursor, None, length=length)
    return following, np.asarray(rules, dtype=np.uint32)


def audit(block: str) -> None:
    root = REPO / "runs/nca-data-search/vertical-lag6" / f"block-{block}"
    config_path = root / "sampling_config.json"
    config = json.loads(config_path.read_text())
    assert config["stream_seed"] == BLOCK_SEEDS[block]
    assert config["threshold"] == THRESHOLD
    assert config["quotas_per_arm"] == quotas().tolist()
    assert config["arm_hash_person"] == "nca-v6-arm"
    assert config["split_hash_person"] == "nca-split"
    assert config["schema"] == "nca-vertical-lag6-v1"
    assert config["reference_sha256"] == sha(REPO / "runs/1m-nca-transfer-v2/data/nca/40_50/packed.pt")
    assert config["sampler_code_sha256"] == sha(REPO / "progress/nca_search/sample_vertical_lag6.py")
    freeze_path = REPO / "progress/nca_search/vertical_lag6_freeze.json"
    assert config["freeze_sha256"] == sha(freeze_path)
    prior = prior_rules(block)
    seen: set[tuple[int, int]] = set()
    counts = np.zeros((3, 2, 5), dtype=np.int64)
    cursor = jax.random.PRNGKey(BLOCK_SEEDS[block])
    attempt = 0
    overlap = 0
    rejected_overlap = 0
    sampled_rechecks = 0
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
                for field in ("states", "rule_seed", "initial_seed", "gzip_ratio", "vertical_excess", "arm", "split", "bin", "attempt"):
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
                    ratio, score = float(data["gzip_ratio"][i]), float(data["vertical_excess"][i])
                    assert arm == arm_for(rule_seed) and split == split_for(rule_seed)
                    assert split < 2 and bin_ == gzip_bin(ratio)
                    assert arm != 0 or score >= THRESHOLD
                    assert arm != 2 or score < THRESHOLD
                    counts[arm, split, bin_] += 1
                    if len(seen) % 256 == 1:
                        assert vertical_excess(data["states"][i]) == score
                        sampled_rechecks += 1
                    output.write(json.dumps({"rule_seed": list(rule_id), "attempt": selected_attempt,
                                             "arm": ARMS[arm], "split": ("train" if split == 0 else "validation"),
                                             "gzip_bin": bin_, "gzip_ratio": ratio,
                                             "vertical_excess": score,
                                             "states_sha256": hashlib.sha256(data["states"][i].tobytes()).hexdigest()},
                                            sort_keys=True) + "\n")
                attempt = end
        output.flush()
        os.fsync(output.fileno())
    assert np.array_equal(counts, np.broadcast_to(quotas(), (3, 2, 5)))
    assert overlap == 0, f"{overlap} selected rules overlap earlier corpora; invalidate this block"
    os.replace(temporary, index_path)
    packed_hashes = {}
    for arm in ARMS:
        receipt = json.loads((root / arm / "done.json").read_text())
        assert receipt["selected"] == 20_231
        packed_hashes[arm] = sha(root / arm / "packed.pt")
        assert packed_hashes[arm] == receipt["packed_sha256"]
        assert receipt["gzip_bin_counts"] == quotas().tolist()
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
    summary = {"schema": "nca-vertical-lag6-audit-v1", "block": block,
               "attempts": attempt, "selected_rules": len(seen), "earlier_rule_overlap": overlap,
               "rejected_rule_overlap": rejected_overlap,
               "feature_rechecks": sampled_rechecks, "counts_by_arm_split_bin": counts.tolist(),
               "rule_index_sha256": sha(index_path), "sampling_config_sha256": sha(config_path),
               "packed_sha256_by_arm": packed_hashes}
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
