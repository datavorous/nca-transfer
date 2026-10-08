"""Audit a fresh patch-copy block before any model training.

Run: uv run --all-packages python progress/nca_search/audit_patch_replication.py --block a
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

from sample_patch_replication import (ARMS, BLOCK_SEEDS, REPO, SOURCE, THRESHOLD,
                                      arm_for, gzip_bin, horizontal_copy_excess,
                                      quotas, sha, split_for)


def earlier_rules(block: str) -> set[tuple[int, int]]:
    seen = set()
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for row in json.loads(path.read_text())["accepted"]:
            seen.add(tuple(map(int, row["rule_seed"])))
    search = REPO / "runs/nca-data-search"
    for directory in sorted(search.glob("iteration-*")):
        for path in sorted((directory / "chunks").glob("chunk-*.npz")):
            with np.load(path) as data:
                seen.update(tuple(map(int, key)) for key in data["rule_seed"])
    for directory in sorted((search / "patch-replication").glob("block-*")):
        if directory.name == f"block-{block}":
            continue
        for path in sorted((directory / "chunks").glob("chunk-*.npz")):
            with np.load(path) as data:
                seen.update(tuple(map(int, key)) for key in data["rule_seed"])
    return seen


def rule_stream(cursor: jax.Array, length: int) -> tuple[jax.Array, np.ndarray]:
    def step(key, _):
        key, rule = jax.random.split(key)
        return key, rule
    following, rules = jax.lax.scan(step, cursor, None, length=length)
    return following, np.asarray(rules, dtype=np.uint32)


def audit(block: str) -> None:
    root = REPO / "runs/nca-data-search/patch-replication" / f"block-{block}"
    config_path = root / "sampling_config.json"
    config = json.loads(config_path.read_text())
    assert config["stream_seed"] == BLOCK_SEEDS[block]
    assert config["threshold"] == THRESHOLD
    assert config["quotas_per_arm"] == quotas().tolist()
    assert config["arm_hash_person"] == "nca-arm"
    assert config["split_hash_person"] == "nca-split"
    prior = earlier_rules(block)
    seen: set[tuple[int, int]] = set()
    counts = np.zeros((3, 2, 5), dtype=np.int64)
    cursor = jax.random.PRNGKey(BLOCK_SEEDS[block])
    attempt = 0
    overlap = 0
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
                for field in ("states", "gzip_ratio", "patch_copy_excess", "arm", "split", "bin", "attempt"):
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
                    assert arm == arm_for(rule_seed) and split == split_for(rule_seed)
                    assert split < 2 and bin_ == gzip_bin(ratio)
                    assert arm != 0 or score >= THRESHOLD
                    assert arm != 2 or score <= THRESHOLD
                    counts[arm, split, bin_] += 1
                    if len(seen) % 256 == 1:
                        assert horizontal_copy_excess(data["states"][i]) == score
                        sampled_rechecks += 1
                    output.write(json.dumps({"rule_seed": list(rule_id), "attempt": selected_attempt,
                                             "arm": ARMS[arm], "split": ("train" if split == 0 else "validation"),
                                             "gzip_bin": bin_, "gzip_ratio": ratio,
                                             "patch_copy_excess": score}, sort_keys=True) + "\n")
                attempt = end
        output.flush()
        os.fsync(output.fileno())
    assert np.array_equal(counts, np.broadcast_to(quotas(), (3, 2, 5)))
    assert overlap == 0, f"{overlap} selected rules overlap earlier corpora; invalidate this block"
    os.replace(temporary, index_path)
    for arm in ARMS:
        receipt = json.loads((root / arm / "done.json").read_text())
        assert receipt["selected"] == 20_231
        assert sha(root / arm / "packed.pt") == receipt["packed_sha256"]
        assert receipt["gzip_bin_counts"] == quotas().tolist()
    summary = {"schema": "nca-patch-replication-audit-v1", "block": block,
               "attempts": attempt, "selected_rules": len(seen), "earlier_rule_overlap": overlap,
               "feature_rechecks": sampled_rechecks, "counts_by_arm_split_bin": counts.tolist(),
               "rule_index_sha256": sha(index_path), "sampling_config_sha256": sha(config_path)}
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
