"""Read-only disjointness and frozen-contract audit; writes one receipt after all blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sample import ARMS, CUTOFFS, ROOT, STREAMS, arm_for, prior


def selected_rules(chunks: Path) -> set[tuple[int, int]]:
    result = set()
    for path in sorted(chunks.glob("chunk-*.npz")):
        with np.load(path) as data:
            result.update(tuple(map(int, pair)) for pair in data["rule_seed"])
    return result


def prior_rules() -> set[tuple[int, int]]:
    result = set()
    for path in (prior.SOURCE / "receipts").glob("batch-*.json"):
        for row in json.loads(path.read_text())["accepted"]:
            result.add(tuple(row["rule_seed"]))
    for chunks in (prior.REPO / "runs/nca-data-search").rglob("chunks"):
        if ROOT in chunks.parents:
            continue
        result.update(selected_rules(chunks))
    return result


def main() -> None:
    target = prior.quotas()
    earlier = prior_rules()
    seen: set[tuple[int, int]] = set()
    blocks = {}
    for block in STREAMS:
        root = ROOT / f"block-{block}"
        config = json.loads((root / "sampling_config.json").read_text())
        assert config["stream_seed"] == STREAMS[block]
        assert config["cutoffs_by_gzip_bin"] == CUTOFFS.tolist()
        assert config["quotas_per_arm"] == target.tolist()
        counts = np.zeros((2, 2, 5), np.int64)
        arm_rules = [set(), set()]
        previous_attempt = -1
        for path in sorted((root / "chunks").glob("chunk-*.npz")):
            with np.load(path) as data:
                for i, rule_seed in enumerate(data["rule_seed"]):
                    rule = tuple(map(int, rule_seed))
                    arm, split, bin_ = map(int, (data["arm"][i], data["split"][i], data["bin"][i]))
                    ratio, score = float(data["gzip_ratio"][i]), float(data["patch_copy_excess"][i])
                    attempt = int(data["attempt"][i])
                    assert attempt > previous_attempt
                    previous_attempt = attempt
                    assert arm == arm_for(rule_seed)
                    assert split == prior.split_for(rule_seed)
                    assert bin_ == prior.gzip_bin(ratio)
                    if arm == 0:
                        assert score >= CUTOFFS[bin_]
                    assert np.isclose(score, prior.horizontal_copy_excess(data["states"][i]))
                    assert rule not in arm_rules[arm] and rule not in earlier and rule not in seen
                    arm_rules[arm].add(rule)
                    counts[arm, split, bin_] += 1
        assert np.array_equal(counts[0], target) and np.array_equal(counts[1], target)
        assert not (arm_rules[0] & arm_rules[1])
        seen.update(arm_rules[0] | arm_rules[1])
        for arm in ARMS:
            receipt = json.loads((root / arm / "done.json").read_text())
            assert receipt["train_usable_tokens"] == 19_715_325
            assert receipt["gzip_bin_counts"] == target.tolist()
            assert prior.sha(root / arm / "packed.pt") == receipt["packed_sha256"]
        blocks[block] = {
            "selected_per_arm": {arm: len(arm_rules[i]) for i, arm in enumerate(ARMS)},
            "attempts": json.loads((root / "designed/done.json").read_text())["attempts"],
            "packed_sha256_by_arm": {arm: prior.sha(root / arm / "packed.pt") for arm in ARMS},
        }
    assert len(seen) == 3 * 2 * 20_231
    receipt = {
        "schema": "nca-top10-confirmation-audit-v1",
        "protocol_sha256": prior.sha(Path(__file__).with_name("PROTOCOL.md")),
        "sampler_sha256": prior.sha(Path(__file__).with_name("sample.py")),
        "earlier_rule_count": len(earlier),
        "earlier_selected_overlap": 0,
        "cross_block_or_arm_overlap": 0,
        "selected_total": len(seen),
        "blocks": blocks,
    }
    prior.write_json(ROOT / "audit.json", receipt)
    print(f"AUDIT PASS: {len(seen)} fresh, disjoint selected rules across three blocks", flush=True)


if __name__ == "__main__":
    main()
