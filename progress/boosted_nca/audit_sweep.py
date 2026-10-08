"""Audit both completed sweep corpora before any model training."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import sample_sweep as sweep
from examples.transfer_1m import NCAWindows
from nca.analyze import gzip_ratio
from sampleeff_nca_pretraining.packing import PackedDataset, pack_one


def main() -> None:
    root = sweep.ROOT
    config_path = root / "sampling_config.json"
    config = json.loads(config_path.read_text())
    assert config["schema"] == "nca-boosted-sweep-v1"
    assert config["cutoffs_by_gzip_bin"] == sweep.CUTOFFS.tolist()
    assert config["game"] == json.loads((sweep.ANCHOR / "sampling_config.json").read_text())["game"]
    target = sweep.source.quotas()
    counts = np.zeros((2, 2, 5), np.int64)
    observed: list[list[tuple]] = [[], []]
    attempts = set()
    end = 0
    for path in sorted(sweep.CHUNKS.glob("chunk-*.npz")):
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
            assert int(arrays["start"]) == end, path
            end = int(arrays["end"])
            assert len(arrays["states"]) == len(arrays["attempt"])
            assert len(np.unique(arrays["attempt"])) == len(arrays["attempt"])
            assert np.all((int(arrays["start"]) <= arrays["attempt"]) & (arrays["attempt"] < end))
            for i in range(len(arrays["states"])):
                attempt = int(arrays["attempt"][i])
                assert attempt not in attempts
                attempts.add(attempt)
                seed = arrays["rule_seed"][i]
                split, bin_ = int(arrays["split"][i]), int(arrays["bin"][i])
                assert sweep.source.arm_for(seed) == 0
                assert sweep.source.split_for(seed) == split and split in (0, 1)
                ratio = float(arrays["gzip_ratio"][i])
                assert sweep.source.gzip_bin(ratio) == bin_
                score = float(arrays["patch_copy_excess"][i])
                assert abs(sweep.source.horizontal_copy_excess(arrays["states"][i])-score) < 1e-12
                for tier, key in enumerate(("keep_25", "keep_10")):
                    if arrays[key][i]:
                        assert score >= sweep.CUTOFFS[tier, bin_]
                        counts[tier, split, bin_] += 1
                        observed[tier].append((seed.copy(), split, bin_, ratio, score,
                                               attempt, arrays["states"][i].copy()))
    assert np.array_equal(counts[0], target) and np.array_equal(counts[1], target)
    assert end <= sweep.MAX_ATTEMPTS

    arm_rules = {}
    for tier, name in enumerate(sweep.TIERS):
        directory = root / name
        receipt = json.loads((directory / "done.json").read_text())
        assert receipt["packed_sha256"] == sweep.source.sha(directory / "packed.pt")
        assert receipt["sampling_config_sha256"] == sweep.source.sha(config_path)
        assert receipt["attempts"] == end and receipt["gzip_bin_counts"] == target.tolist()
        rows = observed[tier]
        assert len(rows) == 20_231 and all(rows[i][5] < rows[i+1][5] for i in range(len(rows)-1))
        rules = {tuple(row[0]) for row in rows}
        assert len(rules) == len(rows)
        arm_rules[name] = rules
        packed = PackedDataset.load(directory / "packed.pt")
        assert packed.sequences.shape == (20_231, 1026)
        assert packed.targets.shape == (20_231, 1026)
        assert packed.sequences.dtype == torch.int32 and packed.targets.dtype == torch.int32
        assert packed.metadata["usable_training_tokens"] == 19_715_325
        expected_split = {"train": [i for i, row in enumerate(rows) if row[1] == 0],
                          "validation": [i for i, row in enumerate(rows) if row[1] == 1], "test": []}
        assert packed.metadata["split_indices"] == expected_split
        assert torch.all(packed.sequences[:, 0::38] == 10_000)
        assert torch.all(packed.sequences[:, 37::38] == 10_001)
        assert torch.all(packed.targets[:, 0::38] == -100)
        assert torch.all(packed.targets[:, 37::38] == -100)
        assert int((packed.targets[expected_split["train"]] != -100).sum()) == 19_975 * 972
        assert int(NCAWindows(packed, "train", 19_975).valid.sum()) == 19_715_325
        for i in sorted(set([0, len(rows)-1] + list(range(0, len(rows), 200)))):
            x, y = pack_one(rows[i][6], num_colors=10, patch_size=2)
            assert torch.equal(packed.sequences[i], x)
            assert torch.equal(packed.targets[i], y)
            assert abs(gzip_ratio(rows[i][6], 10, 2)-rows[i][3]) < 1e-12
    receipt = {
        "schema": "nca-boosted-sweep-audit-v1", "attempts": end,
        "selected_by_tier": {name: len(observed[i]) for i, name in enumerate(sweep.TIERS)},
        "counts_by_tier": counts.tolist(),
        "cross_tier_rule_overlap": len(arm_rules["top_25"] & arm_rules["top_10"]),
        "sampling_config_sha256": sweep.source.sha(config_path),
        "packed_sha256_by_tier": {name: sweep.source.sha(root / name / "packed.pt") for name in sweep.TIERS},
        "cutoffs_by_gzip_bin": sweep.CUTOFFS.tolist(),
        "contract": "int32 [20231,1026], original markers/masks, 19715325 usable train labels, pack_one spot checks",
    }
    sweep.atomic_json(root / "audit.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
