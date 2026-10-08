"""Read-only audit of a candidate against the exact reference train/val selection.

Run: uv run --all-packages python progress/nca_search/audit_iteration_04.py
Pass --candidate iteration-06-high-transition-mi to audit iteration 06.
The feature screen deterministically takes every 20th selected world per
split and 0.02-wide gzip bin, preserving identical sampled bin counts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from screen_existing import features
from screen_transformer_targets import token_features
from sampleeff_nca_pretraining.packing import PackedDataset, pack_one


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "runs/nca-100k-regimes"
SEARCH = REPO / "runs/nca-data-search"
REFERENCE = REPO / "runs/1m-nca-transfer-v2/data/nca/40_50/packed.pt"
LIMIT = (19_975, 256)
STRIDE = 20
KEYS = ("gzip", "change", "spatial_excess_agreement", "transition_mutual_info",
        "patch_entropy", "rare_patch_mass", "horizontal_copy_excess",
        "horizontal_patch_copy", "copy_chance", "context_repeat_256",
        "late_patch_novelty", "lag2_recurrence", "color_entropy")


def ratio_bin(x: float) -> int:
    return min(4, int((x - .4) / .02))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sampled_features(states: np.ndarray, ratio: float) -> dict[str, float]:
    return {**features(states, ratio), **token_features(states)}


def read_candidate(candidate: Path):
    records = []
    samples = []
    by_split = [0, 0]
    by_bin = np.zeros((2, 5), dtype=np.int64)
    for path in sorted((candidate / "chunks").glob("chunk-*.npz")):
        with np.load(path) as data:
            for i in range(len(data["split"])):
                split = int(data["split"][i])
                ordinal = by_split[split]
                by_split[split] += 1
                seed = tuple(map(int, data["rule_seed"][i]))
                ratio = float(data["gzip_ratio"][i])
                bin_ = ratio_bin(ratio)
                bin_ordinal = int(by_bin[split, bin_])
                by_bin[split, bin_] += 1
                records.append((split, seed, ratio, int(data["attempt"][i])))
                if bin_ordinal % STRIDE == 0:
                    samples.append((len(records) - 1,
                                    sampled_features(data["states"][i], ratio),
                                    data["states"][i].copy()))
    assert tuple(by_split) == LIMIT
    return records, samples


def read_reference():
    records = []
    packed_indices = []
    by_split = [0, 0]
    by_bin = np.zeros((2, 5), dtype=np.int64)
    sample_rows = []
    packed_index = 0
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for row in json.loads(path.read_text())["accepted"]:
            if row["band"] != "40_50":
                continue
            split = {"train": 0, "validation": 1, "test": 2}[row["split"]]
            if split < 2 and by_split[split] < LIMIT[split]:
                ordinal = by_split[split]
                by_split[split] += 1
                seed = tuple(row["rule_seed"])
                ratio = float(row["gzip_ratio"])
                bin_ = ratio_bin(ratio)
                bin_ordinal = int(by_bin[split, bin_])
                by_bin[split, bin_] += 1
                records.append((split, seed, ratio, int(row["attempt"])))
                packed_indices.append(packed_index)
                if bin_ordinal % STRIDE == 0:
                    sample_rows.append((len(records) - 1, packed_index, row))
            packed_index += 1
    assert tuple(by_split) == LIMIT
    grouped = defaultdict(list)
    for item in sample_rows:
        grouped[item[2]["dataset"]].append(item)
    samples = []
    for dataset, items in grouped.items():
        states = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for record_index, pack_index, row in items:
            world = states[row["row"]]
            samples.append((record_index, pack_index,
                            sampled_features(world, row["gzip_ratio"]),
                            np.asarray(world)))
    samples.sort(key=lambda row: row[0])
    return records, samples, packed_indices


def describe(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9))}


def report(candidate: Path):
    c_records, c_samples = read_candidate(candidate)
    r_records, r_samples, r_packed_indices = read_reference()
    c_gzip = np.array([record[2] for record in c_records])
    r_gzip = np.array([record[2] for record in r_records])
    c_keys = {(split, seed) for split, seed, _, _ in c_records}
    r_keys = {(split, seed) for split, seed, _, _ in r_records}
    assert len(c_keys) == len(c_records) and len(r_keys) == len(r_records)
    overlap = c_keys & r_keys
    print("selected", len(c_records), len(r_records), "sampled", len(c_samples), len(r_samples))
    print("overlap total", len(overlap), "train", sum(s == 0 for s, _ in overlap),
          "validation", sum(s == 1 for s, _ in overlap))
    print("attempt ranges candidate", min(v[3] for v in c_records), max(v[3] for v in c_records),
          "reference", min(v[3] for v in r_records), max(v[3] for v in r_records))
    print("gzip all", "candidate", describe(c_gzip), "reference", describe(r_gzip))
    print("gzip bins: bin candidate_n reference_n candidate_mean reference_mean delta")
    for b in range(5):
        a = c_gzip[np.array([ratio_bin(x) == b for x in c_gzip])]
        z = r_gzip[np.array([ratio_bin(x) == b for x in r_gzip])]
        print(b, len(a), len(z), float(a.mean()), float(z.mean()), float(a.mean()-z.mean()))
    # Exact empirical CDF gap; captures shape differences beyond mean matching.
    thresholds = np.unique(np.r_[c_gzip, r_gzip])
    cdf_gap = np.max(np.abs(np.searchsorted(np.sort(c_gzip), thresholds, side="right")/len(c_gzip)
                            - np.searchsorted(np.sort(r_gzip), thresholds, side="right")/len(r_gzip)))
    print("gzip KS empirical CDF gap", float(cdf_gap))
    print("sample features: key candidate_mean reference_mean delta naive_se candidate_p10 candidate_p90 reference_p10 reference_p90")
    for key in KEYS:
        c = np.array([row[1][key] for row in c_samples])
        r = np.array([row[2][key] for row in r_samples])
        naive_se = np.sqrt(c.var(ddof=1)/len(c) + r.var(ddof=1)/len(r))
        print(key, float(c.mean()), float(r.mean()), float(c.mean()-r.mean()), float(naive_se),
              float(np.quantile(c,.1)), float(np.quantile(c,.9)),
              float(np.quantile(r,.1)), float(np.quantile(r,.9)))
    # Full packed tensor contract and sampled raw-world tokenizer checks.
    c = PackedDataset.load(candidate / "packed.pt")
    r = PackedDataset.load(REFERENCE)
    print("sha candidate", sha(candidate / "packed.pt"))
    print("packed", tuple(c.sequences.shape), str(c.sequences.dtype),
          tuple(c.targets.shape), str(c.targets.dtype), "reference",
          tuple(r.sequences.shape), str(r.sequences.dtype),
          tuple(r.targets.shape), str(r.targets.dtype))
    for label, packed, samples in (("candidate", c, c_samples), ("reference", r, r_samples)):
        records = c_records if label == "candidate" else r_records
        packed_indices = (list(range(len(c_records))) if label == "candidate"
                          else r_packed_indices)
        expected_split = [[packed_indices[i] for i, x in enumerate(records) if x[0] == split]
                          for split in (0,1)]
        actual = packed.metadata["split_indices"]
        assert actual["train"][:LIMIT[0]] == expected_split[0]
        assert actual["validation"][:LIMIT[1]] == expected_split[1]
        seq = packed.sequences
        targets = packed.targets
        assert seq.ndim == 2 and seq.shape[1] == 1026 and targets.shape == seq.shape
        assert seq.dtype == torch.int32 and targets.dtype == torch.int32
        assert bool(torch.all(seq[:, 0::38] == 10000))
        assert bool(torch.all(seq[:, 37::38] == 10001))
        assert bool(torch.all(targets[:, 0::38] == -100))
        assert bool(torch.all(targets[:, 37::38] == -100))
        assert bool(torch.equal(seq[:, 1::38], targets[:, 1::38]))
        assert bool(torch.equal(seq[:, 36::38], targets[:, 36::38]))
        assert bool(torch.all((seq[:, 1::38] >= 0) & (seq[:, 1::38] < 10000)))
        assert bool(torch.all((seq[:, 36::38] >= 0) & (seq[:, 36::38] < 10000)))
        # Test actual 256-token NCAWindows target mask for a sample world.
        first = expected_split[0][0]
        window_y = seq[first, 1:257].clone().long()
        window_y[:37] = -100
        assert int((window_y != -100).sum()) == 219
        assert sum([219,256,256,256]) == 987
        checked = 0
        for row in samples[::max(1, len(samples)//8)][:8]:
            idx, world = (row[0], row[2]) if label == "candidate" else (row[1], row[3])
            seq_one, target_one = pack_one(world, num_colors=10, patch_size=2)
            assert torch.equal(seq[idx], seq_one)
            assert torch.equal(targets[idx], target_one)
            checked += 1
        print(label, "split/marker/mask checks passed; pack_one worlds", checked,
              "train usable", LIMIT[0] * 987)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="iteration-04-high-patch-copy")
    args = parser.parse_args()
    report(SEARCH / args.candidate)
