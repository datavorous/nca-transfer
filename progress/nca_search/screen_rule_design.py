"""Read saved initial states and trajectories; never generate or train.

Run: uv run --all-packages python progress/nca_search/screen_rule_design.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json

import numpy as np

from screen_existing import SOURCE, entropy, features, spearman, within_gzip_spearman
from screen_transformer_targets import token_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=5_000)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.first < 1 or args.stride < 1 or args.first % args.stride:
        parser.error("first and stride must be positive; first must be divisible by stride")

    selected = []
    seen = 0
    for receipt in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for row in json.loads(receipt.read_text())["accepted"]:
            if row["band"] != "40_50":
                continue
            if seen >= args.first:
                break
            if seen % args.stride == 0:
                selected.append(row)
            seen += 1
        if seen >= args.first:
            break
    assert seen == args.first and len(selected) == args.first // args.stride

    by_batch = defaultdict(list)
    for index, row in enumerate(selected):
        by_batch[row["dataset"]].append((index, row))
    rows = [None] * len(selected)
    for dataset, members in sorted(by_batch.items()):
        directory = SOURCE / dataset
        initials = np.load(directory / "initial-states-000000.npy", mmap_mode="r")
        worlds = np.load(directory / "trajectories-000000.npy", mmap_mode="r")
        for index, row in members:
            initial = np.asarray(initials[row["row"]])[..., 0]
            counts = np.bincount(initial.ravel(), minlength=10)
            trajectory = worlds[row["row"]]
            world = features(trajectory, row["gzip_ratio"])
            patches = token_features(trajectory)
            rows[index] = {
                "initial_entropy": entropy(counts),
                "initial_dominant_share": float(counts.max() / counts.sum()),
                "initial_neighbor_agreement": float(((initial == np.roll(initial, 1, axis=0)).mean() +
                                                       (initial == np.roll(initial, 1, axis=1)).mean()) / 2),
                "gzip": world["gzip"],
                "change": world["change"],
                "transition_mutual_info": world["transition_mutual_info"],
                "horizontal_copy_excess": patches["horizontal_copy_excess"],
                "late_minus_early_patch_entropy": patches["late_minus_early_patch_entropy"],
            }
    assert all(row is not None for row in rows)
    data = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    print(f"Worlds: {len(rows)} (every {args.stride}th of first {args.first} accepted 40–50 worlds)")
    print("initial property | p10 | median | p90 | outcome | rho | within-gzip-bin rho")
    for key in ("initial_entropy", "initial_dominant_share", "initial_neighbor_agreement"):
        q = np.quantile(data[key], [.1, .5, .9])
        for outcome in ("gzip", "change", "transition_mutual_info", "horizontal_copy_excess",
                        "late_minus_early_patch_entropy"):
            print(f"{key} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | {outcome} | "
                  f"{spearman(data[key], data[outcome]):+.3f} | "
                  f"{within_gzip_spearman(data[key], data[outcome], data['gzip']):+.3f}")


if __name__ == "__main__":
    main()
