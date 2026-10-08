"""Read-only common-support screen for a patch-copy mechanism probe.

Run from the repository root:
    uv run --all-packages python progress/nca_search/screen_patch_copy_matching.py

Uses the first 10,000 accepted 40–50 reference receipts, with every second
world by default. No trajectory generation or model training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json

import numpy as np

from screen_attention_lags import SOURCE, selected_receipts
from screen_existing import entropy
from sample_patch_replication import THRESHOLD, gzip_bin, horizontal_copy_excess


def world_features(states: np.ndarray, gzip: float) -> tuple[float, float, float, float]:
    s = states[..., 0].astype(np.uint8)
    transitions = np.bincount(
        (s[:-1].astype(np.int16) * 10 + s[1:]).ravel(), minlength=100
    ).reshape(10, 10)
    total = transitions.sum()
    conditional = sum(row.sum() / total * entropy(row) for row in transitions if row.sum())
    mi = entropy(transitions.sum(axis=0)) - conditional
    lag2 = float((s[2:] == s[:-2]).mean())
    return float(gzip), horizontal_copy_excess(states), lag2, float(mi)


def assess(x: np.ndarray, groups: np.ndarray, bands: np.ndarray, levels: int) -> dict:
    # Reference-derived nuisance quantile cuts, fixed for both arms.
    cuts = np.quantile(x[:, 2:4], np.linspace(0, 1, levels + 1)[1:-1], axis=0)
    lag_bins = np.searchsorted(cuts[:, 0], x[:, 2], side="right")
    mi_bins = np.searchsorted(cuts[:, 1], x[:, 3], side="right")
    cells = bands * levels * levels + lag_bins * levels + mi_bins
    high_keep = []
    low_keep = []
    count_cells = 0
    for cell in range(5 * levels * levels):
        hi = np.flatnonzero((cells == cell) & groups)
        lo = np.flatnonzero((cells == cell) & ~groups)
        n = min(len(hi), len(lo))
        if n:
            count_cells += 1
            high_keep.extend(hi[:n])
            low_keep.extend(lo[:n])
    hi = x[high_keep]
    lo = x[low_keep]
    pooled_sd = np.sqrt((x[groups, 2:4].var(axis=0) + x[~groups, 2:4].var(axis=0)) / 2)
    return {
        "nuisance_quantiles": levels,
        "lag2_cuts": cuts[:, 0].tolist(),
        "mi_cuts_bits": cuts[:, 1].tolist(),
        "occupied_overlap_cells": count_cells,
        "possible_cells": 5 * levels * levels,
        "pairs": len(high_keep),
        "retained_fraction_per_arm": [len(high_keep) / groups.sum(), len(low_keep) / (~groups).sum()],
        "postmatch_high_minus_low": (hi.mean(axis=0) - lo.mean(axis=0)).tolist(),
        "postmatch_nuisance_smd": ((hi[:, 2:4].mean(axis=0) - lo[:, 2:4].mean(axis=0)) / pooled_sd).tolist(),
        "high_mean": hi.mean(axis=0).tolist(),
        "low_mean": lo.mean(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=10_000)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    if args.first <= 0 or args.stride <= 0 or args.first % args.stride:
        parser.error("first and stride must be positive; first divisible by stride")
    receipts = selected_receipts(args.first, args.stride)
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, receipt in enumerate(receipts):
        grouped[receipt["dataset"]].append((index, receipt))
    rows: list[tuple[float, float, float, float] | None] = [None] * len(receipts)
    for dataset, members in sorted(grouped.items()):
        worlds = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for index, receipt in members:
            rows[index] = world_features(worlds[receipt["row"]], receipt["gzip_ratio"])
    assert all(row is not None for row in rows)
    x = np.asarray(rows, dtype=np.float64)
    groups = x[:, 1] >= THRESHOLD
    bands = np.asarray([gzip_bin(g) for g in x[:, 0]], dtype=np.int8)
    assert np.all((bands >= 0) & (bands <= 4))
    out = {
        "source": f"first {args.first} accepted 40–50 reference worlds, stride {args.stride}",
        "sample_worlds": len(x),
        "high_copy_threshold": THRESHOLD,
        "feature_order": ["gzip", "horizontal_copy_excess", "cell_lag2_recurrence", "cell_transition_mi_bits"],
        "high_count": int(groups.sum()),
        "low_count": int((~groups).sum()),
        "high_mean": x[groups].mean(axis=0).tolist(),
        "low_mean": x[~groups].mean(axis=0).tolist(),
        "raw_nuisance_smd": ((x[groups, 2:4].mean(axis=0) - x[~groups, 2:4].mean(axis=0)) /
                             np.sqrt((x[groups, 2:4].var(axis=0) + x[~groups, 2:4].var(axis=0)) / 2)).tolist(),
        "high_share_by_gzip_bin": [float(groups[bands == b].mean()) for b in range(5)],
        "matching": [assess(x, groups, bands, levels) for levels in (3, 5)],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
