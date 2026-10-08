"""Read-only transition-information screen on saved 40–50 trajectories.

Run: uv run --all-packages python progress/nca_search/screen_transition_info_target.py
The deterministic default samples every fifth of the first 5,000 accepted worlds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from screen_existing import entropy, spearman, within_gzip_spearman
from screen_transformer_targets import patch_grid


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/nca-100k-regimes"


def receipts(first: int, stride: int) -> list[dict]:
    selected = []
    seen = 0
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for row in json.loads(path.read_text())["accepted"]:
            if row["band"] != "40_50":
                continue
            if seen >= first:
                break
            if seen % stride == 0:
                selected.append(row)
            seen += 1
        if seen >= first:
            break
    if seen != first:
        raise ValueError(f"found {seen} accepted worlds, requested {first}")
    return selected


def measure(states: np.ndarray, gzip: float) -> dict[str, float]:
    cells = states[..., 0].astype(np.int32)
    patches = patch_grid(states)
    # The exact prior-frame same-site patch is 38 packed positions back.
    # A target is scored at positions 38..1024. The source must be inside
    # that target's nonoverlapping 256-token context window.
    positions = (38 * np.arange(27)[:, None, None]
                 + 1 + 6 * np.arange(6)[None, :, None]
                 + np.arange(6)[None, None, :])
    positions = np.broadcast_to(positions, patches.shape)
    target = positions[14:]
    source = positions[13:-1]
    window_start = ((target - 1) // 256) * 256 + 1
    visible = (target <= 1024) & (source >= window_start)

    # Estimate first-order cell transitions from transitions 0->1 .. 12->13.
    train_pairs = 10 * cells[:13].ravel() + cells[1:14].ravel()
    train_counts = np.bincount(train_pairs, minlength=100).reshape(10, 10)
    next_counts = train_counts.sum(axis=0)
    train_cond = sum(row.sum() / train_counts.sum() * entropy(row)
                     for row in train_counts if row.sum())
    train_mi = entropy(next_counts) - train_cond
    transition = (train_counts + 0.5) / (train_counts.sum(axis=1, keepdims=True) + 5)
    marginal = (next_counts + 0.5) / (next_counts.sum() + 5)

    prior_cells = cells[13:-1].reshape(13, 6, 2, 6, 2).transpose(0, 1, 3, 2, 4).reshape(13, 6, 6, 4)
    next_cells = cells[14:].reshape(13, 6, 2, 6, 2).transpose(0, 1, 3, 2, 4).reshape(13, 6, 6, 4)
    baseline_bits = -np.log2(marginal[next_cells]).sum(axis=-1)
    markov_bits = -np.log2(transition[prior_cells, next_cells]).sum(axis=-1)
    # The copy mixture is a normalized distribution on the 10,000 patch IDs.
    train_copy = float((patches[1:14] == patches[:13]).mean())
    baseline_prob = np.exp2(-baseline_bits)
    copy_prob = train_copy * (patches[14:] == patches[13:-1]) + (1 - train_copy) * baseline_prob
    copy_bits = -np.log2(copy_prob)

    all_pairs = 10 * cells[:-1].ravel() + cells[1:].ravel()
    counts = np.bincount(all_pairs, minlength=100).reshape(10, 10)
    total = counts.sum()
    cond = sum(row.sum() / total * entropy(row) for row in counts if row.sum())
    full_mi = entropy(counts.sum(axis=0)) - cond
    token_counts = np.bincount(patches.ravel(), minlength=10_000).astype(np.int64)
    patch_chance = (token_counts * (token_counts - 1)).sum() / (972 * 971)
    horizontal = float((patches[:, :, 1:] == patches[:, :, :-1]).mean() - patch_chance)
    return {
        "gzip": float(gzip),
        "change": float((cells[1:] != cells[:-1]).mean()),
        "horizontal_excess": horizontal,
        "cell_mi_full_bits": float(full_mi),
        "cell_mi_train_bits": float(train_mi),
        "patch_markov_gain_bits": float((baseline_bits - markov_bits)[visible].mean()),
        "patch_copy_gain_bits": float((baseline_bits - copy_bits)[visible].mean()),
        "patch_base_loss_bits": float(baseline_bits[visible].mean()),
        "patch_markov_loss_bits": float(markov_bits[visible].mean()),
        "patch_copy_loss_bits": float(copy_bits[visible].mean()),
        "visible_targets": int(visible.sum()),
        "train_patch_copy": train_copy,
        "test_patch_copy": float((patches[14:] == patches[13:-1])[visible].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=5_000)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.first <= 0 or args.stride <= 0 or args.first % args.stride:
        parser.error("first and stride must be positive and first divisible by stride")
    rows = receipts(args.first, args.stride)
    by_dataset = defaultdict(list)
    for i, row in enumerate(rows):
        by_dataset[row["dataset"]].append((i, row))
    result = [None] * len(rows)
    for dataset, members in sorted(by_dataset.items()):
        worlds = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for i, row in members:
            result[i] = measure(worlds[row["row"]], row["gzip_ratio"])
    assert all(row is not None for row in result)
    x = {key: np.array([row[key] for row in result]) for key in result[0]}
    print(f"n={len(rows)}, first={args.first}, stride={args.stride}, visible targets/world={np.unique(x['visible_targets'])}")
    print("metric | mean | p10 | median | p90 | rho(cell MI) | rho(gzip) | rho(change) | rho(horizontal)")
    for name in ("cell_mi_full_bits", "cell_mi_train_bits", "patch_base_loss_bits", "patch_markov_loss_bits",
                 "patch_copy_loss_bits", "patch_markov_gain_bits", "patch_copy_gain_bits",
                 "train_patch_copy", "test_patch_copy"):
        v = x[name]
        q = np.quantile(v, [.1, .5, .9])
        print(f"{name} | {v.mean():+.5f} | {q[0]:+.5f} | {q[1]:+.5f} | {q[2]:+.5f} | "
              f"{spearman(v,x['cell_mi_full_bits']):+.3f} | {spearman(v,x['gzip']):+.3f} | "
              f"{spearman(v,x['change']):+.3f} | {spearman(v,x['horizontal_excess']):+.3f}")
    print("\nkey within-gzip-bin rank correlations")
    for a, b in (("cell_mi_full_bits", "patch_markov_gain_bits"),
                 ("cell_mi_train_bits", "patch_markov_gain_bits"),
                 ("cell_mi_full_bits", "patch_copy_gain_bits"),
                 ("patch_markov_gain_bits", "horizontal_excess"),
                 ("patch_markov_gain_bits", "change")):
        print(f"{a} x {b}: {within_gzip_spearman(x[a],x[b],x['gzip']):+.3f}")
    median = np.median(x["cell_mi_full_bits"])
    print(f"\nfull-sample cell MI median (unrounded): {median:.12f} bits")
    bins = np.clip(((x["gzip"] - .40) / .02).astype(int), 0, 4)
    for band in range(5):
        in_band = bins == band
        print(f"gzip band {band}: n={in_band.sum()}, high-MI share={(x['cell_mi_full_bits'][in_band]>=median).mean():.4f}")
    for label, mask in (("low MI", x["cell_mi_full_bits"] < median),
                        ("high MI", x["cell_mi_full_bits"] >= median)):
        print(f"{label}: n={mask.sum()}, mean patch Markov gain={x['patch_markov_gain_bits'][mask].mean():+.5f}, "
              f"positive gain share={(x['patch_markov_gain_bits'][mask]>0).mean():.3f}, "
              f"mean patch copy gain={x['patch_copy_gain_bits'][mask].mean():+.5f}, "
              f"mean gzip={x['gzip'][mask].mean():.5f}, change={x['change'][mask].mean():.5f}, "
              f"horizontal excess={x['horizontal_excess'][mask].mean():+.5f}")


if __name__ == "__main__":
    main()
