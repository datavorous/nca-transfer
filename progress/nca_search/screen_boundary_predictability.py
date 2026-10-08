"""Read-only held-out spatial boundary prediction on saved 40–50 worlds.

Run from repository root:
    uv run --all-packages python progress/nca_search/screen_boundary_predictability.py

No generation or model training. The fitted 10x10 table is a diagnostic oracle:
the Transformer is not given a separate table for each world.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from screen_attention_lags import selected_receipts
from screen_existing import spearman
from screen_transformer_targets import SOURCE, patch_grid


def measure(states: np.ndarray, gzip_ratio: float) -> dict[str, float]:
    cells = states[..., 0].astype(np.int32)
    patches = patch_grid(states)
    # Patches have shape [frame, patch-row, patch-col, inner-row, inner-col].
    grid = cells.reshape(27, 6, 2, 6, 2).transpose(0, 1, 3, 2, 4)
    # Across horizontally adjacent patches, predict the leftmost cell in the
    # target patch using the immediately adjacent cell in the source patch.
    left = grid[:, :, :-1, :, 1]
    right = grid[:, :, 1:, :, 0]
    source_train, target_train = left[:13].ravel(), right[:13].ravel()
    source_test, target_test = left[13:].ravel(), right[13:].ravel()
    counts = np.bincount(10 * source_train + target_train, minlength=100).reshape(10, 10)
    # Jeffreys smoothing keeps rare colours finite and makes held-out score
    # comparable across worlds with different colour marginals.
    conditional = (counts + 0.5) / (counts.sum(axis=1, keepdims=True) + 5)
    marginal = (counts.sum(axis=0) + 0.5) / (counts.sum() + 5)
    gain = np.log2(conditional[source_test, target_test] / marginal[target_test]).mean()
    # 27*6*5 horizontal patch pairs; only scored frame targets 1..26 and
    # within-window predecessor visibility are counted in this copy metric.
    pos = 38 * np.arange(27)[:, None, None] + 1 + 6 * np.arange(6)[None, :, None] + np.arange(6)[None, None, :]
    target_pos = np.broadcast_to(pos[:, :, 1:], (27, 6, 5))
    source_pos = target_pos - 1
    first = ((target_pos - 1) // 256) * 256
    visible = (target_pos >= 38) & (target_pos <= 1024) & (source_pos >= first)
    exact_copy = (patches[:, :, 1:] == patches[:, :, :-1])[visible].mean()
    counts_patch = np.bincount(patches.ravel(), minlength=10_000).astype(np.int64)
    chance = (counts_patch * (counts_patch - 1)).sum() / (972 * 971)
    return {
        "boundary_gain_bits_per_cell": float(gain),
        "horizontal_copy_excess": float((patches[:, :, 1:] == patches[:, :, :-1]).mean() - chance),
        "visible_horizontal_copy": float(exact_copy),
        "gzip": float(gzip_ratio),
        "change": float((cells[1:] != cells[:-1]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=5_000)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.first <= 0 or args.stride <= 0 or args.first % args.stride:
        parser.error("first and stride must be positive; first must be divisible by stride")
    receipts = selected_receipts(args.first, args.stride)
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, receipt in enumerate(receipts):
        grouped[receipt["dataset"]].append((index, receipt))
    measured = [None] * len(receipts)
    for dataset, members in sorted(grouped.items()):
        worlds = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for index, receipt in members:
            measured[index] = measure(worlds[receipt["row"]], receipt["gzip_ratio"])
    assert all(item is not None for item in measured)
    cols = {key: np.array([item[key] for item in measured]) for key in measured[0]}
    gain = cols["boundary_gain_bits_per_cell"]
    cutoff = np.median(gain)
    high = gain >= cutoff
    print(f"n={len(receipts)}, every {args.stride}th of first {args.first} accepted 40–50 worlds")
    print("Train spatial transition table on frames 0–12; score frames 13–26.")
    print(f"gain bits/boundary cell: mean={gain.mean():+.5f}, p10/median/p90={np.quantile(gain,[.1,.5,.9])}")
    print(f"positive gain: {(gain > 0).mean():.3f}")
    print(f"rho gain with gzip/change/horizontal-copy-excess: "
          f"{spearman(gain,cols['gzip']):+.3f}/{spearman(gain,cols['change']):+.3f}/"
          f"{spearman(gain,cols['horizontal_copy_excess']):+.3f}")
    for band in range(5):
        in_bin = np.clip(((cols["gzip"] - .4) / .02).astype(int), 0, 4) == band
        print(f"bin {band}: n={in_bin.sum()}, high-share={high[in_bin].mean():.3f}")
    for label, mask in (("high", high), ("low", ~high)):
        print(f"{label}: n={mask.sum()}, gain={gain[mask].mean():+.5f}, gzip={cols['gzip'][mask].mean():.5f}, "
              f"change={cols['change'][mask].mean():.5f}, horizontal-excess={cols['horizontal_copy_excess'][mask].mean():+.5f}")
    print(f"median threshold (sample only): {cutoff:.12f}")


if __name__ == "__main__":
    main()
