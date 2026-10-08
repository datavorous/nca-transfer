"""Read-only screen of color-shifted patch relations in saved NCA worlds.

Run: uv run --all-packages python progress/nca_search/screen_shifted_patch_relations.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from screen_attention_lags import SOURCE, selected_receipts
from screen_existing import spearman, within_gzip_spearman
from screen_transformer_targets import patch_grid


def shifted_relations(states: np.ndarray, gzip_ratio: float) -> dict[str, float]:
    patches = patch_grid(states)
    # If each of four colors is shifted by the same nonzero amount modulo 10,
    # the three differences from the first cell are identical in both patches.
    digits = np.stack([(patches // (10 ** i)) % 10 for i in range(4)], axis=-1)
    shape = (((digits[..., 1:] - digits[..., :1]) % 10)
             @ np.array([1, 10, 100], dtype=np.int32))
    left_shape, right_shape = shape[:, :, :-1], shape[:, :, 1:]
    left_id, right_id = patches[:, :, :-1], patches[:, :, 1:]
    horizontal_shift = (left_shape == right_shape) & (left_id != right_id)
    # Exact finite-population chance of a shifted-shape match for two patch
    # positions drawn without replacement from the same world.
    ids = np.bincount(patches.ravel(), minlength=10_000).astype(np.int64)
    shapes = np.bincount(shape.ravel(), minlength=1_000).astype(np.int64)
    chance = ((shapes * (shapes - 1)).sum() - (ids * (ids - 1)).sum()) / (972 * 971)
    # Targets in frames 1..26, with predecessor in same 256-token window.
    pos = 38 * np.arange(27)[:, None, None] + 1 + 6 * np.arange(6)[None, :, None] + np.arange(6)[None, None, :]
    target = np.broadcast_to(pos[:, :, 1:], horizontal_shift.shape)
    visible = (target >= 38) & (target <= 1024) & ((target - 1) // 256 == (target - 2) // 256)
    # Temporal nonidentity shape recurrence, at the same site one frame ago.
    temporal_shift = (shape[1:] == shape[:-1]) & (patches[1:] != patches[:-1])
    target_t = pos[1:]
    visible_t = (target_t >= 38) & (target_t <= 1024) & ((target_t - 1) // 256 == (target_t - 39) // 256)
    temporal_chance = np.mean([
        ((np.bincount(shape[t].ravel(), minlength=1_000) *
           np.bincount(shape[t - 1].ravel(), minlength=1_000)).sum() -
          (np.bincount(patches[t].ravel(), minlength=10_000) *
           np.bincount(patches[t - 1].ravel(), minlength=10_000)).sum()) / (36 * 36)
        for t in range(1, 27)
    ])
    exact_chance = (ids * (ids - 1)).sum() / (972 * 971)
    return {
        "gzip": float(gzip_ratio),
        "cell_change": float((states[1:] != states[:-1]).mean()),
        "horizontal_copy_excess": float((left_id == right_id).mean() - exact_chance),
        "horizontal_shift_rate": float(horizontal_shift.mean()),
        "horizontal_shift_excess": float(horizontal_shift.mean() - chance),
        "horizontal_shift_visible_excess": float(horizontal_shift[visible].mean() - chance),
        "horizontal_shift_visible_share": float(visible.mean()),
        "temporal_shift_rate": float(temporal_shift.mean()),
        "temporal_shift_excess": float(temporal_shift.mean() - temporal_chance),
        "temporal_shift_visible_share": float(visible_t.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=5_000)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.first <= 0 or args.stride <= 0 or args.first % args.stride:
        parser.error("first and stride must be positive; first divisible by stride")
    receipts = selected_receipts(args.first, args.stride)
    grouped = defaultdict(list)
    for i, row in enumerate(receipts):
        grouped[row["dataset"]].append((i, row))
    results = [None] * len(receipts)
    for dataset, members in sorted(grouped.items()):
        worlds = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for i, receipt in members:
            results[i] = shifted_relations(worlds[receipt["row"]], receipt["gzip_ratio"])
    assert all(row is not None for row in results)
    values = {key: np.asarray([row[key] for row in results]) for key in results[0]}
    print(f"worlds={len(results)} from first {args.first} accepted, stride={args.stride}")
    for key in ("horizontal_shift_rate", "horizontal_shift_excess",
                "horizontal_shift_visible_excess", "temporal_shift_rate", "temporal_shift_excess"):
        x = values[key]
        print(f"{key}: mean={x.mean():+.6f}, p10/median/p90={np.quantile(x,[.1,.5,.9])}, "
              f"rho_gzip={spearman(x,values['gzip']):+.3f}, "
              f"rho_copy={spearman(x,values['horizontal_copy_excess']):+.3f}, "
              f"rho_change={spearman(x,values['cell_change']):+.3f}")
    x = values["horizontal_shift_excess"]
    print(f"visible horizontal share={values['horizontal_shift_visible_share'].mean():.4f}; "
          f"visible temporal share={values['temporal_shift_visible_share'].mean():.4f}")
    print(f"within-gzip rho shift-vs-copy={within_gzip_spearman(x,values['horizontal_copy_excess'],values['gzip']):+.3f}")
    cutoff = float(np.median(x))
    high = x >= cutoff
    print(f"horizontal shift excess median={cutoff:+.9f}; high n={high.sum()}, low n={(~high).sum()}")
    for b in range(5):
        mask = np.clip(((values['gzip'] - .4) / .02).astype(int), 0, 4) == b
        print(f"gzip bin {b} n={mask.sum()}, high share={high[mask].mean():.3f}")
    for label, mask in (("high", high), ("low", ~high)):
        print(f"{label}: shift={x[mask].mean():+.6f}, gzip={values['gzip'][mask].mean():.5f}, "
              f"copy={values['horizontal_copy_excess'][mask].mean():+.6f}, "
              f"change={values['cell_change'][mask].mean():.5f}")


if __name__ == "__main__":
    main()
