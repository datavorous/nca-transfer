"""Read-only screen of two-frame same-site patch recurrence in saved NCA worlds.

Run from the repository root:
    uv run --all-packages python progress/nca_search/screen_temporal_target.py
    uv run --all-packages python progress/nca_search/screen_temporal_target.py --first 25000 --stride 1
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json

import numpy as np

from screen_attention_lags import SOURCE, selected_receipts
from screen_existing import spearman, within_gzip_spearman
from screen_transformer_targets import patch_grid


FRAME_POS = np.arange(27)[:, None] * 38 + 1 + np.arange(36)[None, :]
TARGET76 = FRAME_POS[2:]
SOURCE76 = FRAME_POS[:-2]
# NCAWindows takes x[0:256], x[256:512], ... and y=x shifted by one.
# A target at sequence position q has context beginning at floor((q-1)/256)*256.
VISIBLE76 = SOURCE76 >= ((TARGET76 - 1) // 256) * 256
assert TARGET76.min() >= 38 and TARGET76.max() <= 1024


def features(states: np.ndarray, gzip: float) -> dict[str, float]:
    patches = patch_grid(states).reshape(27, 36)
    freq = np.zeros((27, 10_000), dtype=np.float32)
    for t in range(27):
        ids, counts = np.unique(patches[t], return_counts=True)
        freq[t, ids] = counts / 36
    # Independent frame rearrangements preserve each frame's patch-ID counts.
    chance76 = np.sum(freq[2:] * freq[:-2], axis=1)
    chance38 = np.sum(freq[1:] * freq[:-1], axis=1)
    equal76 = patches[2:] == patches[:-2]
    equal38 = patches[1:] == patches[:-1]
    excess76 = equal76.astype(np.float32) - chance76[:, None]
    excess38 = equal38.astype(np.float32) - chance38[:, None]
    flat = patches.ravel()
    counts = np.bincount(flat, minlength=10_000).astype(np.int64)
    horizontal_chance = np.sum(counts * (counts - 1)) / (972 * 971)
    horizontal_copy = float((patches.reshape(27, 6, 6)[:, :, 1:] ==
                             patches.reshape(27, 6, 6)[:, :, :-1]).mean())
    return {
        "gzip": gzip,
        "change": float(np.mean(states[1:] != states[:-1])),
        "horizontal_excess": horizontal_copy - horizontal_chance,
        "lag38_excess": float(excess38.mean()),
        "lag76_copy": float(equal76.mean()),
        "lag76_excess": float(excess76.mean()),
        "visible76_copy": float(equal76[VISIBLE76].mean()),
        "visible76_excess": float(excess76[VISIBLE76].mean()),
        "invisible76_excess": float(excess76[~VISIBLE76].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=5_000)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.first <= 0 or args.stride <= 0 or args.first % args.stride:
        parser.error("first and stride must be positive, with first divisible by stride")
    receipts = selected_receipts(args.first, args.stride)
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, receipt in enumerate(receipts):
        grouped[receipt["dataset"]].append((index, receipt))
    rows: list[dict[str, float] | None] = [None] * len(receipts)
    for dataset, members in sorted(grouped.items()):
        worlds = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for index, receipt in members:
            rows[index] = features(worlds[receipt["row"]], receipt["gzip_ratio"])
    assert all(row is not None for row in rows)
    values = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    print(f"worlds={len(rows)} first={args.first} stride={args.stride}")
    print(f"lag76 eligible pairs/world={VISIBLE76.size}; visible={VISIBLE76.sum()} ({VISIBLE76.mean():.6f})")
    print("feature | p10 | median | p90 | rho_gzip | rho_change | rho_horizontal | rho_lag38")
    for key in ("lag76_copy", "lag76_excess", "visible76_copy", "visible76_excess", "invisible76_excess"):
        x = values[key]
        q = np.quantile(x, [.1, .5, .9])
        print(f"{key} | {q[0]:+.6f} | {q[1]:+.6f} | {q[2]:+.6f} | "
              f"{spearman(x, values['gzip']):+.3f} | {spearman(x, values['change']):+.3f} | "
              f"{spearman(x, values['horizontal_excess']):+.3f} | "
              f"{spearman(x, values['lag38_excess']):+.3f}")
    x = values["visible76_excess"]
    print("within-bin rho visible76 with gzip/change/horizontal/lag38: " +
          " / ".join(f"{within_gzip_spearman(x, values[key], values['gzip']):+.3f}"
                     for key in ("gzip", "change", "horizontal_excess", "lag38_excess")))
    print(f"rho visible76 vs all76={spearman(x, values['lag76_excess']):+.3f}; "
          f"vs invisible76={spearman(x, values['invisible76_excess']):+.3f}")
    cut = float(np.median(x))
    high = x >= cut
    print(f"high selector >= median {cut!r}; selected={high.sum()} / {len(x)}")
    print("feature | high mean | low mean | delta")
    for key in ("gzip", "change", "horizontal_excess", "lag38_excess", "lag76_excess", "invisible76_excess"):
        y = values[key]
        print(f"{key} | {y[high].mean():+.6f} | {y[~high].mean():+.6f} | "
              f"{y[high].mean()-y[~high].mean():+.6f}")


if __name__ == "__main__":
    main()
