"""Read-only lag screen on saved 40–50 NCA worlds.

Run from repo root:
    uv run --all-packages python progress/nca_search/screen_attention_lags.py

No generation, model loading, or training. World sampling follows accepted
receipt order. All copy baselines preserve each frame's patch-ID frequencies.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from screen_existing import spearman
from screen_transformer_targets import patch_grid


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/nca-100k-regimes"
LAGS = (1, 6, 37, 38, 39, 76, 114, 152, 266)


def selected_receipts(first: int, stride: int) -> list[dict]:
    rows = []
    seen = 0
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for row in json.loads(path.read_text())["accepted"]:
            if row["band"] != "40_50":
                continue
            if seen >= first:
                break
            if seen % stride == 0:
                rows.append(row)
            seen += 1
        if seen >= first:
            break
    if seen != first:
        raise ValueError(f"found only {seen} accepted 40–50 worlds")
    return rows


def lag_features(states: np.ndarray) -> dict[str, float]:
    p = patch_grid(states)
    # In the packed stream, each frame is START, 36 raster patches, END.
    positions = np.arange(27)[:, None, None] * 38 + 1 + np.arange(6)[None, :, None] * 6 + np.arange(6)[None, None, :]
    pos = np.broadcast_to(positions, p.shape).ravel()
    token = p.ravel()
    frame = np.repeat(np.arange(27), 36)
    frame_counts = np.zeros((27, 10_000), dtype=np.int16)
    for t in range(27):
        frame_counts[t] = np.bincount(p[t].ravel(), minlength=10_000)
    chance_same = (frame_counts.astype(np.int32) * (frame_counts.astype(np.int32) - 1)).sum(axis=1) / (36 * 35)
    frame_freq = frame_counts.astype(np.float64) / 36
    output: dict[str, float] = {}
    for lag in LAGS:
        if lag == 1:
            current = np.arange(972).reshape(27, 6, 6)[:, :, 1:].ravel()
            previous = current - 1
        elif lag == 6:
            current = np.arange(972).reshape(27, 6, 6)[:, 1:, :].ravel()
            previous = current - 6
        else:
            # Look up exact packed positions, excluding markers and missing sites.
            lookup = {int(value): i for i, value in enumerate(pos)}
            current = np.array([i for i, value in enumerate(pos) if int(value - lag) in lookup], dtype=np.int32)
            previous = np.array([lookup[int(pos[i] - lag)] for i in current], dtype=np.int32)
        same_frame = frame[current] == frame[previous]
        chance = np.empty(len(current), dtype=np.float64)
        chance[same_frame] = chance_same[frame[current[same_frame]]]
        cross = np.flatnonzero(~same_frame)
        if len(cross):
            # One chance value for each ordered pair of frame indices.
            pairs = np.unique(np.stack((frame[current[cross]], frame[previous[cross]]), axis=1), axis=0)
            for target_frame, source_frame in pairs:
                mask = (frame[current] == target_frame) & (frame[previous] == source_frame)
                chance[mask] = np.sum(frame_freq[target_frame] * frame_freq[source_frame])
        equal = token[current] == token[previous]
        # NCAWindows uses sequence[1:1025], in four nonoverlapping 256-token
        # windows; only patch targets after the masked first frame are scored.
        target = pos[current]
        first = ((target - 1) // 256) * 256 + 1
        visible = (target >= 38) & (target <= 1024) & (pos[previous] >= first)
        output[f"lag{lag}_copy"] = float(equal.mean())
        output[f"lag{lag}_excess"] = float((equal - chance).mean())
        output[f"lag{lag}_visible_share"] = float(visible.mean())
        output[f"lag{lag}_visible_excess"] = float((equal[visible] - chance[visible]).mean()) if visible.any() else float("nan")
    output["aligned38_vs_neighbors"] = output["lag38_excess"] - (output["lag37_excess"] + output["lag39_excess"]) / 2
    output["lag76_vs_38"] = output["lag76_excess"] - output["lag38_excess"]
    output["lag114_vs_38"] = output["lag114_excess"] - output["lag38_excess"]
    output["cell_change"] = float(np.mean(states[1:] != states[:-1]))
    output["cell_lag2"] = float(np.mean(states[2:] == states[:-2]))
    output["cell_lag3"] = float(np.mean(states[3:] == states[:-3]))
    counts = np.bincount(token, minlength=10_000)
    chance_global = np.sum(counts.astype(np.int64) * (counts.astype(np.int64) - 1)) / (972 * 971)
    output["horizontal_copy_excess_global"] = output["lag1_copy"] - chance_global
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=5_000)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.first <= 0 or args.stride <= 0 or args.first % args.stride:
        parser.error("first and stride must be positive, with first divisible by stride")
    receipts = selected_receipts(args.first, args.stride)
    by_dataset = defaultdict(list)
    for index, row in enumerate(receipts):
        by_dataset[row["dataset"]].append((index, row))
    result = [None] * len(receipts)
    for dataset, members in sorted(by_dataset.items()):
        worlds = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for index, row in members:
            result[index] = {**lag_features(worlds[row["row"]]), "gzip": row["gzip_ratio"]}
    assert all(value is not None for value in result)
    values = {key: np.array([row[key] for row in result]) for key in result[0]}
    print(f"Worlds: {len(result)} (every {args.stride}th of first {args.first} accepted 40–50 worlds)")
    print("lag | copy median | excess median [p10,p90] | visible % | visible excess median | rho gzip | rho change | rho horizontal copy")
    for lag in LAGS:
        excess = values[f"lag{lag}_excess"]
        visible_excess = values[f"lag{lag}_visible_excess"]
        visible_display = f"{np.median(visible_excess):+.4f}" if np.isfinite(visible_excess).any() else "n/a"
        print(f"{lag} | {np.median(values[f'lag{lag}_copy']):.4f} | {np.median(excess):+.4f} [{np.quantile(excess,.1):+.4f},{np.quantile(excess,.9):+.4f}] | "
              f"{100*np.median(values[f'lag{lag}_visible_share']):.1f} | {visible_display} | "
              f"{spearman(excess, values['gzip']):+.3f} | {spearman(excess, values['cell_change']):+.3f} | "
              f"{spearman(excess, values['horizontal_copy_excess_global']):+.3f}")
    print("\nDerived | median [p10,p90] | rho gzip | rho change | rho horizontal copy")
    for key in ("aligned38_vs_neighbors", "lag76_vs_38", "lag114_vs_38"):
        x = values[key]
        print(f"{key} | {np.median(x):+.4f} [{np.quantile(x,.1):+.4f},{np.quantile(x,.9):+.4f}] | "
              f"{spearman(x, values['gzip']):+.3f} | {spearman(x, values['cell_change']):+.3f} | "
              f"{spearman(x, values['horizontal_copy_excess_global']):+.3f}")
    print("\nTemporal | rho cell lag2 | rho cell lag3 | rho lag38 | rho lag76")
    for key in ("lag38_excess", "lag76_excess", "lag114_excess", "lag152_excess"):
        x = values[key]
        print(f"{key} | {spearman(x,values['cell_lag2']):+.3f} | {spearman(x,values['cell_lag3']):+.3f} | "
              f"{spearman(x,values['lag38_excess']):+.3f} | {spearman(x,values['lag76_excess']):+.3f}")


if __name__ == "__main__":
    main()
