"""Read-only vertical-lag screen on saved reference and selected NCA worlds.

Run: uv run --all-packages python progress/nca_search/screen_vertical_lag_target.py
No trajectory generation, model loading, or training.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from screen_existing import spearman, within_gzip_spearman
from screen_transformer_targets import patch_grid


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/nca-100k-regimes"
SEARCH = ROOT / "runs/nca-data-search"
FIRST = 5_000
STRIDE = 5
SELECTED_STRIDE = 20


def features(states: np.ndarray, gzip: float) -> dict[str, float]:
    patches = patch_grid(states)
    counts = np.bincount(patches.ravel(), minlength=10_000).astype(np.int64)
    global_chance = float((counts * (counts - 1)).sum() / (972 * 971))
    frame_counts = np.stack([np.bincount(frame.ravel(), minlength=10_000)
                             for frame in patches]).astype(np.int64)
    frame_chance = (frame_counts * (frame_counts - 1)).sum(axis=1) / (36 * 35)
    vertical_equal = patches[:, 1:, :] == patches[:, :-1, :]
    horizontal_equal = patches[:, :, 1:] == patches[:, :, :-1]
    # The packed position of a patch is 38*frame + 1 + 6*row + column.
    positions = np.arange(27)[:, None, None] * 38 + 1 + np.arange(6)[None, :, None] * 6 + np.arange(6)[None, None, :]
    positions = np.broadcast_to(positions, patches.shape)
    current = positions[:, 1:, :]
    previous = positions[:, :-1, :]
    first = ((current - 1) // 256) * 256 + 1
    visible = (current >= 38) & (current <= 1024) & (previous >= first)
    vertical_expected = np.broadcast_to(frame_chance[:, None, None], vertical_equal.shape)

    # Same patch coordinate two frames earlier; the source and target frame
    # marginals give the chance baseline. Use only scored, visible patch pairs.
    lag76_equal = patches[2:] == patches[:-2]
    lag76_current = positions[2:]
    lag76_previous = positions[:-2]
    lag76_first = ((lag76_current - 1) // 256) * 256 + 1
    lag76_visible = (lag76_current <= 1024) & (lag76_previous >= lag76_first)
    lag76_chance = np.empty(25, dtype=np.float64)
    for frame in range(2, 27):
        lag76_chance[frame - 2] = (frame_counts[frame] * frame_counts[frame - 2]).sum() / (36 * 36)
    lag76_expected = np.broadcast_to(lag76_chance[:, None, None], lag76_equal.shape)
    return {
        "gzip": float(gzip),
        "change": float(np.mean(states[1:] != states[:-1])),
        "vertical_copy": float(vertical_equal.mean()),
        "vertical_excess": float((vertical_equal - vertical_expected).mean()),
        "vertical_visible_excess": float((vertical_equal - vertical_expected)[visible].mean()),
        "vertical_global_excess": float(vertical_equal.mean() - global_chance),
        "horizontal_global_excess": float(horizontal_equal.mean() - global_chance),
        "lag76_visible_excess": float((lag76_equal - lag76_expected)[lag76_visible].mean()),
        "vertical_visible_share": float(visible.mean()),
        "lag76_visible_share": float(lag76_visible.mean()),
    }


def reference() -> list[dict[str, float]]:
    receipts = []
    seen = 0
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for row in json.loads(path.read_text())["accepted"]:
            if row["band"] != "40_50":
                continue
            if seen >= FIRST:
                break
            if seen % STRIDE == 0:
                receipts.append(row)
            seen += 1
        if seen >= FIRST:
            break
    assert len(receipts) == FIRST // STRIDE
    by_dataset = defaultdict(list)
    for index, row in enumerate(receipts):
        by_dataset[row["dataset"]].append((index, row))
    result = [None] * len(receipts)
    for dataset, members in sorted(by_dataset.items()):
        worlds = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for index, row in members:
            result[index] = features(worlds[row["row"]], row["gzip_ratio"])
    assert all(item is not None for item in result)
    return result


def selected(directory: str) -> list[dict[str, float]]:
    result = []
    index = 0
    for path in sorted((SEARCH / directory / "chunks").glob("chunk-*.npz")):
        with np.load(path) as chunk:
            for local in range(len(chunk["states"])):
                if int(chunk["split"][local]) == 0:
                    if index % SELECTED_STRIDE == 0:
                        result.append(features(chunk["states"][local], chunk["gzip_ratio"][local]))
                    index += 1
    assert index == 19_975, index
    return result


def arr(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.array([row[key] for row in rows])


def summary(name: str, rows: list[dict[str, float]]) -> None:
    print(f"{name}: n={len(rows)}")
    for key in ("gzip", "change", "horizontal_global_excess", "vertical_excess",
                "vertical_visible_excess", "vertical_global_excess", "lag76_visible_excess"):
        x = arr(rows, key)
        print(f"  {key}: mean={x.mean():+.6f}; p10/median/p90="
              f"{np.quantile(x,.1):+.6f}/{np.median(x):+.6f}/{np.quantile(x,.9):+.6f}")


def main() -> None:
    cohorts = {
        "reference": reference(),
        "high_horizontal": selected("iteration-04-high-patch-copy"),
        "low_horizontal": selected("iteration-05-low-patch-copy"),
    }
    for name, rows in cohorts.items():
        summary(name, rows)
    rows = cohorts["reference"]
    x = arr(rows, "vertical_excess")
    print("reference correlations: vertical_excess versus metric = raw rho, within-gzip-bin rho")
    for key in ("gzip", "change", "horizontal_global_excess", "lag76_visible_excess",
                "vertical_global_excess", "vertical_visible_excess"):
        y = arr(rows, key)
        print(f"  {key}: {spearman(x,y):+.4f}, {within_gzip_spearman(x,y,arr(rows,'gzip')):+.4f}")
    threshold = float(np.median(x))
    for name, cohort in cohorts.items():
        y = arr(cohort, "vertical_excess")
        print(f"{name}: fraction >= reference sample median {threshold:+.9f}: {np.mean(y >= threshold):.4f}")
    print("reference cross-features: horizontal excess vs visible lag76="
          f"{spearman(arr(rows,'horizontal_global_excess'),arr(rows,'lag76_visible_excess')):+.4f}")
    bins = np.clip(((arr(rows, "gzip") - .4) / .02).astype(int), 0, 4)
    print("reference high-vertical fraction by gzip bin:")
    for bin_ in range(5):
        keep = bins == bin_
        print(f"  {40 + 2*bin_}-{42 + 2*bin_}%: n={keep.sum()}, "
              f"high={np.mean(x[keep] >= threshold):.4f}")
    print("reference pair visibility: vertical="
          f"{arr(rows,'vertical_visible_share').mean():.4f}, lag76="
          f"{arr(rows,'lag76_visible_share').mean():.4f}")


if __name__ == "__main__":
    main()
