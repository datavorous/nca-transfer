"""Read-only feature screen of the saved 40–50 NCA trajectories.

Run from the repository root with:
    uv run --all-packages python progress/nca_search/screen_existing.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/nca-100k-regimes"

def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def features(states: np.ndarray, gzip: float) -> dict[str, float]:
    s = states[..., 0].astype(np.uint8)
    changes = (s[1:] != s[:-1]).mean(axis=(1, 2))
    counts = np.bincount(s.ravel(), minlength=10)
    transitions = np.bincount(
        (s[:-1].astype(np.int16) * 10 + s[1:]).ravel(), minlength=100
    ).reshape(10, 10)
    conditional = sum(
        row.sum() / transitions.sum() * entropy(row)
        for row in transitions if row.sum()
    )
    next_entropy = entropy(transitions.sum(axis=0))
    patches = s.reshape(27, 6, 2, 6, 2).transpose(0, 1, 3, 2, 4).reshape(-1, 4)
    tokens = patches @ np.array([1, 10, 100, 1000], dtype=np.int32)
    active = np.any(s[1:] != s[:-1], axis=0).mean()
    frame_entropy = np.array([
        entropy(np.bincount(frame.ravel(), minlength=10)) for frame in s
    ])
    vertical_edge = 1 - (s == np.roll(s, 1, axis=1)).mean()
    horizontal_edge = 1 - (s == np.roll(s, 1, axis=2)).mean()
    frame_counts = np.stack([np.bincount(frame.ravel(), minlength=10) for frame in s])
    exact_fixed_count_baseline = ((frame_counts * (frame_counts - 1)).sum(axis=1)
                                  / (144 * 143)).mean()
    spatial_excess_agreement = (1 - (vertical_edge + horizontal_edge) / 2
                                - exact_fixed_count_baseline)
    return {
        "gzip": gzip,
        "change": float(changes.mean()),
        "early_change": float(changes[:13].mean()),
        "late_change": float(changes[13:].mean()),
        "change_slope": float(changes[13:].mean() - changes[:13].mean()),
        "change_volatility": float(changes.std()),
        "active_cells": float(active),
        "lag2_recurrence": float((s[2:] == s[:-2]).mean()),
        "lag3_recurrence": float((s[3:] == s[:-3]).mean()),
        "color_entropy": entropy(counts),
        "color_entropy_drift": float(frame_entropy[-1] - frame_entropy[0]),
        "frame_entropy_volatility": float(frame_entropy.std()),
        "dominant_color": float(counts.max() / counts.sum()),
        "spatial_edge": float((vertical_edge + horizontal_edge) / 2),
        "spatial_excess_agreement": float(spatial_excess_agreement),
        "spatial_anisotropy": float(abs(vertical_edge - horizontal_edge)),
        "transition_entropy": conditional,
        "transition_mutual_info": next_entropy - conditional,
        "unique_patch_fraction": float(len(np.unique(tokens)) / len(tokens)),
        "final_color_entropy": entropy(np.bincount(s[-1].ravel(), minlength=10)),
        "initial_color_entropy": entropy(np.bincount(s[0].ravel(), minlength=10)),
    }


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    cuts = np.flatnonzero(np.r_[True, sorted_x[1:] != sorted_x[:-1], True])
    result = np.empty(len(x), dtype=np.float64)
    for start, end in zip(cuts[:-1], cuts[1:]):
        result[order[start:end]] = (start + end - 1) / 2
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def within_gzip_spearman(x: np.ndarray, y: np.ndarray, gzip: np.ndarray) -> float:
    """Pooled rank association after ranking and centering inside .02 gzip bins."""
    bins = np.clip(((gzip - .40) / .02).astype(int), 0, 4)
    rx = np.empty(len(x), dtype=np.float64)
    ry = np.empty(len(y), dtype=np.float64)
    for bin_ in range(5):
        keep = bins == bin_
        n = int(keep.sum())
        assert n > 1
        rx[keep] = ranks(x[keep]) / n - (n - 1) / (2 * n)
        ry[keep] = ranks(y[keep]) / n - (n - 1) / (2 * n)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25_000)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    limit, stride = args.limit, args.stride
    if limit <= 0 or stride <= 0 or limit % stride:
        parser.error("limit and stride must be positive; limit must be divisible by stride")
    rows: list[dict] = []
    seen = 0
    for receipt in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for record in json.loads(receipt.read_text())["accepted"]:
            if record["band"] != "40_50":
                continue
            if seen >= limit:
                break
            if seen % stride == 0:
                rows.append(record)
            seen += 1
        if seen >= limit:
            break
    assert seen == limit and len(rows) == limit // stride
    by_batch: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_batch[row["dataset"]].append((i, row))
    result: list[dict | None] = [None] * len(rows)
    for dataset, members in sorted(by_batch.items()):
        states = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for i, row in members:
            result[i] = features(states[row["row"]], row["gzip_ratio"])
    assert all(row is not None for row in result)
    metrics = {key: np.array([row[key] for row in result]) for key in result[0]}
    print(f"Worlds: {len(result)} (every {stride}th of first {limit} 40–50 receipts)")
    print(f"spatial_excess_agreement_median_exact: {np.median(metrics['spatial_excess_agreement']):.12f}")
    print("property | median | p10 | p90 | rho(gzip) | rho(change)")
    for key, values in metrics.items():
        q = np.quantile(values, [.1, .5, .9])
        print(f"{key} | {q[1]:.4f} | {q[0]:.4f} | {q[2]:.4f} | "
              f"{spearman(values, metrics['gzip']):+.3f} | "
              f"{spearman(values, metrics['change']):+.3f}")
    selected = ["spatial_excess_agreement", "transition_mutual_info",
                "lag2_recurrence", "change_slope", "change_volatility",
                "gzip", "change"]
    print("\nfeature pair | raw rho | within-gzip-bin rho")
    for i, first in enumerate(selected):
        for second in selected[i + 1:]:
            print(f"{first} × {second} | "
                  f"{spearman(metrics[first], metrics[second]):+.3f} | "
                  f"{within_gzip_spearman(metrics[first], metrics[second], metrics['gzip']):+.3f}")


if __name__ == "__main__":
    main()
