"""Read-only early/late scored-patch diversity screen.

Run from repository root:
    uv run --all-packages python progress/nca_search/screen_entropy_stability_target.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from screen_existing import entropy, spearman, within_gzip_spearman
from screen_transformer_targets import patch_grid


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/nca-100k-regimes"
CORPORA = {
    "high copy": ROOT / "runs/nca-data-search/iteration-04-high-patch-copy/packed.pt",
    "low copy": ROOT / "runs/nca-data-search/iteration-05-low-patch-copy/packed.pt",
}
EARLY = slice(1, 10)  # 324 scored patch targets; first frame is masked.
LATE = slice(18, 27)  # 324 scored patch targets.
PATCH_POS = np.arange(27)[:, None] * 38 + 1 + np.arange(36)[None, :]


def patch_features(patches: np.ndarray) -> dict[str, float]:
    patches = patches.reshape(27, 36)
    early, late = patches[EARLY].ravel(), patches[LATE].ravel()
    early_counts = np.bincount(early, minlength=10_000)
    late_counts = np.bincount(late, minlength=10_000)
    all_counts = np.bincount(patches.ravel(), minlength=10_000)
    chance = np.sum(all_counts * (all_counts - 1)) / (972 * 971)
    horizontal = (patches.reshape(27, 6, 6)[:, :, 1:] ==
                  patches.reshape(27, 6, 6)[:, :, :-1]).mean()
    # A scored patch target repeats visibly only if its earlier occurrence is
    # inside the same 256-token training window, with markers in the clock.
    last_position: dict[int, int] = {}
    repeated = np.zeros((27, 36), dtype=bool)
    for frame in range(27):
        for site in range(36):
            pos = int(PATCH_POS[frame, site])
            token = int(patches[frame, site])
            context_start = ((pos - 1) // 256) * 256
            repeated[frame, site] = last_position.get(token, -1) >= context_start
            last_position[token] = pos
    return {
        "entropy_early": entropy(early_counts),
        "entropy_late": entropy(late_counts),
        "entropy_drift": entropy(late_counts) - entropy(early_counts),
        "unique_early": float(np.count_nonzero(early_counts)),
        "unique_late": float(np.count_nonzero(late_counts)),
        "unique_drift": float(np.count_nonzero(late_counts) - np.count_nonzero(early_counts)),
        "late_new_to_early": float(np.mean(early_counts[late] == 0)),
        "visible_repeat_early": float(repeated[EARLY].mean()),
        "visible_repeat_late": float(repeated[LATE].mean()),
        "visible_repeat_drift": float(repeated[LATE].mean() - repeated[EARLY].mean()),
        "horizontal_excess": float(horizontal - chance),
    }


def reference_rows(first: int, stride: int) -> list[dict]:
    chosen = []
    seen = 0
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for row in json.loads(path.read_text())["accepted"]:
            if row["band"] != "40_50":
                continue
            if seen >= first:
                break
            if seen % stride == 0:
                chosen.append(row)
            seen += 1
        if seen >= first:
            break
    assert seen == first and len(chosen) == first // stride
    grouped = defaultdict(list)
    for i, row in enumerate(chosen):
        grouped[row["dataset"]].append((i, row))
    output = [None] * len(chosen)
    for dataset, members in sorted(grouped.items()):
        worlds = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for i, row in members:
            states = worlds[row["row"]]
            output[i] = {
                **patch_features(patch_grid(states)),
                "gzip": float(row["gzip_ratio"]),
                "change": float(np.mean(states[1:] != states[:-1])),
            }
    assert all(row is not None for row in output)
    return output


def candidate_rows(path: Path, stride: int) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    sequence = payload["sequences"]
    assert sequence.shape[1] == 1026 and sequence.dtype == torch.int32
    rows = []
    for i in range(0, len(sequence) - 256, stride):  # training split only
        frames = sequence[i].numpy().reshape(27, 38)
        assert np.all(frames[:, 0] == 10_000) and np.all(frames[:, -1] == 10_001)
        rows.append(patch_features(frames[:, 1:-1]))
    return rows


def columns(rows: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=5_000)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--candidate-stride", type=int, default=20)
    args = parser.parse_args()
    if min(args.first, args.stride, args.candidate_stride) < 1 or args.first % args.stride:
        parser.error("positive first/stride/candidate-stride required, with first divisible by stride")
    ref = columns(reference_rows(args.first, args.stride))
    x = ref["entropy_drift"]
    print(f"reference={len(x)} deterministic accepted worlds; first={args.first}, stride={args.stride}")
    print("Feature | p10 | median | p90 | rho gzip | rho change | rho horizontal")
    for key in ("entropy_early", "entropy_late", "entropy_drift", "unique_drift",
                "late_new_to_early", "visible_repeat_drift"):
        y = ref[key]
        q = np.quantile(y, [.1, .5, .9])
        print(f"{key} | {q[0]:+.4f} | {q[1]:+.4f} | {q[2]:+.4f} | "
              f"{spearman(y, ref['gzip']):+.3f} | {spearman(y, ref['change']):+.3f} | "
              f"{spearman(y, ref['horizontal_excess']):+.3f}")
    print("within gzip-bin rho entropy drift vs change/horizontal/visible-repeat drift: " +
          " / ".join(f"{within_gzip_spearman(x, ref[key], ref['gzip']):+.3f}"
                     for key in ("change", "horizontal_excess", "visible_repeat_drift")))
    threshold = float(np.median(x))
    high = x >= threshold
    print(f"selector >= reference median {threshold:+.6f}; high={high.sum()}, low={(~high).sum()}")
    print("Reference high-minus-low means:")
    for key in ("entropy_early", "entropy_late", "unique_drift", "late_new_to_early",
                "visible_repeat_drift", "gzip", "change", "horizontal_excess"):
        y = ref[key]
        print(f"  {key}: {y[high].mean() - y[~high].mean():+.6f}")
    print("Cohort | n | early H | late H | drift H | unique drift | late novelty | visible repeat drift | horizontal excess")
    for name, data in [("reference", ref)] + [
        (name, columns(candidate_rows(path, args.candidate_stride)))
        for name, path in CORPORA.items()
    ]:
        print(f"{name} | {len(data['entropy_drift'])} | " + " | ".join(
            f"{data[key].mean():+.5f}" for key in (
                "entropy_early", "entropy_late", "entropy_drift", "unique_drift",
                "late_new_to_early", "visible_repeat_drift", "horizontal_excess")))


if __name__ == "__main__":
    main()
