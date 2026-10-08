"""Screen saved NCA worlds using the actual 2x2 raster patch token order.

Run: uv run --all-packages python progress/nca_search/screen_transformer_targets.py
No generation, checkpoint loading, or training is performed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from screen_existing import entropy, features, spearman, within_gzip_spearman


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/nca-100k-regimes"


def patch_grid(states: np.ndarray) -> np.ndarray:
    """[27, 6, 6] int patch IDs, matching pack_one's ordering and powers."""
    s = states[..., 0].astype(np.int32)
    return (
        s.reshape(27, 6, 2, 6, 2)
        .transpose(0, 1, 3, 2, 4)
        .reshape(27, 6, 6, 4)
        @ np.array([1, 10, 100, 1000], dtype=np.int32)
    )


def token_features(states: np.ndarray) -> dict[str, float]:
    p = patch_grid(states)
    flat = p.ravel()
    counts = np.bincount(flat, minlength=10_000)
    expected_copy = float(np.sum(counts * (counts - 1)) / (len(flat) * (len(flat) - 1)))
    first = p[:18].ravel()
    first_nine = p[:9].ravel()
    last = p[18:].ravel()
    seen_first = np.zeros(10_000, dtype=bool)
    seen_first[first] = True

    # NCAWindows scores targets at t=1..1024 in four nonoverlapping 256-token
    # windows; first-frame targets are masked. Count patch tokens t>=38 only.
    # Markers are included in the position clock but not as targets here.
    sequence = np.concatenate([
        np.concatenate((np.array([10_000]), frame.ravel(), np.array([10_001])))
        for frame in p
    ])
    last_position: dict[int, int] = {}
    repeat_256 = []
    for index, token in enumerate(sequence):
        token = int(token)
        if token < 10_000 and 38 <= index <= 1024:
            window_start = ((index - 1) // 256) * 256
            repeat_256.append(last_position.get(token, -1) >= window_start)
        last_position[token] = index

    # Within-frame horizontal bigrams only: boundary marker transitions are excluded.
    bigrams = np.stack((p[:, :, :-1], p[:, :, 1:]), axis=-1).reshape(-1, 2)
    seen_bigrams: set[tuple[int, int]] = set()
    repeated_bigram = 0
    for left, right in bigrams:
        pair = (int(left), int(right))
        repeated_bigram += pair in seen_bigrams
        seen_bigrams.add(pair)

    return {
        "patch_entropy": entropy(counts),
        "effective_patch_vocab": float(2 ** entropy(counts)),
        "rare_patch_mass": float(counts[counts <= 2].sum() / len(flat)),
        "context_repeat_256": float(np.mean(repeat_256)),
        "horizontal_patch_copy": float((p[:, :, 1:] == p[:, :, :-1]).mean()),
        "vertical_patch_copy": float((p[:, 1:, :] == p[:, :-1, :]).mean()),
        "temporal_same_site_copy": float((p[1:] == p[:-1]).mean()),
        "copy_chance": expected_copy,
        "horizontal_copy_excess": float((p[:, :, 1:] == p[:, :, :-1]).mean() - expected_copy),
        "temporal_copy_excess": float((p[1:] == p[:-1]).mean() - expected_copy),
        "previous_frame_vocab_share": float(np.mean([
            np.isin(p[t], p[t - 1]).mean() for t in range(1, 27)
        ])),
        "late_patch_novelty": float((~seen_first[last]).mean()),
        "horizontal_bigram_repeat": repeated_bigram / len(bigrams),
        "late_minus_early_patch_entropy": entropy(np.bincount(last, minlength=10_000))
        - entropy(np.bincount(first_nine, minlength=10_000)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=15_000,
                        help="number of first accepted 40–50 receipts to consider")
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.first <= 0 or args.stride <= 0 or args.first % args.stride:
        parser.error("first and stride must be positive; first must be divisible by stride")

    rows = []
    seen = 0
    for receipt in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for record in json.loads(receipt.read_text())["accepted"]:
            if record["band"] != "40_50":
                continue
            if seen >= args.first:
                break
            if seen % args.stride == 0:
                rows.append(record)
            seen += 1
        if seen >= args.first:
            break
    assert seen == args.first and len(rows) == args.first // args.stride

    by_batch: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_batch[row["dataset"]].append((index, row))
    result: list[dict[str, float] | None] = [None] * len(rows)
    for dataset, members in sorted(by_batch.items()):
        worlds = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for index, row in members:
            states = worlds[row["row"]]
            result[index] = {**features(states, row["gzip_ratio"]), **token_features(states)}
    assert all(row is not None for row in result)
    values = {key: np.array([row[key] for row in result]) for key in result[0]}
    print(f"Worlds: {len(result)} (every {args.stride}th of first {args.first} 40–50 receipts)")
    print(f"horizontal_copy_excess_median_exact: {np.median(values['horizontal_copy_excess']):.12f}")
    print("property | p10 | median | p90 | rho_gzip | rho_change | within_gzip_rho_change")
    for key in token_features(np.load(SOURCE / rows[0]["dataset"] /
                                        "trajectories-000000.npy", mmap_mode="r")[rows[0]["row"]]):
        x = values[key]
        q = np.quantile(x, [.1, .5, .9])
        print(f"{key} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | "
              f"{spearman(x, values['gzip']):+.3f} | "
              f"{spearman(x, values['change']):+.3f} | "
              f"{within_gzip_spearman(x, values['change'], values['gzip']):+.3f}")
    print("\nproperty | rho_spatial_excess | rho_transition_MI | rho_lag2 | rho_context_repeat")
    for key in ("patch_entropy", "rare_patch_mass", "context_repeat_256",
                "horizontal_copy_excess", "temporal_copy_excess",
                "previous_frame_vocab_share", "late_patch_novelty",
                "horizontal_bigram_repeat", "late_minus_early_patch_entropy"):
        print(f"{key} | " + " | ".join(
            f"{spearman(values[key], values[other]):+.3f}" for other in (
                "spatial_excess_agreement", "transition_mutual_info",
                "lag2_recurrence", "context_repeat_256")))


if __name__ == "__main__":
    main()
