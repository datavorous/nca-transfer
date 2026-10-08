"""Read-only screen of visible, nonidentity horizontal successor reuse.

Run from the repository root:
    uv run --all-packages python progress/nca_search/screen_successor_reuse.py

This analyzes saved trajectories only; it never loads or trains a model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from screen_attention_lags import SOURCE, selected_receipts
from screen_existing import spearman, within_gzip_spearman
from screen_transformer_targets import patch_grid


def successor_reuse(states: np.ndarray, gzip_ratio: float) -> dict[str, float]:
    patch = patch_grid(states)
    flat = patch.ravel()
    counts = np.bincount(flat, minlength=10_000).astype(np.int64)
    chance_copy = (counts * (counts - 1)).sum() / (972 * 971)
    copy_excess = (patch[:, :, :-1] == patch[:, :, 1:]).mean() - chance_copy

    # Packed position of the right-hand token of each within-row pair.
    right_pos = (38 * np.arange(27)[:, None, None]
                 + 1 + 6 * np.arange(6)[None, :, None]
                 + np.arange(1, 6)[None, None, :])
    left_ids = patch[:, :, :-1].ravel()
    right_ids = patch[:, :, 1:].ravel()
    positions = np.broadcast_to(right_pos, (27, 6, 5)).ravel()

    # Earlier screen's global bigram reuse, restricted here to noncopy pairs.
    seen_global: set[tuple[int, int]] = set()
    global_repeat = 0
    global_noncopy = 0
    for source, target in zip(left_ids, right_ids):
        pair = (int(source), int(target))
        if pair[0] == pair[1]:
            continue
        global_noncopy += 1
        global_repeat += pair in seen_global
        seen_global.add(pair)

    # Empirical within-window target frequencies form a frequency-only null:
    # with the source fixed, draw its successor from that window's targets.
    window_target_counts: dict[int, np.ndarray] = {}
    window_target_totals: dict[int, int] = {}
    valid = (positions <= 1024) & ((positions - 1) // 256 == (positions - 2) // 256) & (left_ids != right_ids)
    for window in range(4):
        mask = valid & ((positions - 1) // 256 == window)
        window_target_counts[window] = np.bincount(right_ids[mask], minlength=10_000)
        window_target_totals[window] = int(mask.sum())

    # A previous pair must be wholly visible when the current target is scored.
    # As in NCAWindows, scored targets are positions 38..1024. The source of a
    # current pair is position target-1, and each earlier pair must have its
    # own source and target inside the same nonoverlapping 256-token window.
    histories: dict[int, dict[int, list[int]]] = defaultdict(dict)
    prior_source = 0
    repeated_pair = 0
    same_last = 0
    expected_repeat = 0.0
    expected_last = 0.0
    noncopy_targets = 0
    for source, target, pos in zip(left_ids, right_ids, positions):
        source, target, pos = int(source), int(target), int(pos)
        if pos > 1024 or (pos - 1) // 256 != (pos - 2) // 256:
            continue
        if source == target:
            continue
        window = (pos - 1) // 256
        previous = histories[window].get(source)
        if pos >= 38:
            noncopy_targets += 1
            if previous is not None:
                prior_source += 1
                repeated_pair += target in previous
                # Last successor is the most immediately available completed pair.
                same_last += target == previous[-1]
                frequency = window_target_counts[window]
                total = window_target_totals[window]
                expected_repeat += sum(int(frequency[prior]) for prior in set(previous)) / total
                expected_last += int(frequency[previous[-1]]) / total
        if previous is None:
            histories[window][source] = [target]
        else:
            previous.append(target)

    return {
        "gzip": float(gzip_ratio),
        "cell_change": float((states[1:] != states[:-1]).mean()),
        "horizontal_copy_excess": float(copy_excess),
        "noncopy_targets": float(noncopy_targets),
        "prior_source_count": float(prior_source),
        "prior_source_rate": float(prior_source / noncopy_targets),
        "successor_reuse_rate": float(repeated_pair / prior_source) if prior_source else 0.0,
        "last_successor_rate": float(same_last / prior_source) if prior_source else 0.0,
        "repeat_pair_per_target": float(repeated_pair / noncopy_targets),
        "successor_reuse_excess": float((repeated_pair - expected_repeat) / prior_source) if prior_source else 0.0,
        "last_successor_excess": float((same_last - expected_last) / prior_source) if prior_source else 0.0,
        "global_noncopy_bigram_repeat": float(global_repeat / global_noncopy),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=5_000)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.first <= 0 or args.stride <= 0 or args.first % args.stride:
        parser.error("first and stride must be positive; first divisible by stride")
    receipts = selected_receipts(args.first, args.stride)
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, receipt in enumerate(receipts):
        grouped[receipt["dataset"]].append((index, receipt))
    results = [None] * len(receipts)
    for dataset, members in sorted(grouped.items()):
        worlds = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for index, receipt in members:
            results[index] = successor_reuse(worlds[receipt["row"]], receipt["gzip_ratio"])
    assert all(row is not None for row in results)
    cols = {key: np.asarray([row[key] for row in results]) for key in results[0]}
    print(f"n={len(results)}, every {args.stride}th of first {args.first} accepted 40–50 worlds")
    for key in ("noncopy_targets", "prior_source_count", "prior_source_rate",
                "successor_reuse_rate", "successor_reuse_excess", "last_successor_rate",
                "last_successor_excess", "repeat_pair_per_target", "global_noncopy_bigram_repeat"):
        x = cols[key]
        print(f"{key}: mean={x.mean():.6f}; p10/median/p90={np.quantile(x,[.1,.5,.9])}; "
              f"rho_gzip={spearman(x,cols['gzip']):+.3f}; "
              f"rho_copy={spearman(x,cols['horizontal_copy_excess']):+.3f}; "
              f"within_bin_rho_copy={within_gzip_spearman(x,cols['horizontal_copy_excess'],cols['gzip']):+.3f}")
    x = cols["successor_reuse_rate"]
    print(f"rho_reuse_with_global_bigram={spearman(x,cols['global_noncopy_bigram_repeat']):+.3f}; "
          f"rho_excess_with_global_bigram={spearman(cols['successor_reuse_excess'],cols['global_noncopy_bigram_repeat']):+.3f}")
    cutoff = np.median(x)
    high = x >= cutoff
    print(f"cutoff={cutoff:.9f}; zero_prior_source_share={(cols['prior_source_count']==0).mean():.3f}; "
          f"zero_reuse_share={(x==0).mean():.3f}")
    for band in range(5):
        mask = np.clip(((cols["gzip"] - .40) / .02).astype(int), 0, 4) == band
        print(f"bin_{band}: n={mask.sum()}, high_share={high[mask].mean():.3f}, "
              f"p10/median/p90={np.quantile(x[mask],[.1,.5,.9])}")
    for label, mask in (("high", high), ("low", ~high)):
        print(f"{label}: n={mask.sum()}, reuse={x[mask].mean():.5f}, "
              f"gzip={cols['gzip'][mask].mean():.5f}, copy={cols['horizontal_copy_excess'][mask].mean():+.5f}, "
              f"change={cols['cell_change'][mask].mean():.5f}, "
              f"eligible={cols['prior_source_count'][mask].mean():.1f}")
    bins = np.clip(((cols["gzip"] - .40) / .02).astype(int), 0, 4)
    cuts = np.array([np.median(x[bins == band]) for band in range(5)])
    high_within = x >= cuts[bins]
    print(f"within_bin_cuts={cuts}")
    for label, mask in (("within_high", high_within), ("within_low", ~high_within)):
        print(f"{label}: n={mask.sum()}, reuse={x[mask].mean():.5f}, "
              f"excess={cols['successor_reuse_excess'][mask].mean():+.5f}, "
              f"gzip={cols['gzip'][mask].mean():.5f}, copy={cols['horizontal_copy_excess'][mask].mean():+.5f}, "
              f"change={cols['cell_change'][mask].mean():.5f}, "
              f"eligible={cols['prior_source_count'][mask].mean():.1f}")
    adjusted = cols["successor_reuse_excess"]
    adjusted_cuts = np.array([np.median(adjusted[bins == band]) for band in range(5)])
    adjusted_high = adjusted >= adjusted_cuts[bins]
    print(f"adjusted_within_bin_cuts={adjusted_cuts}; "
          f"rho_adjusted_with_raw={spearman(adjusted,x):+.3f}; "
          f"positive_adjusted_share={(adjusted > 0).mean():.3f}")
    for band in range(5):
        mask = bins == band
        print(f"adjusted_bin_{band}: n={mask.sum()}, high_share={adjusted_high[mask].mean():.3f}, "
              f"p10/median/p90={np.quantile(adjusted[mask],[.1,.5,.9])}")
    for label, mask in (("adjusted_high", adjusted_high), ("adjusted_low", ~adjusted_high)):
        print(f"{label}: n={mask.sum()}, excess={adjusted[mask].mean():+.5f}, "
              f"reuse={x[mask].mean():.5f}, gzip={cols['gzip'][mask].mean():.5f}, "
              f"copy={cols['horizontal_copy_excess'][mask].mean():+.5f}, "
              f"change={cols['cell_change'][mask].mean():.5f}, "
              f"eligible={cols['prior_source_count'][mask].mean():.1f}, "
              f"global_bigram={cols['global_noncopy_bigram_repeat'][mask].mean():.5f}")


if __name__ == "__main__":
    main()
