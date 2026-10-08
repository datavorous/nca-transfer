"""Read-only cross-world coverage screen of saved 40–50 NCA trajectories.

Run from repo root:
    uv run --all-packages python progress/nca_search/screen_corpus_diversity.py

Samples every 20th accepted world among the first 25,000 receipts by default.
No trajectories are generated and no models are loaded or trained.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from screen_existing import features, spearman, within_gzip_spearman


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/nca-100k-regimes"
VOCAB = 10_000


def patch_grid(states: np.ndarray) -> np.ndarray:
    """Return [27, 36] patch IDs in the original framewise raster order."""
    s = states[..., 0].astype(np.int32)
    return (
        s.reshape(27, 6, 2, 6, 2)
        .transpose(0, 1, 3, 2, 4)
        .reshape(27, 36, 4)
        @ np.array([1, 10, 100, 1000], dtype=np.int32)
    )


def world_codes(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = patch_grid(states)
    singles = p.ravel()
    # Cross-frame transitions contain delimiters, so keep n-grams within frames.
    pairs = (p[:, :-1].astype(np.int64) * VOCAB + p[:, 1:]).ravel()
    triples = (
        p[:, :-2].astype(np.int64) * VOCAB * VOCAB
        + p[:, 1:-1].astype(np.int64) * VOCAB
        + p[:, 2:]
    ).ravel()
    return singles, pairs, triples


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=25_000)
    parser.add_argument("--stride", type=int, default=20)
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
    output = [None] * len(rows)
    for dataset, members in sorted(by_batch.items()):
        states = np.load(SOURCE / dataset / "trajectories-000000.npy", mmap_mode="r")
        for index, row in members:
            world = states[row["row"]]
            output[index] = (world_codes(world), features(world, row["gzip_ratio"]))
    assert all(item is not None for item in output)

    token_counts = np.zeros(VOCAB, dtype=np.int64)
    token_world_counts = np.zeros(VOCAB, dtype=np.int32)
    pair_world_counts: dict[int, int] = defaultdict(int)
    triple_world_counts: dict[int, int] = defaultdict(int)
    for codes, _ in output:
        singles, pairs, triples = codes
        token_counts += np.bincount(singles, minlength=VOCAB)
        for code in np.unique(singles):
            token_world_counts[int(code)] += 1
        for code in np.unique(pairs):
            pair_world_counts[int(code)] += 1
        for code in np.unique(triples):
            triple_world_counts[int(code)] += 1

    # Leave-one-world-out support: fraction of positions with an n-gram seen in
    # at least one OTHER sampled world. Unlike sequential novelty, this is order-free.
    measured = []
    for codes, base in output:
        singles, pairs, triples = codes
        world = dict(base)
        world["unigram_cross_world_mass"] = float((token_world_counts[singles] > 1).mean())
        world["bigram_cross_world_mass"] = float(np.mean(
            [pair_world_counts[int(code)] > 1 for code in pairs]
        ))
        world["trigram_cross_world_mass"] = float(np.mean(
            [triple_world_counts[int(code)] > 1 for code in triples]
        ))
        world["unigram_cross_world_vocab"] = float(np.mean(token_world_counts[np.unique(singles)] > 1))
        world["bigram_cross_world_vocab"] = float(np.mean(
            [pair_world_counts[int(code)] > 1 for code in np.unique(pairs)]
        ))
        world["within_world_unique_bigram_fraction"] = len(np.unique(pairs)) / len(pairs)
        measured.append(world)

    data = {key: np.array([world[key] for world in measured]) for key in measured[0]}
    total = int(token_counts.sum())
    print(f"worlds={len(rows)} first={args.first} stride={args.stride}; patch_positions={total}")
    print(f"unigram_vocab={np.count_nonzero(token_counts)}/{VOCAB}; token_entropy_bits={entropy(token_counts):.4f}; effective_vocab={2**entropy(token_counts):.1f}")
    print(f"top_100_token_mass={np.sort(token_counts)[-100:].sum()/total:.4f}; token_singleton_worlds={int((token_world_counts == 1).sum())}")
    print(f"bigram_vocab={len(pair_world_counts)}; bigram_singleton_worlds={sum(n == 1 for n in pair_world_counts.values())}")
    print(f"trigram_vocab={len(triple_world_counts)}; trigram_singleton_worlds={sum(n == 1 for n in triple_world_counts.values())}")
    print("property | p10 | median | p90 | rho_gzip | within_gzip_rho_change")
    keys = ["bigram_cross_world_mass", "trigram_cross_world_mass",
            "bigram_cross_world_vocab", "within_world_unique_bigram_fraction"]
    for key in keys:
        q = np.quantile(data[key], [.1, .5, .9])
        print(f"{key} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | "
              f"{spearman(data[key], data['gzip']):+.3f} | "
              f"{within_gzip_spearman(data[key], data['change'], data['gzip']):+.3f}")
    print("property | rho_spatial_excess | rho_transition_MI | rho_patch_uniqueness")
    for key in keys[:3]:
        print(f"{key} | " + " | ".join(f"{spearman(data[key], data[other]):+.3f}"
              for other in ("spatial_excess_agreement", "transition_mutual_info",
                            "unique_patch_fraction")))
    print("property | within_bin_rho_spatial_excess | within_bin_rho_transition_MI")
    for key in keys[:3]:
        print(f"{key} | " + " | ".join(
            f"{within_gzip_spearman(data[key], data[other], data['gzip']):+.3f}"
            for other in ("spatial_excess_agreement", "transition_mutual_info")))
    print("gzip_bin | worlds | mean_bigram_cross_world_mass | mean_trigram_cross_world_mass")
    bins = np.clip(((data["gzip"] - .40) / .02).astype(int), 0, 4)
    for b in range(5):
        keep = bins == b
        print(f"{.40 + .02*b:.2f}-{.42 + .02*b:.2f} | {keep.sum()} | "
              f"{data['bigram_cross_world_mass'][keep].mean():.4f} | "
              f"{data['trigram_cross_world_mass'][keep].mean():.4f}")


if __name__ == "__main__":
    main()
