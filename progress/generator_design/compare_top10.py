"""Feature-only comparison of generated patch-copy worlds with frozen Top-10 worlds.

Reads the saved confirmation chunks. Generates new worlds in memory; never trains
models or changes the frozen corpora. Run from the repository root with uv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nca import GameConfig, HorizontalRecurrenceGenerator
from nca.analyze import gzip_ratio
from nca.generate import _rollout

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "runs/nca-data-search/top10-confirmation"
CUTOFFS = np.array([.034107767225, .025746229122, .017753705187,
                    .012177637072, .008144630498])
EDGES = np.array([.40, .42, .44, .46, .48, .50])
FIELDS = ("gzip", "horizontal", "vertical", "cell_change", "lag2", "transition_mi", "patch_entropy")


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def features(states: np.ndarray, gzip: float | None = None) -> np.ndarray:
    cells = states[..., 0].astype(np.int16)
    patches = cells.reshape(27, 6, 2, 6, 2).transpose(0, 1, 3, 2, 4)
    tokens = patches.reshape(27, 6, 6, 4) @ np.array([1, 10, 100, 1000])
    counts = np.bincount(tokens.ravel(), minlength=10_000)
    chance = (counts * (counts - 1)).sum() / (972 * 971)
    transitions = np.bincount((cells[:-1] * 10 + cells[1:]).ravel(),
                              minlength=100).reshape(10, 10)
    mi = entropy(transitions.sum(axis=0)) - sum(
        row.sum() / transitions.sum() * entropy(row)
        for row in transitions if row.sum()
    )
    return np.array([
        gzip_ratio(states, 10, 2) if gzip is None else gzip,
        (tokens[:, :, 1:] == tokens[:, :, :-1]).mean() - chance,
        (tokens[:, 1:, :] == tokens[:, :-1, :]).mean() - chance,
        (cells[1:] != cells[:-1]).mean(),
        (cells[2:] == cells[:-2]).mean(), mi, entropy(counts),
    ])


def bin_for(gzip: float) -> int:
    return int(np.searchsorted(EDGES, gzip, side="right") - 1) if .4 <= gzip < .5 else -1


def frozen_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, ratios, sampled = [], [], []
    for block in "abc":
        for path in sorted((FROZEN / f"block-{block}/chunks").glob("chunk-*.npz")):
            with np.load(path) as chunk:
                keep = np.flatnonzero(chunk["arm"] == 0)
                scores.extend(chunk["patch_copy_excess"][keep].tolist())
                ratios.extend(chunk["gzip_ratio"][keep].tolist())
                # Fixed receipt-index thinning gives a manageable feature sample.
                sampled_indices = keep[chunk["attempt"][keep] % 37 == 0]
                if len(sampled_indices):
                    worlds = chunk["states"][sampled_indices]
                    gzip_values = chunk["gzip_ratio"][sampled_indices]
                    sampled.extend(features(world, float(g)) for world, g in zip(
                        worlds, gzip_values, strict=True))
    return np.asarray(scores), np.asarray(ratios), np.asarray(sampled)


def generated_features(rules: int, seeds: list[int], probabilities: list[float]) -> dict[float, np.ndarray]:
    config = GameConfig()
    rollouts = {p: _rollout(HorizontalRecurrenceGenerator(config, p).game, config, config.frames)
                for p in probabilities}
    values = {p: [] for p in probabilities}
    for seed in seeds:
        cursor = jax.random.PRNGKey(seed)
        for _ in range(rules):
            cursor, rule = jax.random.split(cursor)
            initial = jax.random.fold_in(rule, 1)
            for p, rollout in rollouts.items():
                _, states = rollout(rule, initial)
                values[p].append(features(np.asarray(states, dtype=np.uint8)))
    return {p: np.asarray(rows) for p, rows in values.items()}


def summary(x: np.ndarray) -> dict:
    return {"n": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "p10": float(np.quantile(x, .1)), "median": float(np.median(x)),
            "p90": float(np.quantile(x, .9))}


def ks(x: np.ndarray, y: np.ndarray) -> float:
    grid = np.sort(np.unique(np.r_[x, y]))
    return float(np.max(np.abs(np.searchsorted(np.sort(x), grid, side="right") / len(x)
                               - np.searchsorted(np.sort(y), grid, side="right") / len(y))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=int, default=384, help="attempted rules per fresh stream")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2471, 8123])
    parser.add_argument("--probabilities", type=float, nargs="+",
                        default=[0, .01, .015, .02, .03, .04, .05])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("comparison.json"))
    args = parser.parse_args()
    if args.rules < 1 or not args.seeds or not args.probabilities:
        parser.error("rules, seeds, and probabilities must be nonempty")
    frozen_scores, frozen_gzip, frozen_sample = frozen_features()
    assert len(frozen_scores) == 3 * 20231
    assert np.all([bin_for(float(g)) >= 0 for g in frozen_gzip])
    assert np.all([s >= CUTOFFS[bin_for(float(g))] - 1e-12
                   for s, g in zip(frozen_scores, frozen_gzip, strict=True)])
    generated = generated_features(args.rules, args.seeds, args.probabilities)
    out = {
        "attempted_rules_per_stream": args.rules, "fresh_stream_seeds": args.seeds,
        "frozen_source": "three completed top10-confirmation designed corpora",
        "frozen": {
            "all_scores": summary(frozen_scores),
            "gzip_subbin_counts": np.bincount([bin_for(float(g)) for g in frozen_gzip], minlength=5).tolist(),
            "horizontal_by_subbin": [summary(frozen_scores[np.array(
                [bin_for(float(g)) == i for g in frozen_gzip])]) for i in range(5)],
            "thinned_feature_sample_n": len(frozen_sample),
            "features": {name: summary(frozen_sample[:, i]) for i, name in enumerate(FIELDS)},
        }, "generated": {},
    }
    for p, values in generated.items():
        bins = np.array([bin_for(float(g)) for g in values[:, 0]])
        keep = bins >= 0
        band = values[keep]
        selected_bins = bins[keep]
        passes = band[:, 1] >= CUTOFFS[selected_bins]
        out["generated"][str(p)] = {
            "attempted": len(values), "gzip_40_50_worlds": len(band),
            "gzip_subbin_counts": np.bincount(selected_bins, minlength=5).tolist(),
            "top10_cutoff_pass_fraction": float(passes.mean()),
            "top10_cutoff_pass_by_subbin": [float(passes[selected_bins == i].mean())
                                             if np.any(selected_bins == i) else None for i in range(5)],
            "horizontal_ks_vs_frozen": ks(band[:, 1], frozen_scores),
            "horizontal_by_subbin": [summary(band[selected_bins == i, 1])
                                     if np.any(selected_bins == i) else None for i in range(5)],
            "horizontal_ks_by_subbin": [ks(band[selected_bins == i, 1],
                frozen_scores[np.array([bin_for(float(g)) == i for g in frozen_gzip])])
                if np.any(selected_bins == i) else None for i in range(5)],
            "features": {name: summary(band[:, i]) for i, name in enumerate(FIELDS)},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
