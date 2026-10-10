"""Cheap feature-only probe of a configurable patch-copy NCA update.

Run from the repository root:
    uv run --locked --all-packages python progress/generator_design/probe_patch_copy.py

This script does not train a model or save trajectories. Its patch-copy operation
is an exploratory generator intervention, separate from the frozen Top-10 selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from nca.analyze import gzip_ratio
from nca.config import GameConfig
from nca.generate import _rollout
from nca.recurrence import HorizontalRecurrenceGame


def features(states: np.ndarray) -> tuple[float, float, float, float]:
    patch_grid = states[..., 0].reshape(27, 6, 2, 6, 2).transpose(0, 1, 3, 2, 4)
    patch_ids = (patch_grid.reshape(27, 6, 6, 4).astype(np.int32)
                 * np.array([1, 10, 100, 1000], np.int32)).sum(axis=-1)
    counts = np.bincount(patch_ids.ravel(), minlength=10_000).astype(np.int64)
    chance = float((counts * (counts - 1)).sum() / (972 * 971))
    horizontal = float((patch_ids[:, :, 1:] == patch_ids[:, :, :-1]).mean() - chance)
    vertical = float((patch_ids[:, 1:, :] == patch_ids[:, :-1, :]).mean() - chance)
    change = float((states[1:] != states[:-1]).mean())
    return gzip_ratio(states, 10, 2), horizontal, vertical, change


def summarize(values: np.ndarray) -> dict:
    band = (values[:, 0] >= 0.4) & (values[:, 0] < 0.5)
    selected = values[band]
    return {
        "worlds": len(values),
        "gzip_40_50_worlds": int(band.sum()),
        "gzip_mean": float(values[:, 0].mean()),
        "horizontal_excess_mean": float(values[:, 1].mean()),
        "cell_change_mean": float(values[:, 3].mean()),
        "within_gzip_40_50": {
            "gzip_mean": float(selected[:, 0].mean()) if len(selected) else None,
            "horizontal_excess_mean": float(selected[:, 1].mean()) if len(selected) else None,
            "horizontal_excess_median": float(np.median(selected[:, 1])) if len(selected) else None,
            "vertical_excess_mean": float(selected[:, 2].mean()) if len(selected) else None,
            "cell_change_mean": float(selected[:, 3].mean()) if len(selected) else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=int, default=192)
    parser.add_argument("--seed", type=int, default=2471)
    parser.add_argument("--probabilities", type=float, nargs="+",
                        default=[0.0, 0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rules < 1 or not args.probabilities:
        parser.error("rules and probabilities must be nonempty")
    if len(set(args.probabilities)) != len(args.probabilities):
        parser.error("probabilities must be distinct")

    config = GameConfig()
    rollouts = {
        probability: _rollout(HorizontalRecurrenceGame(config, probability), config, config.frames)
        for probability in args.probabilities
    }
    rows = {probability: [] for probability in args.probabilities}
    cursor = jax.random.PRNGKey(args.seed)
    for _ in range(args.rules):
        cursor, rule_seed = jax.random.split(cursor)
        initial_seed = jax.random.fold_in(rule_seed, 1)
        for probability, rollout in rollouts.items():
            _, states = rollout(rule_seed, initial_seed)
            rows[probability].append(features(np.asarray(states, dtype=np.uint8)))

    result = {
        "design": "original neural update plus simultaneous left-neighbor 2x2 patch copy",
        "seed": args.seed,
        "rules": args.rules,
        "gzip_band": [0.4, 0.5],
        "conditions": {
            str(probability): summarize(np.asarray(values))
            for probability, values in rows.items()
        },
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
