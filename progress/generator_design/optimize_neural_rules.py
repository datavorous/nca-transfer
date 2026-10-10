"""Optimize original NCA neural-rule weights toward requested trajectory features.

Exploratory feature-only (1+lambda) evolution strategy. It changes Flax rule
weights, not the cell update or tokenizer, and keeps no selected worlds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from nca import GameConfig
from nca.game import Game

from compare_top10 import FIELDS, features


def scorer(rows: np.ndarray, target: float) -> float:
    mean = rows.mean(axis=0)
    gzip = rows[:, 0]
    outside = np.maximum(.40 - gzip, 0) + np.maximum(gzip - .50, 0)
    return -(
        abs(mean[1] - target) / .02
        + 2.0 * outside.mean() / .02
        + .20 * abs(mean[3] - .73) / .15
        + .10 * abs(mean[6] - 7.25) / .5
    )


def make_rollout(config: GameConfig, game: Game, unravel):
    sample_times = config.burn_in + jnp.arange(config.frames) * config.stride
    total_steps = config.burn_in + (config.frames - 1) * config.stride + 1

    def one(flat, initial_seed):
        rule = unravel(flat)
        initial = game.initial_state(initial_seed)
        step_seeds = jax.random.split(initial_seed, total_steps)

        def advance(state, step_seed):
            following = game.step(step_seed, state, rule)
            return following, state

        _, states = jax.lax.scan(advance, initial, step_seeds)
        return states[sample_times].astype(jnp.uint8)

    return jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None)))


def evaluate(rollout, candidates: np.ndarray, initial_seeds: jax.Array,
             target: float) -> tuple[np.ndarray, list[np.ndarray]]:
    trajectories = np.asarray(rollout(jnp.asarray(candidates), initial_seeds))
    rows = [np.stack([features(states) for states in candidate])
            for candidate in trajectories]
    return np.array([scorer(group, target) for group in rows]), rows


def optimize(config: GameConfig, rule_seed: int, target: float, generations: int,
             population: int, sigma: float, seed: int, train_worlds: int,
             heldout_worlds: int):
    game = Game(config)
    initial_params = game.rule(jax.random.PRNGKey(rule_seed))
    flat, unravel = ravel_pytree(initial_params)
    rollout = make_rollout(config, game, unravel)
    train_seeds = jax.random.split(jax.random.PRNGKey(seed), train_worlds)
    heldout_seeds = jax.random.split(jax.random.PRNGKey(seed + 1), heldout_worlds)
    parent = np.asarray(flat)
    random = np.random.default_rng(seed)
    before_train, before_rows = evaluate(rollout, parent[None], train_seeds, target)
    before_holdout, before_holdout_rows = evaluate(rollout, parent[None], heldout_seeds, target)
    best = float(before_train[0])
    accepted = 0
    trace = []
    for generation in range(generations):
        scale = sigma * (0.99 ** generation)
        perturbations = random.normal(size=(population, len(parent))).astype(np.float32)
        candidates = np.concatenate((parent[None], parent[None] + scale * perturbations))
        fitness, rows = evaluate(rollout, candidates, train_seeds, target)
        winner = int(np.argmax(fitness))
        if winner:
            parent = candidates[winner].copy()
            best = float(fitness[winner])
            accepted += 1
        if generation in (0, generations // 2, generations - 1):
            trace.append({"generation": generation + 1, "fitness": best,
                          "mean_train_features": rows[winner].mean(axis=0).tolist()})
    after_train, after_rows = evaluate(rollout, parent[None], train_seeds, target)
    after_holdout, after_holdout_rows = evaluate(rollout, parent[None], heldout_seeds, target)
    return {
        "rule_seed": rule_seed, "target_horizontal_excess": target,
        "parameter_count": len(parent), "generations": generations,
        "population": population, "sigma": sigma, "accepted_updates": accepted,
        "train_worlds": train_worlds, "heldout_worlds_count": heldout_worlds,
        "feature_order": FIELDS,
        "before_train_mean": before_rows[0].mean(axis=0).tolist(),
        "after_train_mean": after_rows[0].mean(axis=0).tolist(),
        "before_heldout_mean": before_holdout_rows[0].mean(axis=0).tolist(),
        "after_heldout_mean": after_holdout_rows[0].mean(axis=0).tolist(),
        "before_fitness": [float(before_train[0]), float(before_holdout[0])],
        "after_fitness": [float(after_train[0]), float(after_holdout[0])],
        "heldout_worlds": after_holdout_rows[0].tolist(),
        "trace": trace,
    }, parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-seed", type=int, default=17)
    parser.add_argument("--target", type=float, default=.04)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=.03)
    parser.add_argument("--seed", type=int, default=4181)
    parser.add_argument("--train-worlds", type=int, default=4)
    parser.add_argument("--heldout-worlds", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("optimized_rule_probe.json"))
    args = parser.parse_args()
    if min(args.generations, args.population, args.train_worlds, args.heldout_worlds) < 1:
        parser.error("generations, population, and world counts must be positive")
    config = GameConfig()
    receipt, flat = optimize(config, args.rule_seed, args.target, args.generations,
                             args.population, args.sigma, args.seed,
                             args.train_worlds, args.heldout_worlds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    np.save(args.output.with_suffix(".npy"), flat)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
