"""Sample NCA rules and stream their trajectories in bounded batches."""

from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .analyze import gzip_ratio
from .config import GameConfig
from .game import Game


@dataclass(frozen=True)
class TrajectoryBatch:
    states: np.ndarray  # [batch, frames, height, width, groups]
    rule_ids: np.ndarray  # [batch]
    initial_seeds: np.ndarray  # [batch, 2]
    initial_states: np.ndarray  # [batch, height, width, groups]


@dataclass(frozen=True)
class Trajectory:
    states: np.ndarray  # [frames, height, width, groups]
    rule_seed: np.ndarray  # [2]
    initial_seed: np.ndarray  # [2]
    initial_state: np.ndarray  # [height, width, groups]


def _rollout(game: Game, config: GameConfig, frames: int):
    sample_times = config.burn_in + jnp.arange(frames) * config.stride
    total_steps = config.burn_in + (frames - 1) * config.stride + 1

    def run(rule_seed, trajectory_seed):
        rule = game.rule(rule_seed)
        initial = game.initial_state(trajectory_seed)
        step_seeds = jax.random.split(trajectory_seed, total_steps)

        def advance(state, step_seed):
            next_state = game.step(step_seed, state, rule)
            return next_state, state

        _, states = jax.lax.scan(advance, initial, step_seeds)
        return initial, states[sample_times]

    return jax.jit(run)


def sample_rules(config: GameConfig) -> np.ndarray:
    """Return reproducible PRNG keys for accepted NCA transition rules."""
    game = Game(config)
    probe = _rollout(game, config, config.probe_frames) if config.filter_mode != "none" else None
    cursor = jax.random.PRNGKey(config.seed)
    accepted = []
    for attempt in range(config.max_rule_attempts):
        cursor, rule_seed = jax.random.split(cursor)
        if probe is not None:
            trajectory_seed = jax.random.fold_in(rule_seed, 0)
            _, sampled = probe(rule_seed, trajectory_seed)
            sampled = np.asarray(sampled)
            score = (
                float(np.mean(sampled[1:] != sampled[:-1]))
                if config.filter_mode == "diff"
                else gzip_ratio(sampled, config.num_colors, config.patch_size)
            )
            if not config.min_complexity < score < config.max_complexity:
                continue
        accepted.append(np.asarray(rule_seed, dtype=np.uint32))
        if len(accepted) == config.rule_count:
            return np.stack(accepted)
    raise RuntimeError(
        f"accepted {len(accepted)} of {config.rule_count} rules in {config.max_rule_attempts} attempts; "
        "relax the complexity bounds or raise max_rule_attempts"
    )


def generate(config: GameConfig, rule_seeds: np.ndarray | None = None):
    """Yield trajectory batches without holding the whole dataset in memory."""
    rules = sample_rules(config) if rule_seeds is None else np.asarray(rule_seeds)
    if rules.shape != (config.rule_count, 2):
        raise ValueError(f"rule_seeds must have shape ({config.rule_count}, 2)")
    game = Game(config)
    rollout = _rollout(game, config, config.frames)
    states = []
    ids = []
    initial_seeds = []
    initial_states = []
    dtype = np.uint8 if config.num_colors <= 256 else np.uint16
    for rule_id, rule_seed in enumerate(rules):
        key = jnp.asarray(rule_seed, dtype=jnp.uint32)
        for trajectory_id in range(config.trajectories_per_rule):
            trajectory_seed = jax.random.fold_in(key, trajectory_id + 1)
            initial, trajectory = rollout(key, trajectory_seed)
            states.append(np.asarray(trajectory, dtype=dtype))
            initial_states.append(np.asarray(initial, dtype=dtype))
            initial_seeds.append(np.asarray(trajectory_seed, dtype=np.uint32))
            ids.append(rule_id)
            if len(states) == config.batch_size:
                yield TrajectoryBatch(
                    np.stack(states), np.asarray(ids, dtype=np.int32),
                    np.stack(initial_seeds), np.stack(initial_states)
                )
                states, ids, initial_seeds, initial_states = [], [], [], []
    if states:
        yield TrajectoryBatch(
            np.stack(states), np.asarray(ids, dtype=np.int32),
            np.stack(initial_seeds), np.stack(initial_states)
        )


def reproduce(config: GameConfig, rule_seed: np.ndarray, initial_seed: np.ndarray):
    """Recreate one raw trajectory and its initial grid from saved seeds."""
    game = Game(config)
    initial, states = _rollout(game, config, config.frames)(
        jnp.asarray(rule_seed, dtype=jnp.uint32),
        jnp.asarray(initial_seed, dtype=jnp.uint32),
    )
    return np.asarray(initial, dtype=np.uint8), np.asarray(states, dtype=np.uint8)


class NCAGenerator:
    """Python entry point for the existing rule sampler and batch generator."""

    def __init__(self, config: GameConfig):
        self.config = config

    def sample_rule(self) -> np.ndarray:
        return sample_rules(replace(self.config, rule_count=1))[0]

    def generate_one(self, rule_seed: np.ndarray | None = None) -> Trajectory:
        one = replace(self.config, rule_count=1, trajectories_per_rule=1)
        seed = self.sample_rule() if rule_seed is None else np.asarray(rule_seed, dtype=np.uint32)
        if seed.shape != (2,):
            raise ValueError("rule_seed must be a JAX PRNG key with shape (2,)")
        batch = next(generate(one, seed[None, :]))
        return Trajectory(batch.states[0], seed, batch.initial_seeds[0],
                          batch.initial_states[0])

    def generate(self, rule_seeds: np.ndarray | None = None):
        return generate(self.config, rule_seeds)

    def save(self, output: str | Path) -> Path:
        from .io import save

        return save(self.config, output)
