"""Opt-in zero-player NCA with a controllable horizontal patch-copy update."""

from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import GameConfig
from .dataset import get_trajectory, read_manifest
from .game import Game
from .generate import Trajectory, _rollout, generate, sample_rules
from .io import save_batches

DESIGN_FILE = "design.json"
DESIGN_SCHEMA = "horizontal-patch-copy-v1"
GATE_FOLD_IN = 42


class HorizontalRecurrenceGame(Game):
    """Neural cell update followed by simultaneous left-to-right 2x2 patch copies.

    The probability controls the patch-copy gate, not an exact recurrence score.
    All patches are copied from the *newly updated* grid, before any copy is applied.
    """

    def __init__(self, config: GameConfig, copy_probability: float):
        if config.patch_size != 2 or config.groups != 1 or config.grid_size % 2:
            raise ValueError("horizontal patch copy needs 2x2 patches, one group, and an even grid")
        if not np.isfinite(copy_probability) or not 0 <= copy_probability <= 1:
            raise ValueError("copy_probability must be a finite number in [0, 1]")
        super().__init__(config)
        self.copy_probability = float(copy_probability)

    def step(self, seed, state, rule):
        updated = super().step(seed, state, rule)
        if self.copy_probability == 0:
            return updated
        side = self.config.grid_size // 2
        patches = updated.reshape(side, 2, side, 2, 1).transpose(0, 2, 1, 3, 4)
        mask = jax.random.bernoulli(
            jax.random.fold_in(seed, GATE_FOLD_IN), self.copy_probability, (side, side)
        ).at[:, 0].set(False)
        # Read every source from the pre-copy grid to avoid a left-to-right cascade.
        patches = jnp.where(
            mask[:, :, None, None, None], jnp.roll(patches, 1, axis=1), patches
        )
        return patches.transpose(0, 2, 1, 3, 4).reshape(
            self.config.grid_size, self.config.grid_size, 1
        )


class HorizontalRecurrenceGenerator:
    """Generate v1 raw trajectories with a horizontal recurrence control knob."""

    def __init__(self, config: GameConfig, copy_probability: float):
        self.config = config
        self.copy_probability = float(copy_probability)
        self.game = HorizontalRecurrenceGame(config, copy_probability)

    def sample_rule(self) -> np.ndarray:
        return sample_rules(replace(self.config, rule_count=1), game=self.game)[0]

    def generate_one(self, rule_seed: np.ndarray | None = None) -> Trajectory:
        one = replace(self.config, rule_count=1, trajectories_per_rule=1)
        seed = self.sample_rule() if rule_seed is None else np.asarray(rule_seed, dtype=np.uint32)
        if seed.shape != (2,):
            raise ValueError("rule_seed must be a JAX PRNG key with shape (2,)")
        batch = next(generate(one, seed[None, :], game=self.game))
        return Trajectory(batch.states[0], seed, batch.initial_seeds[0],
                          batch.initial_states[0])

    def generate(self, rule_seeds: np.ndarray | None = None):
        return generate(self.config, rule_seeds, game=self.game)

    def save(self, output: str | Path) -> Path:
        """Save original v1 arrays/manifest plus a replayable design sidecar."""
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        rules = sample_rules(self.config, game=self.game)
        implementation = _implementation()
        saved = save_batches(
            self.config, output, rules, self.generate(rules), implementation=implementation
        )
        (saved / DESIGN_FILE).write_text(json.dumps({
            "schema": DESIGN_SCHEMA,
            "copy_probability": self.copy_probability,
            "patch_size": 2,
            "gate_fold_in": GATE_FOLD_IN,
            "update": "neural cell update, then simultaneous copy of each gated 2x2 patch from its updated left neighbor",
        }, indent=2) + "\n")
        return saved


def _implementation() -> dict:
    """Retain the original implementation keys, with truthful designed provenance."""
    from .io import implementation_metadata

    result = implementation_metadata()
    result.update({
        "source": "SampleEff hybrid horizontal patch-copy NCA; neural rule adapted from https://github.com/danihyunlee/nca-pre-pretraining",
        "neighborhood": "3x3 Moore neural update, then aligned 2x2 left-neighbor patch copy; periodic neural boundary",
        "network": "3x3 conv -> 1x1 conv -> ReLU -> 1x1 conv; then stochastic patch-copy gate",
        "seed_derivation": "rule: successive jax.random.split from config seed; initial: fold_in(rule_seed, trajectory_id + 1); gate: fold_in(step_seed, 42)",
    })
    return result


def reproduce_saved(directory: str | Path, index: int):
    """Replay an original, patch-copy, or optimized-rule v1 dataset."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    config = GameConfig(**manifest["config"])
    sidecar = directory / DESIGN_FILE
    if sidecar.exists():
        design = json.loads(sidecar.read_text())
        if design.get("schema") == "optimized-neural-rules-v1":
            from .rule_design import reproduce_optimized

            record = get_trajectory(directory, index)
            return reproduce_optimized(directory, record, design, config)
        if (design.get("schema") != DESIGN_SCHEMA or design.get("patch_size") != 2
                or design.get("gate_fold_in") != GATE_FOLD_IN):
            raise ValueError("unknown NCA design sidecar")
        game = HorizontalRecurrenceGame(config, design["copy_probability"])
    else:
        network = manifest["implementation"]["network"]
        # Designed trajectories cannot be replayed from rules.npy alone.
        if "patch-copy" in network or "optimized weights" in network:
            raise ValueError("designed dataset is missing its design.json replay parameters")
        game = Game(config)
    record = get_trajectory(directory, index)
    initial, states = _rollout(game, config, config.frames)(
        jnp.asarray(record["rule_seed"], dtype=jnp.uint32),
        jnp.asarray(record["initial_seed"], dtype=jnp.uint32),
    )
    return np.asarray(initial, dtype=np.uint8), np.asarray(states, dtype=np.uint8)
