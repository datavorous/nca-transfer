"""Feature-only screen of spatial and temporal NCA update controls.

Exploratory code, separate from the public generator and frozen corpora.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nca import GameConfig
from nca.game import Game
from nca.generate import _rollout

from compare_top10 import CUTOFFS, FIELDS, bin_for, features, frozen_features, ks, summary


class ProbeGame(Game):
    """Original neural update plus independent, local feature controls."""

    def __init__(self, config: GameConfig, horizontal: float, vertical: float,
                 two_cycle: float, enforce_change: float = 0.):
        super().__init__(config)
        if config.patch_size != 2 or config.groups != 1 or config.num_colors % 2:
            raise ValueError("probe needs 2x2 patches, one group, and even colors")
        for value in (horizontal, vertical, two_cycle, enforce_change):
            if not 0 <= value <= 1:
                raise ValueError("probabilities must be in [0, 1]")
        self.horizontal = horizontal
        self.vertical = vertical
        self.two_cycle = two_cycle
        self.enforce_change = enforce_change

    def step(self, seed, state, rule):
        updated = super().step(seed, state, rule)
        if self.two_cycle:
            gate = jax.random.bernoulli(jax.random.fold_in(seed, 44),
                                        self.two_cycle, updated.shape)
            updated = jnp.where(gate, (state + self.config.num_colors // 2)
                                % self.config.num_colors, updated)
        if self.enforce_change:
            gate = jax.random.bernoulli(jax.random.fold_in(seed, 45),
                                        self.enforce_change, updated.shape)
            updated = jnp.where(gate & (updated == state),
                                (state + self.config.num_colors // 2)
                                % self.config.num_colors, updated)
        if not self.horizontal and not self.vertical:
            return updated
        side = self.config.grid_size // 2
        patches = updated.reshape(side, 2, side, 2, 1).transpose(0, 2, 1, 3, 4)
        if self.vertical:
            gate = jax.random.bernoulli(jax.random.fold_in(seed, 43),
                                        self.vertical, (side, side)).at[0].set(False)
            patches = jnp.where(gate[:, :, None, None, None],
                                jnp.roll(patches, 1, axis=0), patches)
        if self.horizontal:
            gate = jax.random.bernoulli(jax.random.fold_in(seed, 42),
                                        self.horizontal, (side, side)).at[:, 0].set(False)
            patches = jnp.where(gate[:, :, None, None, None],
                                jnp.roll(patches, 1, axis=1), patches)
        return patches.transpose(0, 2, 1, 3, 4).reshape(
            self.config.grid_size, self.config.grid_size, 1)


def main() -> None:
    config = GameConfig()
    # One-axis probes first; combination tests whether the controls interfere.
    variants = {
        "horizontal_only": (.03, 0., 0.),
        "vertical_001": (.03, .01, 0.),
        "vertical_002": (.03, .02, 0.),
        "vertical_004": (.03, .04, 0.),
        "two_cycle_002": (.03, 0., .02),
        "two_cycle_005": (.03, 0., .05),
        "two_cycle_010": (.03, 0., .10),
        "combined": (.03, .02, .05),
        "temporal_vertical": (.03, .01, .10),
        "change_008": (.03, .01, .10, .08),
        "change_010": (.03, .01, .10, .10),
        "change_012": (.03, .01, .10, .12),
        "change_016": (.03, .01, .10, .16),
    }
    rollouts = {name: _rollout(ProbeGame(config, *knobs), config, config.frames)
                for name, knobs in variants.items()}
    frozen_scores, frozen_gzip, frozen_sample = frozen_features()
    rows = {name: [] for name in variants}
    for stream_seed in (4591, 9323):
        cursor = jax.random.PRNGKey(stream_seed)
        for _ in range(384):
            cursor, rule = jax.random.split(cursor)
            initial = jax.random.fold_in(rule, 1)
            for name, rollout in rollouts.items():
                _, states = rollout(rule, initial)
                values = features(np.asarray(states, dtype=np.uint8))
                if bin_for(float(values[0])) >= 0:
                    rows[name].append(values)
    output = {
        "attempted_rules_per_variant": 768,
        "stream_seeds": [4591, 9323],
        "frozen_feature_sample_n": len(frozen_sample),
        "frozen_features": {name: summary(frozen_sample[:, i])
                            for i, name in enumerate(FIELDS)},
        "frozen_full_horizontal": summary(frozen_scores),
        "variants": {},
    }
    for name, parameters in variants.items():
        values = np.asarray(rows[name])
        bins = np.array([bin_for(float(g)) for g in values[:, 0]])
        output["variants"][name] = {
            "horizontal_copy_probability": parameters[0],
            "vertical_copy_probability": parameters[1],
            "two_cycle_probability": parameters[2],
            "enforce_change_probability": parameters[3] if len(parameters) > 3 else 0.,
            "gzip_40_50_worlds": len(values),
            "features": {field: summary(values[:, i]) for i, field in enumerate(FIELDS)},
            "feature_ks_vs_frozen_sample": {field: ks(values[:, i], frozen_sample[:, i])
                                            for i, field in enumerate(FIELDS)},
            "horizontal_ks_vs_all_frozen": ks(values[:, 1], frozen_scores),
            "top10_cutoff_pass_fraction": float(np.mean(values[:, 1] >= CUTOFFS[bins])),
            "horizontal_by_subbin": [summary(values[bins == i, 1])
                                     if np.any(bins == i) else None for i in range(5)],
        }
    path = Path(__file__).with_name("additional_knobs.json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
