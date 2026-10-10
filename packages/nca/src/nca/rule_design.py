"""Feature-directed optimization of the original neural NCA rule weights.

The simulator remains ``Game``. Only its Flax parameter values change. The
search evaluates hard categorical trajectories, so no differentiable surrogate
or change to the cell update is needed.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .analyze import gzip_ratio
from .config import GameConfig
from .game import Game
from .generate import TrajectoryBatch
from .io import implementation_metadata, save_batches

DESIGN_SCHEMA = "optimized-neural-rules-v1"
WEIGHTS_FILE = "optimized-rule-weights.npy"


@dataclass(frozen=True)
class RuleDesign:
    """Desired trajectory properties, measured on full hard rollouts."""

    horizontal_excess: float
    gzip_min: float = .4
    gzip_max: float = .5
    cell_change: float = .73
    patch_entropy: float = 7.25
    vertical_excess: float | None = None
    lag2: float | None = None
    transition_mi: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.horizontal_excess):
            raise ValueError("horizontal_excess must be finite")
        if (not np.isfinite(self.gzip_min) or not np.isfinite(self.gzip_max)
                or not 0 <= self.gzip_min < self.gzip_max):
            raise ValueError("gzip bounds must be nonnegative and increasing")
        if (not np.isfinite(self.cell_change) or not np.isfinite(self.patch_entropy)
                or not 0 <= self.cell_change <= 1 or self.patch_entropy < 0):
            raise ValueError("cell_change and patch_entropy targets are invalid")
        if any(value is not None and not np.isfinite(value) for value in
               (self.vertical_excess, self.lag2, self.transition_mi)):
            raise ValueError("optional feature targets must be finite")
        if self.lag2 is not None and not 0 <= self.lag2 <= 1:
            raise ValueError("lag2 target must lie in [0, 1]")
        if self.transition_mi is not None and self.transition_mi < 0:
            raise ValueError("transition_mi target must be nonnegative")


@dataclass(frozen=True)
class RuleSearch:
    """Small (1+lambda) search over neural weights; seeds fix the feature split."""

    generations: int = 100
    population: int = 16
    sigma: float = .03
    train_worlds: int = 4
    heldout_worlds: int = 8
    seed: int = 4181

    def __post_init__(self):
        if min(self.generations, self.population, self.train_worlds,
               self.heldout_worlds) < 1 or not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("search counts and sigma must be positive")


@dataclass(frozen=True)
class OptimizedRule:
    rule_seed: np.ndarray  # uint32[2], original neural initialization key
    weights: np.ndarray  # flattened float32 Flax parameters, same architecture
    receipt: dict


def trajectory_features(states: np.ndarray, config: GameConfig) -> dict[str, float]:
    """Exact hard-trajectory features used by the rule search."""
    if (config.groups != 1 or config.patch_size != 2 or config.grid_size < 4
            or config.grid_size % 2 or config.frames < 2):
        raise ValueError("rule design needs at least two frames, a grid >= 4, one group, and 2x2 patches")
    cells = np.asarray(states[..., 0], dtype=np.int32)
    frames, side, _ = cells.shape
    patches = cells.reshape(frames, side // 2, 2, side // 2, 2).transpose(0, 1, 3, 2, 4)
    tokens = patches.reshape(frames, side // 2, side // 2, 4) @ (
        config.num_colors ** np.arange(4, dtype=np.int32))
    counts = np.bincount(tokens.ravel(), minlength=config.num_colors ** 4)
    total = tokens.size
    chance = float(np.sum(counts * (counts - 1)) / (total * (total - 1)))
    nonzero = counts[counts > 0] / total
    transitions = np.bincount(
        (cells[:-1] * config.num_colors + cells[1:]).ravel(),
        minlength=config.num_colors ** 2,
    ).reshape(config.num_colors, config.num_colors)
    transition_total = transitions.sum()

    def entropy(values):
        p = values[values > 0] / values.sum()
        return float(-(p * np.log2(p)).sum())

    mutual_info = entropy(transitions.sum(axis=0)) - sum(
        row.sum() / transition_total * entropy(row)
        for row in transitions if row.sum()
    )
    return {
        "gzip": gzip_ratio(states, config.num_colors, 2),
        "horizontal_excess": float((tokens[:, :, 1:] == tokens[:, :, :-1]).mean() - chance),
        "vertical_excess": float((tokens[:, 1:, :] == tokens[:, :-1, :]).mean() - chance),
        "cell_change": float((cells[1:] != cells[:-1]).mean()),
        "lag2": float((cells[2:] == cells[:-2]).mean()) if frames >= 3 else 0.,
        "transition_mi": float(mutual_info),
        "patch_entropy": float(-(nonzero * np.log2(nonzero)).sum()),
    }


def _fitness(rows: list[dict], design: RuleDesign) -> float:
    mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    # Penalize gzip drift on every rollout so a good mean cannot hide out-of-band worlds.
    outside = np.mean([max(design.gzip_min - row["gzip"], 0,
                           row["gzip"] - design.gzip_max) for row in rows])
    penalty = (
        abs(mean["horizontal_excess"] - design.horizontal_excess) / .02
        + 2 * outside / .02
        + .20 * abs(mean["cell_change"] - design.cell_change) / .15
        + .10 * abs(mean["patch_entropy"] - design.patch_entropy) / .5
    )
    for name, target, scale in (
        ("vertical_excess", design.vertical_excess, .02),
        ("lag2", design.lag2, .1),
        ("transition_mi", design.transition_mi, .1),
    ):
        if target is not None:
            penalty += .5 * abs(mean[name] - target) / scale
    return -float(penalty)


def _rollout_with_weights(game: Game, config: GameConfig, unravel):
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
        return initial.astype(jnp.uint8), states[sample_times].astype(jnp.uint8)

    return jax.jit(one)


def optimize_rule(config: GameConfig, rule_seed: int, design: RuleDesign,
                  search: RuleSearch = RuleSearch()) -> OptimizedRule:
    """Tune one original neural rule on train worlds; record held-out worlds."""
    if (config.groups != 1 or config.patch_size != 2 or config.grid_size < 4
            or config.grid_size % 2 or config.frames < 2):
        raise ValueError("rule design needs at least two frames, a grid >= 4, one group, and 2x2 patches")
    game = Game(config)
    key = jax.random.PRNGKey(rule_seed)
    flat, unravel = ravel_pytree(game.rule(key))
    single = _rollout_with_weights(game, config, unravel)
    batch = jax.jit(jax.vmap(jax.vmap(single, in_axes=(None, 0)), in_axes=(0, None)))
    train_keys = jax.random.split(jax.random.PRNGKey(search.seed), search.train_worlds)
    # Reuse these initial states for every candidate; held-out keys never select weights.
    heldout_keys = jax.random.split(jax.random.PRNGKey(search.seed + 1), search.heldout_worlds)
    parent = np.asarray(flat, dtype=np.float32)
    rng = np.random.default_rng(search.seed)

    def assess(candidates, keys):
        _, trajectories = batch(jnp.asarray(candidates), keys)
        groups = [[trajectory_features(states, config) for states in candidate]
                  for candidate in np.asarray(trajectories)]
        return np.array([_fitness(rows, design) for rows in groups]), groups

    before_train, before_train_rows = assess(parent[None], train_keys)
    before_heldout, before_heldout_rows = assess(parent[None], heldout_keys)
    accepted = 0
    for generation in range(search.generations):
        scale = search.sigma * .99 ** generation
        perturbations = rng.normal(size=(search.population, len(parent))).astype(np.float32)
        candidates = np.concatenate((parent[None], parent[None] + scale * perturbations))
        scores, _ = assess(candidates, train_keys)
        winner = int(np.argmax(scores))
        if winner:
            parent = candidates[winner].copy()
            accepted += 1
    after_train, after_train_rows = assess(parent[None], train_keys)
    after_heldout, after_heldout_rows = assess(parent[None], heldout_keys)
    receipt = {
        "rule_seed": int(rule_seed), "design": asdict(design), "search": asdict(search),
        "accepted_updates": accepted,
        "before_train": {"fitness": float(before_train[0]), "worlds": before_train_rows[0]},
        "after_train": {"fitness": float(after_train[0]), "worlds": after_train_rows[0]},
        "before_heldout": {"fitness": float(before_heldout[0]), "worlds": before_heldout_rows[0]},
        "after_heldout": {"fitness": float(after_heldout[0]), "worlds": after_heldout_rows[0]},
    }
    return OptimizedRule(np.asarray(key, dtype=np.uint32), parent, receipt)


class NeuralRuleGenerator:
    """Roll out optimized neural rules without changing the original cell update."""

    def __init__(self, config: GameConfig, rules: list[OptimizedRule]):
        if len(rules) != config.rule_count:
            raise ValueError("rules must match config.rule_count")
        self.config = config
        self.rules = rules
        game = Game(config)
        template, unravel = ravel_pytree(game.rule(jax.random.PRNGKey(0)))
        if any(rule.weights.shape != template.shape or rule.rule_seed.shape != (2,)
               for rule in rules):
            raise ValueError("optimized rule shape does not match the game")
        self.rollout = _rollout_with_weights(game, config, unravel)

    def generate(self):
        """Yield ordinary v1 trajectory batches; only rule weights differ."""
        config = self.config
        states, ids, initial_seeds, initial_states = [], [], [], []
        for rule_id, rule in enumerate(self.rules):
            key = jnp.asarray(rule.rule_seed, dtype=jnp.uint32)
            for trajectory_id in range(config.trajectories_per_rule):
                initial_seed = jax.random.fold_in(key, trajectory_id + 1)
                initial, trajectory = self.rollout(jnp.asarray(rule.weights), initial_seed)
                states.append(np.asarray(trajectory, dtype=np.uint8))
                ids.append(rule_id)
                initial_seeds.append(np.asarray(initial_seed, dtype=np.uint32))
                initial_states.append(np.asarray(initial, dtype=np.uint8))
                if len(states) == config.batch_size:
                    yield TrajectoryBatch(np.stack(states), np.asarray(ids, np.int32),
                                          np.stack(initial_seeds), np.stack(initial_states))
                    states, ids, initial_seeds, initial_states = [], [], [], []
        if states:
            yield TrajectoryBatch(np.stack(states), np.asarray(ids, np.int32),
                                  np.stack(initial_seeds), np.stack(initial_states))

    def save(self, output: str | Path) -> Path:
        """Save v1 arrays plus learned weights and an explicit replay sidecar."""
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        implementation = implementation_metadata()
        implementation.update({
            "source": "SampleEff feature-optimized weights of the original upstream neural NCA",
            "neighborhood": "3x3 Moore, periodic wrap boundary",
            "network": "3x3 conv -> 1x1 conv -> ReLU -> 1x1 conv; optimized weights in sidecar",
            "seed_derivation": "rule seed initializes network; optimized weights override initialization; initial: fold_in(rule_seed, trajectory_id + 1)",
        })
        keys = np.stack([rule.rule_seed for rule in self.rules]).astype(np.uint32)
        saved = save_batches(self.config, output, keys, self.generate(),
                             implementation=implementation)
        # rules.npy retains v1 initialization keys; this sidecar holds the actual tuned rules.
        np.save(saved / WEIGHTS_FILE, np.stack([rule.weights for rule in self.rules]))
        (saved / "design.json").write_text(json.dumps({
            "schema": DESIGN_SCHEMA, "weights_file": WEIGHTS_FILE,
            "rules": [rule.receipt for rule in self.rules],
        }, indent=2) + "\n")
        return saved


def reproduce_optimized(directory: str | Path, record: dict, design: dict,
                        config: GameConfig):
    """Replay one saved optimized-rule world using its actual neural weights."""
    directory = Path(directory)
    weights = np.load(directory / design["weights_file"], mmap_mode="r")
    game = Game(config)
    template, unravel = ravel_pytree(game.rule(jax.random.PRNGKey(0)))
    if weights.shape != (config.rule_count, len(template)):
        raise ValueError("optimized neural-rule weights have the wrong shape")
    rollout = _rollout_with_weights(game, config, unravel)
    initial, states = rollout(jnp.asarray(weights[record["rule_id"]]),
                              jnp.asarray(record["initial_seed"], dtype=jnp.uint32))
    return np.asarray(initial, np.uint8), np.asarray(states, np.uint8)
