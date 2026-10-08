"""Save raw, memory-mappable NCA trajectories."""

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import jax
import numpy as np

from .config import GameConfig

UPSTREAM_COMMIT = "bdd1c71e04eb2e5a2c91481faf278e95846b4431"


def save(config: GameConfig, output: str | Path) -> Path:
    """Write trajectories and replay metadata to a new directory."""
    from .generate import generate, sample_rules

    rules = sample_rules(config)
    return save_batches(config, output, rules, generate(config, rules))


def save_batches(config: GameConfig, output: str | Path, rules: np.ndarray,
                 batches) -> Path:
    """Write already-generated batches using the unchanged version 1 schema."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rules = np.asarray(rules, dtype=np.uint32)
    if rules.shape != (config.rule_count, 2):
        raise ValueError(f"rules must have shape ({config.rule_count}, 2)")
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "rules.npy", rules)
    shards = []
    written = 0
    for index, batch in enumerate(batches):
        count = len(batch.states)
        if (batch.states.dtype != np.uint8
                or batch.rule_ids.dtype != np.int32
                or batch.initial_seeds.dtype != np.uint32
                or batch.initial_states.dtype != np.uint8
                or batch.states.shape != (count, config.frames, config.grid_size,
                                          config.grid_size, config.groups)
                or batch.rule_ids.shape != (count,)
                or batch.initial_seeds.shape != (count, 2)
                or batch.initial_states.shape != (count, config.grid_size,
                                                  config.grid_size, config.groups)):
            raise ValueError("batch does not match version 1 trajectory layout")
        if np.any(batch.rule_ids < 0) or np.any(batch.rule_ids >= config.rule_count):
            raise ValueError("batch contains a rule ID outside rules.npy")
        stem = f"{index:06d}"
        files = {
            "states": f"trajectories-{stem}.npy",
            "rule_ids": f"rule-ids-{stem}.npy",
            "initial_seeds": f"initial-seeds-{stem}.npy",
            "initial_states": f"initial-states-{stem}.npy",
        }
        for field, filename in files.items():
            np.save(output / filename, getattr(batch, field))
        shards.append({**files, "count": count})
        written += count
    if written != config.rule_count * config.trajectories_per_rule:
        raise ValueError("batch count does not match GameConfig")
    manifest = {
        "schema_version": 1,
        "config": asdict(config),
        "implementation": {
            "source": "https://github.com/danihyunlee/nca-pre-pretraining",
            "upstream_commit": UPSTREAM_COMMIT,
            "jax": jax.__version__,
            "flax": version("flax"),
            "backend": jax.default_backend(),
            "neighborhood": "3x3 Moore, periodic wrap boundary",
            "network": "3x3 conv -> 1x1 conv -> ReLU -> 1x1 conv",
            "initial_state": "one categorical draw per cell from grid-shared random logits",
            "seed_derivation": "rule: successive jax.random.split from config seed; initial: fold_in(rule_seed, trajectory_id + 1)",
        },
        "shape": [config.frames, config.grid_size, config.grid_size, config.groups],
        "frame_times": [config.burn_in + i * config.stride for i in range(config.frames)],
        "rules": "rules.npy",
        "shards": shards,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
