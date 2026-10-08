"""Extend frozen block-A high-arm rules into top-25/top-10 cohorts.

Run from repo root: uv run --all-packages python progress/boosted_nca/sample_sweep.py
Only writes runs/nca-data-search/boosted-nca/sweep-a/.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "progress/nca_search"))
import sample_patch_replication as source
from nca import GameConfig
from nca.analyze import gzip_ratio
from nca.game import Game
from nca.generate import _rollout
from sampleeff_nca_pretraining.packing import PackedDataset, pack_one


ANCHOR = REPO / "runs/nca-data-search/patch-replication/block-a"
ROOT = REPO / "runs/nca-data-search/boosted-nca/sweep-a"
CHUNKS = ROOT / "chunks"
TIERS = ("top_25", "top_10")
CUTOFFS = np.array([
    [.013787809437, .009925705543, .006992070455, .004674553831, .003439879976],
    [.034107767225, .025746229122, .017753705187, .012177637072, .008144630498],
], dtype=np.float64)
MAX_ATTEMPTS = 5_000_000
CHUNK_ATTEMPTS = 4_096
BATCH = 64


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def row_arrays(rows: list[tuple]) -> dict[str, np.ndarray]:
    return {
        "states": np.stack([r[0] for r in rows]) if rows else np.empty((0, 27, 12, 12, 1), np.uint8),
        "rule_seed": np.stack([r[1] for r in rows]) if rows else np.empty((0, 2), np.uint32),
        "initial_seed": np.stack([r[2] for r in rows]) if rows else np.empty((0, 2), np.uint32),
        "gzip_ratio": np.asarray([r[3] for r in rows], dtype=np.float64),
        "patch_copy_excess": np.asarray([r[4] for r in rows], dtype=np.float64),
        "split": np.asarray([r[5] for r in rows], dtype=np.uint8),
        "bin": np.asarray([r[6] for r in rows], dtype=np.uint8),
        "attempt": np.asarray([r[7] for r in rows], dtype=np.int64),
        "keep_25": np.asarray([r[8] for r in rows], dtype=bool),
        "keep_10": np.asarray([r[9] for r in rows], dtype=bool),
    }


def save_chunk(start: int, end: int, cursor: jax.Array, rows: list[tuple]) -> None:
    CHUNKS.mkdir(parents=True, exist_ok=True)
    path = CHUNKS / f"chunk-{start:012d}.npz"
    temporary = path.with_name(f".{path.name}.tmp")
    arrays = row_arrays(rows)
    arrays.update(start=np.asarray(start, np.int64), end=np.asarray(end, np.int64),
                  cursor_end=np.asarray(cursor, np.uint32))
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def add(rows: list[tuple], counts: np.ndarray, target: np.ndarray, states: np.ndarray,
        rule_seed: np.ndarray, initial_seed: np.ndarray, ratio: float, score: float,
        split: int, bin_: int, attempt: int) -> None:
    keep = [(counts[t, split, bin_] < target[split, bin_]) and score >= CUTOFFS[t, bin_]
            for t in range(2)]
    if not any(keep):
        return
    counts[:, split, bin_] += np.asarray(keep, np.int64)
    rows.append((states.copy(), rule_seed.copy(), initial_seed.copy(), ratio, score,
                 split, bin_, attempt, keep[0], keep[1]))


def anchor_initial_chunk(target: np.ndarray) -> tuple[int, jax.Array]:
    source_paths = sorted((ANCHOR / "chunks").glob("chunk-*.npz"))
    assert source_paths, "completed block-A chunks required"
    counts = np.zeros((2, 2, 5), np.int64)
    rows: list[tuple] = []
    end = 0
    cursor = None
    for path in source_paths:
        with np.load(path) as data:
            assert int(data["start"]) == end
            end = int(data["end"])
            cursor = jnp.asarray(data["cursor_end"], dtype=jnp.uint32)
            for i in np.flatnonzero(data["arm"] == 0):
                split, bin_ = int(data["split"][i]), int(data["bin"][i])
                add(rows, counts, target, data["states"][i], data["rule_seed"][i],
                    data["initial_seed"][i], float(data["gzip_ratio"][i]),
                    float(data["patch_copy_excess"][i]), split, bin_, int(data["attempt"][i]))
    assert cursor is not None and end == json.loads((ANCHOR / "high/done.json").read_text())["attempts"]
    save_chunk(0, end, cursor, rows)
    print(f"anchor candidates: attempts={end}, tier counts={counts.sum(axis=(1,2)).tolist()}", flush=True)
    return end, cursor


def replay(target: np.ndarray) -> tuple[int, jax.Array, np.ndarray]:
    attempt = 0
    cursor = None
    counts = np.zeros((2, 2, 5), np.int64)
    for path in sorted(CHUNKS.glob("chunk-*.npz")):
        with np.load(path) as data:
            assert int(data["start"]) == attempt, path
            attempt = int(data["end"])
            cursor = jnp.asarray(data["cursor_end"], dtype=jnp.uint32)
            for t, key in enumerate(("keep_25", "keep_10")):
                for split, bin_ in zip(data["split"][data[key]], data["bin"][data[key]]):
                    counts[t, int(split), int(bin_)] += 1
            assert np.all(counts <= target), path
    assert cursor is not None
    return attempt, cursor, counts


def pack_tier(tier: int, target: np.ndarray, attempt: int, game: GameConfig) -> None:
    directory = ROOT / TIERS[tier]
    directory.mkdir(parents=True, exist_ok=True)
    key = ("keep_25", "keep_10")[tier]
    fields = ("states", "rule_seed", "initial_seed", "gzip_ratio", "patch_copy_excess", "split", "bin", "attempt")
    pieces = []
    for path in sorted(CHUNKS.glob("chunk-*.npz")):
        with np.load(path) as data:
            keep = data[key]
            if np.any(keep):
                pieces.append({field: data[field][keep] for field in fields})
    values = {field: np.concatenate([piece[field] for piece in pieces]) for field in fields}
    states, splits, bins = values["states"], values["split"], values["bin"]
    assert len(states) == 20_231
    assert len({tuple(seed) for seed in values["rule_seed"]}) == len(states)
    assert np.all(values["patch_copy_excess"] >= CUTOFFS[tier, bins])
    assert np.all(np.diff(values["attempt"]) > 0)
    observed = np.zeros((2, 5), np.int64)
    for split, bin_ in zip(splits, bins):
        observed[int(split), int(bin_)] += 1
    assert np.array_equal(observed, target)
    for i in (0, len(states) // 2, len(states)-1):
        assert np.isclose(source.horizontal_copy_excess(states[i]), values["patch_copy_excess"][i])
        assert source.gzip_bin(float(values["gzip_ratio"][i])) == int(bins[i])

    patches = states.reshape(-1, 27, 6, 2, 6, 2).transpose(0, 1, 2, 4, 3, 5)
    patches = patches.reshape(-1, 27, 36, 4).astype(np.int32)
    tokens = (patches * np.array([1, 10, 100, 1000], np.int32)).sum(axis=-1, dtype=np.int32)
    sequences = np.empty((len(states), 27, 38), np.int32)
    sequences[:, :, 0] = 10_000
    sequences[:, :, 1:37] = tokens
    sequences[:, :, 37] = 10_001
    targets = sequences.copy()
    targets[:, :, 0] = -100
    targets[:, :, 37] = -100
    sequences = sequences.reshape(len(states), 1026)
    targets = targets.reshape(len(states), 1026)
    for i in (0, len(states) // 2, len(states)-1):
        expected_x, expected_y = pack_one(states[i], num_colors=10, patch_size=2)
        assert np.array_equal(sequences[i], expected_x.numpy())
        assert np.array_equal(targets[i], expected_y.numpy())
    reference = PackedDataset.load(source.REFERENCE)
    metadata = {key: value for key, value in reference.metadata.items()
                if key in ("grid_len", "num_colors", "patch_size", "packing_config")}
    metadata.update({
        "split_indices": {"train": np.flatnonzero(splits == 0).tolist(),
                          "validation": np.flatnonzero(splits == 1).tolist(), "test": []},
        "usable_training_tokens": 19_715_325,
        "selection": f"block-A high-arm horizontal recurrence {TIERS[tier]}",
        "patch_copy_thresholds_by_gzip_bin": CUTOFFS[tier].tolist(),
        "gzip_subbin_edges": list(source.EDGES),
        "gzip_subbin_quotas": target.tolist(),
        "source_generator": asdict(game),
        "source_reference_sha256": source.sha(source.REFERENCE),
        "source_anchor_packed_sha256": source.sha(ANCHOR / "high/packed.pt"),
    })
    output = directory / "packed.pt"
    temporary = directory / ".packed.pt.tmp"
    PackedDataset(torch.from_numpy(sequences), torch.from_numpy(targets), metadata).save(temporary)
    os.replace(temporary, output)
    anchor_rules = set()
    for path in sorted((ANCHOR / "chunks").glob("chunk-*.npz")):
        with np.load(path) as data:
            anchor_rules.update(tuple(seed) for seed in data["rule_seed"][data["arm"] == 0])
    overlap = sum(tuple(seed) in anchor_rules for seed in values["rule_seed"])
    receipt = {
        "schema": "nca-boosted-sweep-arm-v1", "tier": TIERS[tier],
        "source_block": "a", "source_arm": "high", "seed": 101,
        "attempts": attempt, "selected": len(states), "train_worlds": 19_975,
        "validation_worlds": 256, "train_usable_tokens": 19_715_325,
        "gzip_bin_counts": observed.tolist(), "gzip_ratio_mean": float(values["gzip_ratio"].mean()),
        "patch_copy_excess_mean": float(values["patch_copy_excess"].mean()),
        "cutoffs_by_bin": CUTOFFS[tier].tolist(), "anchor_rule_overlap": overlap,
        "packed_sha256": source.sha(output),
        "sampling_config_sha256": source.sha(ROOT / "sampling_config.json"),
    }
    atomic_json(directory / "done.json", receipt)
    print(f"DONE {TIERS[tier]}: {output}; anchor overlap={overlap}", flush=True)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    target = source.quotas()
    assert target.shape == (2, 5) and target.sum(axis=1).tolist() == [19_975, 256]
    anchor_config = json.loads((ANCHOR / "sampling_config.json").read_text())
    assert anchor_config["stream_seed"] == 101 and anchor_config["threshold"] == source.THRESHOLD
    assert source.sha(ANCHOR / "high/packed.pt") == json.loads((ANCHOR / "high/done.json").read_text())["packed_sha256"]
    game = replace(GameConfig(rule_count=BATCH, trajectories_per_rule=1, batch_size=BATCH), seed=101)
    config = {
        "schema": "nca-boosted-sweep-v1", "source_block": "a", "source_arm": "high",
        "stream_seed": 101, "arm_hash_person": "nca-arm", "split_hash_person": "nca-split",
        "tiers": TIERS, "cutoffs_by_gzip_bin": CUTOFFS.tolist(),
        "gzip_edges": source.EDGES, "quotas_per_tier": target.tolist(),
        "max_attempts": MAX_ATTEMPTS, "source_anchor_packed_sha256": source.sha(ANCHOR / "high/packed.pt"),
        "source_sampling_config_sha256": source.sha(ANCHOR / "sampling_config.json"),
        "game": asdict(game),
    }
    config_path = ROOT / "sampling_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == json.loads(json.dumps(config))
    else:
        atomic_json(config_path, config)
    if all((ROOT / tier / "done.json").is_file() for tier in TIERS):
        for tier in TIERS:
            receipt = json.loads((ROOT / tier / "done.json").read_text())
            assert source.sha(ROOT / tier / "packed.pt") == receipt["packed_sha256"]
        print("SKIP completed sweep corpora", flush=True)
        return
    if not list(CHUNKS.glob("chunk-*.npz")):
        anchor_initial_chunk(target)
    attempt, cursor, counts = replay(target)
    rollout = _rollout(Game(game), game, game.frames)
    started = time.monotonic()
    while not np.all(counts == target):
        if attempt >= MAX_ATTEMPTS:
            raise RuntimeError(f"quota not met after {attempt} attempts: {counts.tolist()}")
        start = attempt
        rows: list[tuple] = []
        for _ in range(CHUNK_ATTEMPTS // BATCH):
            if np.all(counts == target):
                break
            cursor, rules = source.next_keys(cursor)
            for index, rule_seed in enumerate(rules):
                if source.arm_for(rule_seed) != 0:
                    continue
                split = source.split_for(rule_seed)
                if split > 1 or np.all(counts[:, split, :] == target[split]):
                    continue
                key = jnp.asarray(rule_seed, dtype=jnp.uint32)
                initial_seed = jax.random.fold_in(key, 1)
                _, trajectory = rollout(key, initial_seed)
                states = np.asarray(trajectory, np.uint8)
                ratio = gzip_ratio(states, game.num_colors, game.patch_size)
                bin_ = source.gzip_bin(ratio)
                if bin_ is None or np.all(counts[:, split, bin_] == target[split, bin_]):
                    continue
                score = source.horizontal_copy_excess(states)
                add(rows, counts, target, states, rule_seed,
                    np.asarray(initial_seed, np.uint32), ratio, score,
                    split, bin_, attempt + index)
            attempt += BATCH
        save_chunk(start, attempt, cursor, rows)
        print(f"attempts={attempt}; train={counts[:,0,:].sum(axis=1).tolist()}/19975; "
              f"validation={counts[:,1,:].sum(axis=1).tolist()}/256; "
              f"elapsed={time.monotonic()-started:.0f}s", flush=True)
    for tier in range(2):
        pack_tier(tier, target, attempt, game)


if __name__ == "__main__":
    main()
