"""Sample disjoint high, unfiltered, and low patch-copy arms for one rule block.

Run from the repo root: uv run --all-packages python progress/nca_search/sample_patch_replication.py --block a
Chunks and packed outputs are resume-safe; no existing search corpus is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from nca import GameConfig
from nca.analyze import gzip_ratio
from nca.game import Game
from nca.generate import _rollout
from sampleeff_nca_pretraining.packing import PackedDataset, pack_one


REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "runs/1m-nca-transfer-v2/data/nca/40_50/packed.pt"
SOURCE = REPO / "runs/nca-100k-regimes"
ARMS = ("high", "unfiltered", "low")
BLOCK_SEEDS = {"a": 101, "b": 202, "c": 303}
THRESHOLD = -0.000374227071
TRAIN_WORLDS = 19_975
VAL_WORLDS = 256
EDGES = (.40, .42, .44, .46, .48, .50)
CHUNK_ATTEMPTS = 4_096
BATCH = 64
MAX_ATTEMPTS = 2_000_000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gzip_bin(ratio: float) -> int | None:
    if not .4 <= ratio < .5:
        return None
    return min(4, int((ratio - .4) / .02))


def hash_fraction(rule_seed: np.ndarray, person: bytes) -> float:
    return int.from_bytes(hashlib.blake2b(rule_seed.tobytes(), digest_size=8,
                                          person=person).digest(), "big") / 2**64


def split_for(rule_seed: np.ndarray) -> int:
    value = hash_fraction(rule_seed, b"nca-split")
    return 0 if value < .8 else 1 if value < .9 else 2


def arm_for(rule_seed: np.ndarray) -> int:
    return min(2, int(3 * hash_fraction(rule_seed, b"nca-arm")))


def horizontal_copy_excess(states: np.ndarray) -> float:
    cells = states[..., 0].astype(np.int32)
    patches = cells.reshape(27, 6, 2, 6, 2).transpose(0, 1, 3, 2, 4)
    tokens = patches.reshape(27, 6, 6, 4) @ np.array([1, 10, 100, 1000], dtype=np.int32)
    observed = np.mean(tokens[:, :, 1:] == tokens[:, :, :-1])
    counts = np.bincount(tokens.ravel(), minlength=10_000)
    chance = np.sum(counts * (counts - 1)) / (972 * 971)
    return float(observed - chance)


def quotas() -> np.ndarray:
    counts = np.zeros((2, 5), dtype=np.int64)
    seen = [0, 0]
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for row in json.loads(path.read_text())["accepted"]:
            if row["band"] != "40_50":
                continue
            split = {"train": 0, "validation": 1, "test": 2}[row["split"]]
            if split == 2 or seen[split] >= (TRAIN_WORLDS, VAL_WORLDS)[split]:
                continue
            bin_ = gzip_bin(row["gzip_ratio"])
            assert bin_ is not None
            counts[split, bin_] += 1
            seen[split] += 1
    assert seen == [TRAIN_WORLDS, VAL_WORLDS]
    return counts


def next_keys(cursor: jax.Array) -> tuple[jax.Array, np.ndarray]:
    def step(key, _):
        key, rule = jax.random.split(key)
        return key, rule
    following, rules = jax.lax.scan(step, cursor, None, length=BATCH)
    return following, np.asarray(rules, dtype=np.uint32)


def replay(chunks: Path, target: np.ndarray, stream_seed: int) -> tuple[jax.Array, int, np.ndarray]:
    cursor = jax.random.PRNGKey(stream_seed)
    attempt = 0
    counts = np.zeros((3, 2, 5), dtype=np.int64)
    for path in sorted(chunks.glob("chunk-*.npz")):
        with np.load(path) as data:
            assert int(data["start"]) == attempt, path
            attempt = int(data["end"])
            cursor = jnp.asarray(data["cursor_end"], dtype=jnp.uint32)
            for arm, split, bin_ in zip(data["arm"], data["split"], data["bin"]):
                counts[int(arm), int(split), int(bin_)] += 1
            assert np.all(counts <= target), path
    return cursor, attempt, counts


def save_chunk(chunks: Path, start: int, end: int, cursor: jax.Array, selected: list[tuple]) -> None:
    path = chunks / f"chunk-{start:012d}.npz"
    temporary = path.with_name(f".{path.name}.tmp")
    fields = {
        "start": np.asarray(start, dtype=np.int64),
        "end": np.asarray(end, dtype=np.int64),
        "cursor_end": np.asarray(cursor, dtype=np.uint32),
        "states": np.stack([v[0] for v in selected]) if selected else np.empty((0, 27, 12, 12, 1), np.uint8),
        "rule_seed": np.stack([v[1] for v in selected]) if selected else np.empty((0, 2), np.uint32),
        "initial_seed": np.stack([v[2] for v in selected]) if selected else np.empty((0, 2), np.uint32),
        "gzip_ratio": np.asarray([v[3] for v in selected], dtype=np.float64),
        "patch_copy_excess": np.asarray([v[4] for v in selected], dtype=np.float64),
        "arm": np.asarray([v[5] for v in selected], dtype=np.uint8),
        "split": np.asarray([v[6] for v in selected], dtype=np.uint8),
        "bin": np.asarray([v[7] for v in selected], dtype=np.uint8),
        "attempt": np.asarray([v[8] for v in selected], dtype=np.int64),
    }
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **fields)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def pack_arm(root: Path, arm: int, target: np.ndarray, attempt: int, game: GameConfig) -> None:
    directory = root / ARMS[arm]
    directory.mkdir(exist_ok=True)
    chunks = []
    for path in sorted((root / "chunks").glob("chunk-*.npz")):
        with np.load(path) as data:
            keep = data["arm"] == arm
            if np.any(keep):
                chunks.append({key: data[key][keep] for key in ("states", "rule_seed", "gzip_ratio",
                                                               "patch_copy_excess", "split", "bin")})
    states = np.concatenate([data["states"] for data in chunks])
    split = np.concatenate([data["split"] for data in chunks])
    ratios = np.concatenate([data["gzip_ratio"] for data in chunks])
    scores = np.concatenate([data["patch_copy_excess"] for data in chunks])
    rules = np.concatenate([data["rule_seed"] for data in chunks])
    assert len(states) == TRAIN_WORLDS + VAL_WORLDS
    assert len({tuple(v) for v in rules}) == len(rules)
    counts = np.zeros((2, 5), dtype=np.int64)
    for data in chunks:
        for s, b in zip(data["split"], data["bin"]):
            counts[int(s), int(b)] += 1
    assert np.array_equal(counts, target)
    assert np.all(scores >= THRESHOLD) if arm == 0 else True
    assert np.all(scores <= THRESHOLD) if arm == 2 else True

    patches = states.reshape(-1, 27, 6, 2, 6, 2).transpose(0, 1, 2, 4, 3, 5)
    patches = patches.reshape(-1, 27, 36, 4).astype(np.int32)
    tokens = (patches * np.array([1, 10, 100, 1000], dtype=np.int32)).sum(axis=-1, dtype=np.int32)
    sequences = np.empty((len(states), 27, 38), dtype=np.int32)
    sequences[:, :, 0] = 10_000
    sequences[:, :, 1:37] = tokens
    sequences[:, :, 37] = 10_001
    targets = sequences.copy()
    targets[:, :, 0] = -100
    targets[:, :, 37] = -100
    sequences = sequences.reshape(len(states), 1026)
    targets = targets.reshape(len(states), 1026)
    for index in (0, len(states) - 1):
        expected_seq, expected_target = pack_one(states[index], num_colors=10, patch_size=2)
        assert np.array_equal(sequences[index], expected_seq.numpy())
        assert np.array_equal(targets[index], expected_target.numpy())
    reference = PackedDataset.load(REFERENCE)
    metadata = {key: value for key, value in reference.metadata.items()
                if key in ("grid_len", "num_colors", "patch_size", "packing_config")}
    metadata.update({
        "split_indices": {"train": np.flatnonzero(split == 0).tolist(),
                          "validation": np.flatnonzero(split == 1).tolist(), "test": []},
        "usable_training_tokens": TRAIN_WORLDS * 987,
        "selection": f"patch-copy replication block {root.name}, arm {ARMS[arm]}",
        "patch_copy_threshold": THRESHOLD,
        "gzip_subbin_edges": list(EDGES),
        "gzip_subbin_quotas": target.tolist(),
        "source_generator": asdict(game),
        "source_reference_sha256": sha(REFERENCE),
    })
    output = directory / "packed.pt"
    temporary = directory / ".packed.pt.tmp"
    PackedDataset(torch.from_numpy(sequences), torch.from_numpy(targets), metadata).save(temporary)
    os.replace(temporary, output)
    write_json(directory / "done.json", {
        "schema": "nca-patch-replication-arm-v1", "arm": ARMS[arm], "block": root.name,
        "packed_sha256": sha(output), "reference_packed_sha256": sha(REFERENCE),
        "attempts": attempt, "selected": len(states), "train_worlds": TRAIN_WORLDS,
        "validation_worlds": VAL_WORLDS, "train_usable_tokens": TRAIN_WORLDS * 987,
        "gzip_bin_counts": counts.tolist(), "gzip_ratio_mean": float(ratios.mean()),
        "patch_copy_excess_mean": float(scores.mean()), "patch_copy_threshold": THRESHOLD,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=BLOCK_SEEDS, required=True)
    args = parser.parse_args()
    stream_seed = BLOCK_SEEDS[args.block]
    root = REPO / "runs/nca-data-search/patch-replication" / f"block-{args.block}"
    chunks = root / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    target = quotas()
    game = replace(GameConfig(rule_count=BATCH, trajectories_per_rule=1, batch_size=BATCH),
                   seed=stream_seed)
    config = {"schema": "nca-patch-replication-v1", "block": args.block,
              "stream_seed": stream_seed, "arm_hash_person": "nca-arm",
              "split_hash_person": "nca-split", "arms": ARMS, "threshold": THRESHOLD,
              "gzip_edges": EDGES, "quotas_per_arm": target.tolist(),
              "reference_sha256": sha(REFERENCE), "game": asdict(game),
              "max_attempts": MAX_ATTEMPTS}
    config_path = root / "sampling_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == json.loads(json.dumps(config))
    else:
        write_json(config_path, config)
    if all((root / arm / "done.json").is_file() for arm in ARMS):
        for arm in ARMS:
            receipt = json.loads((root / arm / "done.json").read_text())
            assert sha(root / arm / "packed.pt") == receipt["packed_sha256"]
        print("SKIP complete block", root, flush=True)
        return
    cursor, attempt, counts = replay(chunks, target, stream_seed)
    rollout = _rollout(Game(game), game, game.frames)
    started = time.monotonic()
    while not np.all(counts == target):
        if attempt >= MAX_ATTEMPTS:
            raise RuntimeError(f"quota not met after {attempt} attempts: {counts.tolist()}")
        start = attempt
        selected = []
        for _ in range(CHUNK_ATTEMPTS // BATCH):
            if np.all(counts == target):
                break
            cursor, rules = next_keys(cursor)
            for index, rule_seed in enumerate(rules):
                arm = arm_for(rule_seed)
                split = split_for(rule_seed)
                if split > 1 or np.all(counts[arm, split] == target[split]):
                    continue
                key = jnp.asarray(rule_seed, dtype=jnp.uint32)
                initial_seed = jax.random.fold_in(key, 1)
                _, trajectory = rollout(key, initial_seed)
                states = np.asarray(trajectory, dtype=np.uint8)
                ratio = gzip_ratio(states, game.num_colors, game.patch_size)
                bin_ = gzip_bin(ratio)
                if bin_ is None or counts[arm, split, bin_] >= target[split, bin_]:
                    continue
                score = horizontal_copy_excess(states)
                if arm == 0 and score < THRESHOLD or arm == 2 and score > THRESHOLD:
                    continue
                selected.append((states.copy(), rule_seed.copy(), np.asarray(initial_seed, dtype=np.uint32),
                                 ratio, score, arm, split, bin_, attempt + index))
                counts[arm, split, bin_] += 1
            attempt += BATCH
        save_chunk(chunks, start, attempt, cursor, selected)
        print(f"attempts={attempt}; train={[int(v) for v in counts[:,0,:].sum(axis=1)]}/{TRAIN_WORLDS}; "
              f"validation={[int(v) for v in counts[:,1,:].sum(axis=1)]}/{VAL_WORLDS}; "
              f"elapsed={time.monotonic()-started:.0f}s", flush=True)
    for arm in range(3):
        pack_arm(root, arm, target, attempt, game)
        print("DONE", ARMS[arm], root / ARMS[arm] / "packed.pt", flush=True)


if __name__ == "__main__":
    main()
