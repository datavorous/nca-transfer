"""Sample disjoint high, unfiltered, and low vertical lag-6 arms for one block.

Run from the repo root: uv run --all-packages python progress/nca_search/sample_vertical_lag6.py --block d
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
BLOCK_SEEDS = {"d": 404, "e": 505, "f": 606}
THRESHOLD = -0.0005878894767783627
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
    # Preserve the existing sampler's exact floating-point bin assignment.
    # The frozen reference has no worlds exactly on an internal bin edge.
    return min(4, int((ratio - .4) / .02))


def hash_fraction(rule_seed: np.ndarray, person: bytes) -> float:
    return int.from_bytes(hashlib.blake2b(rule_seed.tobytes(), digest_size=8,
                                          person=person).digest(), "big") / 2**64


def split_for(rule_seed: np.ndarray) -> int:
    value = hash_fraction(rule_seed, b"nca-split")
    return 0 if value < .8 else 1 if value < .9 else 2


def arm_for(rule_seed: np.ndarray) -> int:
    return min(2, int(3 * hash_fraction(rule_seed, b"nca-v6-arm")))


def vertical_excess(states: np.ndarray) -> float:
    cells = states[..., 0].astype(np.int32)
    patches = cells.reshape(27, 6, 2, 6, 2).transpose(0, 1, 3, 2, 4)
    tokens = patches.reshape(27, 6, 6, 4) @ np.array([1, 10, 100, 1000], dtype=np.int32)
    observed = (tokens[:, 1:, :] == tokens[:, :-1, :]).mean()
    frame_counts = np.stack([np.bincount(frame.ravel(), minlength=10_000)
                             for frame in tokens]).astype(np.int64)
    chance = ((frame_counts * (frame_counts - 1)).sum(axis=1) / (36 * 35)).mean()
    return float(observed - chance)


def quotas() -> np.ndarray:
    # Frozen from the reference in VERTICAL_LAG6_FROZEN_PROTOCOL.md.
    return np.array([[2913, 3544, 4034, 4559, 4925],
                     [45, 42, 49, 57, 63]], dtype=np.int64)


def prior_rules(block: str) -> set[tuple[int, int]]:
    """Rule IDs already accepted by the reference or any other search corpus."""
    seen: set[tuple[int, int]] = set()
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for row in json.loads(path.read_text())["accepted"]:
            seen.add(tuple(map(int, row["rule_seed"])))
    search = REPO / "runs/nca-data-search"
    for directory in sorted(search.glob("iteration-*")) + sorted((search / "patch-replication").glob("block-*")) + sorted((search / "vertical-lag6").glob("block-*")):
        if directory == search / "vertical-lag6" / f"block-{block}":
            continue
        for path in sorted((directory / "chunks").glob("chunk-*.npz")):
            with np.load(path) as data:
                seen.update(tuple(map(int, key)) for key in data["rule_seed"])
    return seen


def own_rules(chunks: Path) -> set[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    for path in sorted(chunks.glob("chunk-*.npz")):
        with np.load(path) as data:
            for key in data["rule_seed"]:
                rule_id = tuple(map(int, key))
                assert rule_id not in seen, (path, rule_id)
                seen.add(rule_id)
    return seen


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


def save_chunk(chunks: Path, start: int, end: int, cursor: jax.Array,
               selected: list[tuple], overlap_rejections: int) -> None:
    path = chunks / f"chunk-{start:012d}.npz"
    temporary = path.with_name(f".{path.name}.tmp")
    fields = {
        "start": np.asarray(start, dtype=np.int64),
        "end": np.asarray(end, dtype=np.int64),
        "overlap_rejections": np.asarray(overlap_rejections, dtype=np.int64),
        "cursor_end": np.asarray(cursor, dtype=np.uint32),
        "states": np.stack([v[0] for v in selected]) if selected else np.empty((0, 27, 12, 12, 1), np.uint8),
        "rule_seed": np.stack([v[1] for v in selected]) if selected else np.empty((0, 2), np.uint32),
        "initial_seed": np.stack([v[2] for v in selected]) if selected else np.empty((0, 2), np.uint32),
        "gzip_ratio": np.asarray([v[3] for v in selected], dtype=np.float64),
        "vertical_excess": np.asarray([v[4] for v in selected], dtype=np.float64),
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
    done_path = directory / "done.json"
    if done_path.exists():
        receipt = json.loads(done_path.read_text())
        assert receipt["schema"] == "nca-vertical-lag6-arm-v1"
        assert receipt["arm"] == ARMS[arm] and receipt["block"] == root.name
        assert receipt["attempts"] == attempt
        assert receipt["gzip_bin_counts"] == target.tolist()
        assert sha(directory / "packed.pt") == receipt["packed_sha256"]
        return
    chunks = []
    for path in sorted((root / "chunks").glob("chunk-*.npz")):
        with np.load(path) as data:
            keep = data["arm"] == arm
            if np.any(keep):
                chunks.append({key: data[key][keep] for key in ("states", "rule_seed", "gzip_ratio",
                                                               "vertical_excess", "split", "bin")})
    states = np.concatenate([data["states"] for data in chunks])
    split = np.concatenate([data["split"] for data in chunks])
    ratios = np.concatenate([data["gzip_ratio"] for data in chunks])
    scores = np.concatenate([data["vertical_excess"] for data in chunks])
    rules = np.concatenate([data["rule_seed"] for data in chunks])
    assert len(states) == TRAIN_WORLDS + VAL_WORLDS
    assert len({tuple(v) for v in rules}) == len(rules)
    counts = np.zeros((2, 5), dtype=np.int64)
    for data in chunks:
        for s, b in zip(data["split"], data["bin"]):
            counts[int(s), int(b)] += 1
    assert np.array_equal(counts, target)
    assert np.all(scores >= THRESHOLD) if arm == 0 else True
    assert np.all(scores < THRESHOLD) if arm == 2 else True

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
        "selection": f"vertical lag-6 block {root.name}, arm {ARMS[arm]}",
        "vertical_excess_threshold": THRESHOLD,
        "gzip_subbin_edges": list(EDGES),
        "gzip_subbin_quotas": target.tolist(),
        "source_generator": asdict(game),
        "source_reference_sha256": sha(REFERENCE),
    })
    output = directory / "packed.pt"
    temporary = directory / ".packed.pt.tmp"
    PackedDataset(torch.from_numpy(sequences), torch.from_numpy(targets), metadata).save(temporary)
    os.replace(temporary, output)
    write_json(done_path, {
        "schema": "nca-vertical-lag6-arm-v1", "arm": ARMS[arm], "block": root.name,
        "packed_sha256": sha(output), "reference_packed_sha256": sha(REFERENCE),
        "attempts": attempt, "selected": len(states), "train_worlds": TRAIN_WORLDS,
        "validation_worlds": VAL_WORLDS, "train_usable_tokens": TRAIN_WORLDS * 987,
        "gzip_bin_counts": counts.tolist(), "gzip_ratio_mean": float(ratios.mean()),
        "vertical_excess_mean": float(scores.mean()), "vertical_excess_threshold": THRESHOLD,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=BLOCK_SEEDS, required=True)
    args = parser.parse_args()
    stream_seed = BLOCK_SEEDS[args.block]
    root = REPO / "runs/nca-data-search/vertical-lag6" / f"block-{args.block}"
    chunks = root / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    target = quotas()
    game = replace(GameConfig(rule_count=BATCH, trajectories_per_rule=1, batch_size=BATCH),
                   seed=stream_seed)
    freeze_path = REPO / "progress/nca_search/vertical_lag6_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    assert freeze["threshold"] == THRESHOLD
    feature_path = REPO / "progress/nca_search/screen_vertical_lag_target.py"
    assert sha(feature_path) == freeze["feature_code_sha256"]
    assert sha(REFERENCE) == "1ba5e8ce68c70faf19183170ac107959d5348e8db220898baa20bb9668d7ffbf"
    config = {"schema": "nca-vertical-lag6-v1", "block": args.block,
              "stream_seed": stream_seed, "arm_hash_person": "nca-v6-arm",
              "split_hash_person": "nca-split", "arms": ARMS, "threshold": THRESHOLD,
              "gzip_edges": EDGES, "quotas_per_arm": target.tolist(),
              "reference_sha256": sha(REFERENCE), "game": asdict(game),
              "freeze_sha256": sha(freeze_path),
              "reference_source_bundle_sha256": freeze["reference_source_bundle_sha256"],
              "feature_code_sha256": freeze["feature_code_sha256"],
              "sampler_code_sha256": sha(Path(__file__)),
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
    previous = prior_rules(args.block)
    own = own_rules(chunks)
    assert not (previous & own), "existing block overlaps another corpus"
    rollout = _rollout(Game(game), game, game.frames)
    started = time.monotonic()
    while not np.all(counts == target):
        if attempt >= MAX_ATTEMPTS:
            raise RuntimeError(f"quota not met after {attempt} attempts: {counts.tolist()}")
        start = attempt
        selected = []
        overlap_rejections = 0
        for _ in range(min(CHUNK_ATTEMPTS, MAX_ATTEMPTS - attempt) // BATCH):
            if np.all(counts == target):
                break
            cursor, rules = next_keys(cursor)
            for index, rule_seed in enumerate(rules):
                rule_id = tuple(map(int, rule_seed))
                if rule_id in previous or rule_id in own:
                    overlap_rejections += 1
                    continue
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
                score = vertical_excess(states)
                if arm == 0 and score < THRESHOLD or arm == 2 and score >= THRESHOLD:
                    continue
                selected.append((states.copy(), rule_seed.copy(), np.asarray(initial_seed, dtype=np.uint32),
                                 ratio, score, arm, split, bin_, attempt + index))
                own.add(rule_id)
                counts[arm, split, bin_] += 1
            attempt += BATCH
        save_chunk(chunks, start, attempt, cursor, selected, overlap_rejections)
        print(f"attempts={attempt}; train={[int(v) for v in counts[:,0,:].sum(axis=1)]}/{TRAIN_WORLDS}; "
              f"validation={[int(v) for v in counts[:,1,:].sum(axis=1)]}/{VAL_WORLDS}; "
              f"prior-rule-rejections={overlap_rejections}; "
              f"elapsed={time.monotonic()-started:.0f}s", flush=True)
    for arm in range(3):
        pack_arm(root, arm, target, attempt, game)
        print("DONE", ARMS[arm], root / ARMS[arm] / "packed.pt", flush=True)


if __name__ == "__main__":
    main()
