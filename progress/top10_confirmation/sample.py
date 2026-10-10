"""Generate one frozen fresh Designed/Ordinary corpus pair; resumable chunks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
import sample_patch_replication as prior
from nca import GameConfig
from nca.analyze import gzip_ratio
from nca.game import Game
from nca.generate import _rollout
from sampleeff_nca_pretraining.packing import PackedDataset, pack_one

ROOT = REPO / "runs/nca-data-search/top10-confirmation"
STREAMS = {"a": 8101, "b": 8202, "c": 8303}
ARMS = ("designed", "ordinary")
CUTOFFS = np.array([.034107767225, .025746229122, .017753705187,
                    .012177637072, .008144630498], dtype=np.float64)
MAX_ATTEMPTS = 8_000_000
CHUNK_ATTEMPTS = 4096
BATCH = 64


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def arm_for(rule_seed: np.ndarray) -> int:
    return min(1, int(2 * prior.hash_fraction(rule_seed, b"nca-confirm")))


def replay(chunks: Path, target: np.ndarray, stream_seed: int):
    cursor = jax.random.PRNGKey(stream_seed)
    attempt = 0
    counts = np.zeros((2, 2, 5), np.int64)
    for path in sorted(chunks.glob("chunk-*.npz")):
        with np.load(path) as data:
            assert int(data["start"]) == attempt, path
            attempt = int(data["end"])
            cursor = jnp.asarray(data["cursor_end"], dtype=jnp.uint32)
            for arm, split, bin_ in zip(data["arm"], data["split"], data["bin"]):
                counts[int(arm), int(split), int(bin_)] += 1
            assert np.all(counts <= target), path
    return cursor, attempt, counts


def save_chunk(chunks: Path, start: int, end: int, cursor, rows: list[tuple]) -> None:
    fields = {
        "start": np.asarray(start, np.int64),
        "end": np.asarray(end, np.int64),
        "cursor_end": np.asarray(cursor, np.uint32),
        "states": np.stack([v[0] for v in rows]) if rows else np.empty((0, 27, 12, 12, 1), np.uint8),
        "rule_seed": np.stack([v[1] for v in rows]) if rows else np.empty((0, 2), np.uint32),
        "initial_seed": np.stack([v[2] for v in rows]) if rows else np.empty((0, 2), np.uint32),
        "gzip_ratio": np.asarray([v[3] for v in rows], np.float64),
        "patch_copy_excess": np.asarray([v[4] for v in rows], np.float64),
        "arm": np.asarray([v[5] for v in rows], np.uint8),
        "split": np.asarray([v[6] for v in rows], np.uint8),
        "bin": np.asarray([v[7] for v in rows], np.uint8),
        "attempt": np.asarray([v[8] for v in rows], np.int64),
    }
    path = chunks / f"chunk-{start:012d}.npz"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **fields)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pack(root: Path, arm: int, target: np.ndarray, attempt: int, game: GameConfig) -> None:
    directory = root / ARMS[arm]
    directory.mkdir(exist_ok=True)
    fields = ("states", "rule_seed", "gzip_ratio", "patch_copy_excess", "split", "bin", "attempt")
    pieces = []
    for path in sorted((root / "chunks").glob("chunk-*.npz")):
        with np.load(path) as data:
            keep = data["arm"] == arm
            if np.any(keep):
                pieces.append({field: data[field][keep] for field in fields})
    values = {field: np.concatenate([piece[field] for piece in pieces]) for field in fields}
    states, splits, bins, scores = (values[key] for key in ("states", "split", "bin", "patch_copy_excess"))
    assert len(states) == 20_231
    assert len({tuple(v) for v in values["rule_seed"]}) == len(states)
    assert np.all(np.diff(values["attempt"]) > 0)
    if arm == 0:
        assert np.all(scores >= CUTOFFS[bins])
    observed = np.zeros((2, 5), np.int64)
    for split, bin_ in zip(splits, bins):
        observed[int(split), int(bin_)] += 1
    assert np.array_equal(observed, target)
    for i in (0, len(states)//2, len(states)-1):
        assert np.isclose(prior.horizontal_copy_excess(states[i]), scores[i])
        assert prior.gzip_bin(float(values["gzip_ratio"][i])) == int(bins[i])

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
    for i in (0, len(states)//2, len(states)-1):
        expected_x, expected_y = pack_one(states[i], num_colors=10, patch_size=2)
        assert np.array_equal(sequences[i], expected_x.numpy())
        assert np.array_equal(targets[i], expected_y.numpy())
    reference = PackedDataset.load(prior.REFERENCE)
    metadata = {key: value for key, value in reference.metadata.items()
                if key in ("grid_len", "num_colors", "patch_size", "packing_config")}
    metadata.update({
        "split_indices": {"train": np.flatnonzero(splits == 0).tolist(),
                          "validation": np.flatnonzero(splits == 1).tolist(), "test": []},
        "usable_training_tokens": 19_715_325,
        "selection": f"fresh top10 confirmation block {root.name}, arm {ARMS[arm]}",
        "patch_copy_thresholds_by_gzip_bin": CUTOFFS.tolist() if arm == 0 else None,
        "gzip_subbin_edges": list(prior.EDGES),
        "gzip_subbin_quotas": target.tolist(),
        "source_generator": asdict(game),
        "source_reference_sha256": prior.sha(prior.REFERENCE),
    })
    output = directory / "packed.pt"
    temporary = directory / ".packed.pt.tmp"
    PackedDataset(torch.from_numpy(sequences), torch.from_numpy(targets), metadata).save(temporary)
    os.replace(temporary, output)
    atomic_json(directory / "done.json", {
        "schema": "nca-top10-confirmation-arm-v1", "block": root.name, "arm": ARMS[arm],
        "stream_seed": game.seed, "attempts": attempt, "selected": len(states),
        "train_worlds": 19_975, "validation_worlds": 256,
        "train_usable_tokens": 19_715_325, "gzip_bin_counts": observed.tolist(),
        "gzip_ratio_mean": float(values["gzip_ratio"].mean()),
        "patch_copy_excess_mean": float(scores.mean()),
        "cutoffs_by_bin": CUTOFFS.tolist() if arm == 0 else None,
        "packed_sha256": prior.sha(output),
        "sampling_config_sha256": prior.sha(root / "sampling_config.json"),
    })
    print(f"DONE {root.name} {ARMS[arm]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=tuple(STREAMS), required=True)
    args = parser.parse_args()
    stream_seed = STREAMS[args.block]
    root = ROOT / f"block-{args.block}"
    chunks = root / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    target = prior.quotas()
    assert target.shape == (2, 5) and target.sum(axis=1).tolist() == [19_975, 256]
    game = replace(GameConfig(rule_count=BATCH, trajectories_per_rule=1, batch_size=BATCH),
                   seed=stream_seed)
    config = {
        "schema": "nca-top10-confirmation-sampling-v1", "block": args.block,
        "stream_seed": stream_seed, "arm_hash_person": "nca-confirm",
        "split_hash_person": "nca-split", "arms": ARMS,
        "cutoffs_by_gzip_bin": CUTOFFS.tolist(), "gzip_edges": prior.EDGES,
        "quotas_per_arm": target.tolist(), "max_attempts": MAX_ATTEMPTS,
        "reference_sha256": prior.sha(prior.REFERENCE), "game": asdict(game),
    }
    path = root / "sampling_config.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(config))
    else:
        atomic_json(path, config)
    if all((root / arm / "done.json").is_file() for arm in ARMS):
        for arm in ARMS:
            receipt = json.loads((root / arm / "done.json").read_text())
            assert prior.sha(root / arm / "packed.pt") == receipt["packed_sha256"]
        print(f"SKIP complete block {args.block}", flush=True)
        return
    cursor, attempt, counts = replay(chunks, target, stream_seed)
    rollout = _rollout(Game(game), game, game.frames)
    started = time.monotonic()
    while not np.all(counts == target):
        if attempt >= MAX_ATTEMPTS:
            raise RuntimeError(f"quota incomplete at {attempt}: {counts.tolist()}")
        start = attempt
        rows = []
        for _ in range(CHUNK_ATTEMPTS // BATCH):
            if np.all(counts == target):
                break
            cursor, rules = prior.next_keys(cursor)
            for index, rule_seed in enumerate(rules):
                arm = arm_for(rule_seed)
                split = prior.split_for(rule_seed)
                if split > 1 or np.all(counts[arm, split] == target[split]):
                    continue
                key = jnp.asarray(rule_seed, dtype=jnp.uint32)
                initial_seed = jax.random.fold_in(key, 1)
                _, trajectory = rollout(key, initial_seed)
                states = np.asarray(trajectory, np.uint8)
                ratio = gzip_ratio(states, game.num_colors, game.patch_size)
                bin_ = prior.gzip_bin(ratio)
                if bin_ is None or counts[arm, split, bin_] >= target[split, bin_]:
                    continue
                score = prior.horizontal_copy_excess(states)
                if arm == 0 and score < CUTOFFS[bin_]:
                    continue
                rows.append((states.copy(), rule_seed.copy(), np.asarray(initial_seed, np.uint32),
                             ratio, score, arm, split, bin_, attempt + index))
                counts[arm, split, bin_] += 1
            attempt += BATCH
        save_chunk(chunks, start, attempt, cursor, rows)
        if attempt % (CHUNK_ATTEMPTS * 8) == 0 or np.all(counts == target):
            print(f"block={args.block} attempts={attempt}; "
                  f"train={counts[:,0,:].sum(axis=1).tolist()}/19975; "
                  f"validation={counts[:,1,:].sum(axis=1).tolist()}/256; "
                  f"elapsed={time.monotonic()-started:.0f}s", flush=True)
    for arm in range(2):
        pack(root, arm, target, attempt, game)


if __name__ == "__main__":
    main()
