"""Resume-safe 40–50 gzip sampling with matched sub-bin quotas.

Run from repo root: uv run --all-packages python progress/nca_search/sample_low_change.py
"""

from __future__ import annotations

from dataclasses import asdict
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

from nca import GameConfig
from nca.analyze import gzip_ratio
from nca.game import Game
from nca.generate import _rollout
from sampleeff_nca_pretraining.packing import PackedDataset, pack_one
from screen_existing import entropy


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "runs/nca-100k-regimes"
REFERENCE = REPO / "runs/1m-nca-transfer-v2/data/nca/40_50/packed.pt"
if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] not in ("--high", "--spatial-high", "--patch-high", "--patch-low", "--transition-high")):
    raise SystemExit("usage: sample_low_change.py [--high|--spatial-high|--patch-high|--patch-low|--transition-high]")
HIGH = len(sys.argv) == 2 and sys.argv[1] == "--high"
SPATIAL = len(sys.argv) == 2 and sys.argv[1] == "--spatial-high"
TRANSITION = len(sys.argv) == 2 and sys.argv[1] == "--transition-high"
PATCH_LOW = len(sys.argv) == 2 and sys.argv[1] == "--patch-low"
PATCH = len(sys.argv) == 2 and sys.argv[1] in ("--patch-high", "--patch-low")
ROOT = REPO / ("runs/nca-data-search/iteration-06-high-transition-mi" if TRANSITION else
               "runs/nca-data-search/iteration-05-low-patch-copy" if PATCH_LOW else
               "runs/nca-data-search/iteration-04-high-patch-copy" if PATCH else
               "runs/nca-data-search/iteration-03-high-spatial" if SPATIAL else
               "runs/nca-data-search/iteration-02-high-change" if HIGH else
               "runs/nca-data-search/iteration-01-low-change")
CHUNKS = ROOT / "chunks"
THRESHOLD = (0.141948363126 if TRANSITION else
             -0.000374227071 if PATCH else -0.001144367464 if SPATIAL
             else 0.7086004273504274)
TRAIN_WORLDS = 19_975
VAL_WORLDS = 256
GZIP_EDGES = (.40, .42, .44, .46, .48, .50)
CHUNK_ATTEMPTS = 4_096
BATCH = 64
MAX_ATTEMPTS = 1_000_000
GAME = GameConfig(rule_count=BATCH, trajectories_per_rule=1, batch_size=BATCH)


def digest(path: Path) -> str:
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


def split_for(rule_seed: np.ndarray) -> int:
    value = int.from_bytes(hashlib.blake2b(rule_seed.tobytes(), digest_size=8,
                                           person=b"nca-split").digest(), "big") / 2**64
    return 0 if value < .8 else 1 if value < .9 else 2


def spatial_excess_agreement(states: np.ndarray) -> float:
    """Neighbor agreement above the fixed-count random-neighbor expectation."""
    cells = states[..., 0]
    observed = ((cells == np.roll(cells, 1, axis=1)).mean() +
                (cells == np.roll(cells, 1, axis=2)).mean()) / 2
    counts = np.stack([np.bincount(frame.ravel(), minlength=10) for frame in cells])
    expected = np.mean((counts * (counts - 1)).sum(axis=1) / (144 * 143))
    return float(observed - expected)


def horizontal_copy_excess(states: np.ndarray) -> float:
    """Adjacent raster patch copying above the fixed-count random-pair rate."""
    cells = states[..., 0].astype(np.int32)
    patches = cells.reshape(27, 6, 2, 6, 2).transpose(0, 1, 3, 2, 4)
    tokens = patches.reshape(27, 6, 6, 4) @ np.array([1, 10, 100, 1000], dtype=np.int32)
    observed = np.mean(tokens[:, :, 1:] == tokens[:, :, :-1])
    counts = np.bincount(tokens.ravel(), minlength=10_000)
    expected = np.sum(counts * (counts - 1)) / (972 * 971)
    return float(observed - expected)


def transition_mi(states: np.ndarray) -> float:
    """Full-trajectory cell transition MI, in bits, as in the frozen reference screen."""
    cells = states[..., 0].astype(np.int16)
    counts = np.bincount((10 * cells[:-1] + cells[1:]).ravel(), minlength=100).reshape(10, 10)
    total = counts.sum()
    conditional = sum(row.sum() / total * entropy(row) for row in counts if row.sum())
    return entropy(counts.sum(axis=0)) - conditional


def reference_quotas() -> np.ndarray:
    counts = np.zeros((2, 5), dtype=np.int64)
    seen = [0, 0]
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        for row in json.loads(path.read_text())["accepted"]:
            if row["band"] != "40_50":
                continue
            split = {"train": 0, "validation": 1, "test": 2}[row["split"]]
            if split == 2 or seen[split] >= (TRAIN_WORLDS, VAL_WORLDS)[split]:
                continue
            index = gzip_bin(row["gzip_ratio"])
            assert index is not None
            counts[split, index] += 1
            seen[split] += 1
    assert seen == [TRAIN_WORLDS, VAL_WORLDS], seen
    return counts


def next_keys(cursor: jax.Array) -> tuple[jax.Array, np.ndarray]:
    def step(key, _):
        key, rule = jax.random.split(key)
        return key, rule
    following, rules = jax.lax.scan(step, cursor, None, length=BATCH)
    return following, np.asarray(rules, dtype=np.uint32)


def replay_chunks(quotas: np.ndarray) -> tuple[jax.Array, int, np.ndarray]:
    cursor = jax.random.PRNGKey(GAME.seed)
    attempt = 0
    counts = np.zeros_like(quotas)
    for path in sorted(CHUNKS.glob("chunk-*.npz")):
        with np.load(path) as data:
            assert int(data["start"]) == attempt, path
            attempt = int(data["end"])
            cursor = jnp.asarray(data["cursor_end"], dtype=jnp.uint32)
            for split, bin_ in zip(data["split"], data["bin"]):
                counts[int(split), int(bin_)] += 1
            assert np.all(counts <= quotas), path
    return cursor, attempt, counts


def save_chunk(start: int, end: int, cursor: jax.Array, selected: list[tuple]) -> None:
    path = CHUNKS / f"chunk-{start:012d}.npz"
    temporary = path.with_name(f".{path.name}.tmp")
    fields = {
        "start": np.asarray(start, dtype=np.int64),
        "end": np.asarray(end, dtype=np.int64),
        "cursor_end": np.asarray(cursor, dtype=np.uint32),
        "states": np.stack([v[0] for v in selected]) if selected else np.empty((0,27,12,12,1), np.uint8),
        "rule_seed": np.stack([v[1] for v in selected]) if selected else np.empty((0,2), np.uint32),
        "initial_seed": np.stack([v[2] for v in selected]) if selected else np.empty((0,2), np.uint32),
        "gzip_ratio": np.asarray([v[3] for v in selected], dtype=np.float64),
        "change_fraction": np.asarray([v[4] for v in selected], dtype=np.float64),
        "split": np.asarray([v[5] for v in selected], dtype=np.uint8),
        "bin": np.asarray([v[6] for v in selected], dtype=np.uint8),
        "attempt": np.asarray([v[7] for v in selected], dtype=np.int64),
    }
    if SPATIAL:
        fields["spatial_excess_agreement"] = np.asarray([v[8] for v in selected], dtype=np.float64)
    if PATCH:
        fields["horizontal_copy_excess"] = np.asarray([v[8] for v in selected], dtype=np.float64)
    if TRANSITION:
        fields["transition_mi_bits"] = np.asarray([v[8] for v in selected], dtype=np.float64)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **fields)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def pack_selected(quotas: np.ndarray, counts: np.ndarray, attempts: int) -> None:
    assert np.array_equal(counts, quotas)
    chunks = [np.load(path) for path in sorted(CHUNKS.glob("chunk-*.npz"))]
    states = np.concatenate([data["states"] for data in chunks])
    split = np.concatenate([data["split"] for data in chunks])
    ratios = np.concatenate([data["gzip_ratio"] for data in chunks])
    changes = np.concatenate([data["change_fraction"] for data in chunks])
    spatial_scores = (np.concatenate([data["spatial_excess_agreement"] for data in chunks])
                      if SPATIAL else None)
    patch_scores = (np.concatenate([data["horizontal_copy_excess"] for data in chunks])
                    if PATCH else None)
    transition_scores = (np.concatenate([data["transition_mi_bits"] for data in chunks])
                         if TRANSITION else None)
    rules = np.concatenate([data["rule_seed"] for data in chunks])
    assert len(states) == TRAIN_WORLDS + VAL_WORLDS
    assert len({tuple(v) for v in rules}) == len(rules)
    for data in chunks:
        data.close()
    # Vectorized equivalent of the original 2×2 base-10 patch tokenization.
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
    for index in (0, len(states)-1):
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
        "selection": ("40–50 gzip; cell-transition MI >= reference median; reference gzip sub-bin quotas"
                      if TRANSITION else
                      "40–50 gzip; horizontal patch-copy excess <= reference median; reference gzip sub-bin quotas"
                      if PATCH_LOW else
                      "40–50 gzip; horizontal patch-copy excess >= reference median; reference gzip sub-bin quotas"
                      if PATCH else
                      "40–50 gzip; spatial excess agreement >= reference median; reference gzip sub-bin quotas"
                      if SPATIAL else
                      "40–50 gzip; change fraction >= reference median; reference gzip sub-bin quotas"
                      if HIGH else "40–50 gzip; change fraction <= reference median; reference gzip sub-bin quotas"),
        "transition_mi_threshold_bits" if TRANSITION else "patch_copy_threshold" if PATCH else "spatial_threshold" if SPATIAL else "change_threshold": THRESHOLD,
        "gzip_subbin_edges": list(GZIP_EDGES),
        "gzip_subbin_quotas": quotas.tolist(),
        "source_generator": asdict(GAME),
        "source_reference_sha256": digest(REFERENCE),
    })
    output = ROOT / "packed.pt"
    temporary = ROOT / ".packed.pt.tmp"
    PackedDataset(torch.from_numpy(sequences), torch.from_numpy(targets), metadata).save(temporary)
    os.replace(temporary, output)
    done = {
        "schema": ("nca-high-transition-mi-search-v1" if TRANSITION else
                   "nca-low-patch-copy-search-v1" if PATCH_LOW else
                   "nca-high-patch-copy-search-v1" if PATCH else
                   "nca-high-spatial-search-v1" if SPATIAL else
                   "nca-high-change-search-v1" if HIGH else "nca-low-change-search-v1"),
        "packed_sha256": digest(output),
        "reference_packed_sha256": digest(REFERENCE), "attempts": attempts,
        "selected": len(states), "train_worlds": TRAIN_WORLDS, "validation_worlds": VAL_WORLDS,
        "train_usable_tokens": TRAIN_WORLDS * 987,
        "gzip_bin_counts": counts.tolist(), "gzip_ratio_mean": float(ratios.mean()),
        "change_fraction_mean": float(changes.mean()),
        "change_fraction_min": float(changes.min()), "change_fraction_max": float(changes.max()),
        "transition_mi_threshold_bits" if TRANSITION else "patch_copy_threshold" if PATCH else "spatial_threshold" if SPATIAL else "change_threshold": THRESHOLD,
    }
    if SPATIAL:
        done.update(spatial_excess_agreement_mean=float(spatial_scores.mean()),
                    spatial_excess_agreement_min=float(spatial_scores.min()),
                    spatial_excess_agreement_max=float(spatial_scores.max()))
    if PATCH:
        done.update(horizontal_copy_excess_mean=float(patch_scores.mean()),
                    horizontal_copy_excess_min=float(patch_scores.min()),
                    horizontal_copy_excess_max=float(patch_scores.max()))
    if TRANSITION:
        done.update(transition_mi_bits_mean=float(transition_scores.mean()),
                    transition_mi_bits_min=float(transition_scores.min()),
                    transition_mi_bits_max=float(transition_scores.max()))
    write_json(ROOT / "done.json", done)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    CHUNKS.mkdir(exist_ok=True)
    quotas = reference_quotas()
    config = {"schema": ("nca-high-transition-mi-search-v1" if TRANSITION else
                         "nca-low-patch-copy-search-v1" if PATCH_LOW else
                         "nca-high-patch-copy-search-v1" if PATCH else
                         "nca-high-spatial-search-v1" if SPATIAL else
                         "nca-high-change-search-v1" if HIGH else "nca-low-change-search-v1"),
              "threshold": THRESHOLD,
              "gzip_edges": GZIP_EDGES, "quotas": quotas.tolist(),
              "reference_sha256": digest(REFERENCE), "game": asdict(GAME)}
    config_path = ROOT / "sampling_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == json.loads(json.dumps(config))
    else:
        write_json(config_path, config)
    done_path = ROOT / "done.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        assert digest(ROOT / "packed.pt") == done["packed_sha256"]
        print("SKIP complete sampling", flush=True)
        return
    cursor, attempt, counts = replay_chunks(quotas)
    rollout = _rollout(Game(GAME), GAME, GAME.frames)
    started = time.monotonic()
    while not np.array_equal(counts, quotas):
        if attempt >= MAX_ATTEMPTS:
            raise RuntimeError(f"quota not met after {attempt} attempts: {counts.tolist()}")
        start = attempt
        selected = []
        for _ in range(CHUNK_ATTEMPTS // BATCH):
            if np.array_equal(counts, quotas):
                break
            cursor, rules = next_keys(cursor)
            for index, rule_seed in enumerate(rules):
                key = jnp.asarray(rule_seed, dtype=jnp.uint32)
                initial_seed = jax.random.fold_in(key, 1)
                _, trajectory = rollout(key, initial_seed)
                states = np.asarray(trajectory, dtype=np.uint8)
                ratio = gzip_ratio(states, GAME.num_colors, GAME.patch_size)
                bin_ = gzip_bin(ratio)
                if bin_ is None:
                    continue
                change = float(np.mean(states[1:] != states[:-1]))
                score = (transition_mi(states) if TRANSITION else
                         horizontal_copy_excess(states) if PATCH else
                         spatial_excess_agreement(states) if SPATIAL else None)
                if (score > THRESHOLD if PATCH_LOW else
                    score < THRESHOLD if PATCH or SPATIAL or TRANSITION else
                    change < THRESHOLD if HIGH else change > THRESHOLD):
                    continue
                split = split_for(rules[index])
                if split > 1 or counts[split, bin_] >= quotas[split, bin_]:
                    continue
                selected.append((states.copy(), rule_seed.copy(), np.asarray(initial_seed, dtype=np.uint32),
                                 ratio, change, split, bin_, attempt+index, score))
                counts[split, bin_] += 1
            attempt += BATCH
        save_chunk(start, attempt, cursor, selected)
        print(f"sampled {attempt} attempts; train {counts[0].sum()}/{TRAIN_WORLDS}; "
              f"validation {counts[1].sum()}/{VAL_WORLDS}; elapsed {time.monotonic()-started:.0f}s",
              flush=True)
    pack_selected(quotas, counts, attempt)
    print("DONE sampling", ROOT / "packed.pt", flush=True)


if __name__ == "__main__":
    main()
