"""Resumable, rule-disjoint rejection sampling into raw v1 NCA datasets.

Run from the repository root:
  uv run --locked --package sampleeff-nca python examples/generate_100k_regimes.py
Each completed batch is a set of immutable v1 datasets plus an atomic receipt.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from nca import GameConfig, TrajectoryDataset
from nca.analyze import gzip_ratio
from nca.generate import TrajectoryBatch, generate
from nca.io import save_batches


BANDS = ("20_30", "30_40", "40_50", "50_plus")
CONFIG = GameConfig(rule_count=100_000, trajectories_per_rule=1, batch_size=64)
DEFAULT_ROOT = Path("runs/nca-100k-regimes")


def band_for(ratio: float) -> str | None:
    if not math.isfinite(ratio):
        raise ValueError(f"non-finite gzip ratio: {ratio}")
    if ratio < 0.2:
        return None
    if ratio < 0.3:
        return "20_30"
    if ratio < 0.4:
        return "30_40"
    if ratio < 0.5:
        return "40_50"
    return "50_plus"


def split_for(rule_seed: np.ndarray) -> str:
    value = int.from_bytes(hashlib.blake2b(rule_seed.tobytes(), digest_size=8,
                                           person=b"nca-split").digest(), "big")
    fraction = value / 2**64
    return "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def config_hash(config: GameConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def empty_state(config: GameConfig, target: int) -> dict:
    return {
        "schema": "nca-regimes-run-v1",
        "config": asdict(config),
        "config_sha256": config_hash(config),
        "target_per_band": target,
        "band_intervals": {"20_30": "[0.20,0.30)", "30_40": "[0.30,0.40)",
                           "40_50": "[0.40,0.50)", "50_plus": "[0.50,infinity)"},
        "split_method": "blake2b(rule seed) thresholds 0.8/0.9",
        "next_attempt": 0,
        "attempted_rules": 0,
        "attempted_worlds": 0,
        "accepted": {band: 0 for band in BANDS},
        "eligible": {band: 0 for band in BANDS},
        "rejected": 0,
        "duplicate_pairs": 0,
        "duplicate_trajectory_hashes": 0,
        "gzip_histogram_0_1_steps": [0] * 10 + [0],
        "splits": {band: {split: 0 for split in ("train", "validation", "test")}
                   for band in BANDS},
        "chunks": 0,
        "storage_bytes": 0,
        "elapsed_seconds": 0.0,
        "throughput_worlds_per_second": 0.0,
        "current_throughput_worlds_per_second": 0.0,
        "updated_at_utc": None,
        "process_pid": None,
    }


def add_receipt(state: dict, receipt: dict, root: Path,
                pairs: set, hashes: set) -> None:
    start, end = receipt["attempt_start"], receipt["attempt_end"]
    if start != state["next_attempt"] or end <= start:
        raise ValueError(f"noncontiguous receipt {start}:{end}")
    state["next_attempt"] = end
    state["attempted_rules"] += end - start
    state["attempted_worlds"] += end - start
    for index, value in enumerate(receipt["gzip_histogram_0_1_steps"]):
        state["gzip_histogram_0_1_steps"][index] += value
    for band in BANDS:
        state["eligible"][band] += receipt["eligible"][band]
    state["duplicate_pairs"] += receipt["duplicate_pairs"]
    for row in receipt["accepted"]:
        band = row["band"]
        trajectory_id = row["trajectory_id"]
        if trajectory_id != f"trajectory-{row['attempt']:012d}" or not start <= row["attempt"] < end:
            raise ValueError(f"invalid trajectory ID in receipt {start}")
        pair = (tuple(row["rule_seed"]), tuple(row["initial_seed"]))
        if pair in pairs:
            raise ValueError(f"duplicate rule/initial seed pair: {trajectory_id}")
        pairs.add(pair)
        if row["states_sha256"] in hashes:
            state["duplicate_trajectory_hashes"] += 1
        hashes.add(row["states_sha256"])
        state["accepted"][band] += 1
        if state["accepted"][band] > state["target_per_band"]:
            raise ValueError(f"target exceeded for {band}")
        state["splits"][band][row["split"]] += 1
        dataset = root / row["dataset"]
        if not (dataset / "manifest.json").is_file():
            raise ValueError(f"missing complete v1 dataset: {dataset}")
    state["rejected"] = state["attempted_worlds"] - sum(state["accepted"].values())
    state["chunks"] += len(receipt["datasets"])
    state["storage_bytes"] += receipt["storage_bytes"]


def load_state(root: Path, config: GameConfig, target: int):
    manifest_path = root / "manifest.json"
    saved_elapsed = 0.0
    if root.exists():
        if not manifest_path.is_file():
            raise ValueError(f"existing run has no manifest: {root}")
        saved = json.loads(manifest_path.read_text())
        if (saved.get("schema") != "nca-regimes-run-v1" or
                saved.get("config") != asdict(config) or
                saved.get("target_per_band") != target):
            raise ValueError("existing run configuration or target differs; refusing to mix datasets")
        saved_elapsed = saved.get("elapsed_seconds", 0.0)
    else:
        root.mkdir(parents=True)
        (root / "receipts").mkdir()
        for band in BANDS:
            (root / band).mkdir()
        write_json(manifest_path, empty_state(config, target))
    state = empty_state(config, target)
    pairs, hashes = set(), set()
    for path in sorted((root / "receipts").glob("batch-*.json")):
        receipt = json.loads(path.read_text())
        if receipt["config_sha256"] != state["config_sha256"]:
            raise ValueError(f"receipt configuration differs: {path}")
        add_receipt(state, receipt, root, pairs, hashes)
    for band in BANDS:
        for path in (root / band).glob(".batch-*.tmp-*"):
            if path.is_dir():
                shutil.rmtree(path)
    state["elapsed_seconds"] = saved_elapsed
    return state, pairs, hashes


def next_rule_keys(cursor: jax.Array, batch_size: int):
    def step(key, _):
        following, rule = jax.random.split(key)
        return following, rule
    return jax.lax.scan(step, cursor, None, length=batch_size)


def chunk_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def make_receipt(root: Path, config: GameConfig, state: dict,
                 rules: np.ndarray, batch: TrajectoryBatch, pairs: set) -> dict:
    start = state["next_attempt"]
    chosen = {band: [] for band in BANDS}
    eligible = {band: 0 for band in BANDS}
    histogram = [0] * 11
    rows = []
    new_pairs = set()
    duplicate_pairs = 0
    for index, states in enumerate(batch.states):
        ratio = gzip_ratio(states, config.num_colors, config.patch_size)
        if not math.isfinite(ratio):
            raise ValueError(f"non-finite gzip ratio at attempt {start + index}")
        histogram[min(int(ratio * 10), 10)] += 1
        band = band_for(ratio)
        if band is None:
            continue
        eligible[band] += 1
        pair = (tuple(rules[index].tolist()), tuple(batch.initial_seeds[index].tolist()))
        if pair in pairs or pair in new_pairs:
            duplicate_pairs += 1
            continue
        if state["accepted"][band] + len(chosen[band]) >= state["target_per_band"]:
            continue
        new_pairs.add(pair)
        chosen[band].append(index)
        rule_seed = rules[index]
        initial_seed = batch.initial_seeds[index]
        rows.append({
            "trajectory_id": f"trajectory-{start + index:012d}",
            "attempt": start + index,
            "rule_id": len(chosen[band]) - 1,
            "rule_seed": rule_seed.tolist(),
            "initial_seed": initial_seed.tolist(),
            "generator_seed": config.seed,
            "gzip_ratio": ratio,
            "band": band,
            "split": split_for(rule_seed),
            "dataset": f"{band}/batch-{start:012d}",
            "row": len(chosen[band]) - 1,
            "states_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
            "config_sha256": state["config_sha256"],
        })
    datasets = {}
    storage_bytes = 0
    for band, indices in chosen.items():
        if not indices:
            continue
        destination = root / band / f"batch-{start:012d}"
        selected = np.asarray(indices, dtype=np.intp)
        if destination.exists():
            existing = TrajectoryDataset.load(destination)
            if len(existing) != len(indices):
                raise ValueError(f"orphaned dataset count mismatch: {destination}")
            for local, index in enumerate(indices):
                record = existing[local]
                if (not np.array_equal(record["states"], batch.states[index]) or
                        not np.array_equal(record["rule_seed"], rules[index]) or
                        not np.array_equal(record["initial_seed"], batch.initial_seeds[index])):
                    raise ValueError(f"orphaned dataset content mismatch: {destination}")
        else:
            temporary = Path(tempfile.mkdtemp(prefix=f".batch-{start:012d}.tmp-",
                                              dir=root / band))
            selected_batch = TrajectoryBatch(
                states=batch.states[selected],
                rule_ids=np.arange(len(indices), dtype=np.int32),
                initial_seeds=batch.initial_seeds[selected],
                initial_states=batch.initial_states[selected],
            )
            save_batches(replace(config, rule_count=len(indices)), temporary / "raw",
                         rules[selected], (selected_batch,))
            os.replace(temporary / "raw", destination)
            temporary.rmdir()
        datasets[band] = str(destination.relative_to(root))
        storage_bytes += chunk_bytes(destination)
    return {
        "attempt_start": start,
        "attempt_end": start + len(batch.states),
        "config_sha256": state["config_sha256"],
        "eligible": eligible,
        "duplicate_pairs": duplicate_pairs,
        "gzip_histogram_0_1_steps": histogram,
        "accepted": rows,
        "datasets": datasets,
        "storage_bytes": storage_bytes,
    }


def claim_lock(root: Path) -> Path:
    path = root / "generation.lock"
    if path.exists():
        try:
            pid = int(path.read_text().strip())
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink()
        except (ValueError, PermissionError) as error:
            raise RuntimeError(f"cannot inspect existing generation lock {path}") from error
        else:
            raise RuntimeError(f"generation is already running as PID {pid}")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(f"{os.getpid()}\n")
    return path


def run(root: Path, config: GameConfig, target: int, max_batches: int | None):
    if config.trajectories_per_rule != 1 or config.filter_mode != "none":
        raise ValueError("this run requires one world per unfiltered rule")
    state, pairs, hashes = load_state(root, config, target)
    lock = claim_lock(root)
    started = time.monotonic()
    try:
        cursor = jax.random.PRNGKey(config.seed)
        # Advance the exact sequential sampler stream to the committed attempt.
        # Resume is O(attempts / batch_size), with no need to trust an uncommitted cursor.
        if state["next_attempt"]:
            cursor, _ = next_rule_keys(cursor, state["next_attempt"])
        print(f"START {datetime.now(timezone.utc).isoformat()} pid={os.getpid()} root={root} "
              f"resuming_at={state['next_attempt']} config={json.dumps(asdict(config), sort_keys=True)}",
              flush=True)
        batches = 0
        stream = None
        while any(state["accepted"][band] < target for band in BANDS):
            if max_batches is not None and batches >= max_batches:
                break
            if stream is None:
                window_size = config.batch_size * 128
                cursor, keys = next_rule_keys(cursor, window_size)
                window_rules = np.asarray(keys, dtype=np.uint32)
                stream = iter(generate(replace(config, rule_count=window_size), window_rules))
                window_offset = 0
            batch = next(stream)
            rules = window_rules[window_offset:window_offset + len(batch.states)]
            window_offset += len(batch.states)
            if window_offset == len(window_rules):
                stream = None
            receipt = make_receipt(root, config, state, rules, batch, pairs)
            receipt_path = root / "receipts" / f"batch-{state['next_attempt']:012d}.json"
            write_json(receipt_path, receipt)
            add_receipt(state, receipt, root, pairs, hashes)
            batches += 1
            batch_seconds = time.monotonic() - started
            elapsed = state["elapsed_seconds"] + batch_seconds
            state["elapsed_seconds"] = elapsed
            started = time.monotonic()
            state["throughput_worlds_per_second"] = state["attempted_worlds"] / max(elapsed, 1e-9)
            state["current_throughput_worlds_per_second"] = len(batch.states) / max(batch_seconds, 1e-9)
            state["process_pid"] = os.getpid()
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(root / "manifest.json", state)
            rates = {band: round(state["eligible"][band] / state["attempted_worlds"], 5)
                     for band in BANDS}
            print(f"PROGRESS attempts={state['attempted_worlds']} accepted={state['accepted']} "
                  f"eligible_rates={rates} worlds_per_second={state['throughput_worlds_per_second']:.2f} "
                  f"current_worlds_per_second={state['current_throughput_worlds_per_second']:.2f} "
                  f"storage_bytes={state['storage_bytes']}", flush=True)
        print(f"STOP complete={all(value == target for value in state['accepted'].values())} "
              f"attempts={state['attempted_worlds']} accepted={state['accepted']}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target-per-band", type=int, default=25_000)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="bounded preflight only; leave unset for the full run")
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size,
                        help="preflight only; fixed at 64 for the full run")
    options = parser.parse_args()
    if options.target_per_band < 1 or options.batch_size < 1:
        parser.error("target and batch size must be positive")
    run(options.output, replace(CONFIG, batch_size=options.batch_size),
        options.target_per_band, options.max_batches)


if __name__ == "__main__":
    main()
