"""One-seed NCA-to-TinyStories transfer queue; run with uv --all-packages.

`--setup` prepares immutable inputs and benchmarks the real update loop.
Without flags, completed stages are skipped and interrupted training resumes
from its latest complete checkpoint. Output stays under runs/1m-nca-transfer-v1.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

from datasets import load_dataset
import jax
import numpy as np
import tiktoken
import torch
from torch.nn import functional as F

from nca import TrajectoryDataset
from sampleeff_nca_pretraining.model import ModelConfig, create_model, trainable_parameters
from sampleeff_nca_pretraining.packing import PackedDataset, pack_one
from sampleeff_nca_pretraining.training import (TrainingConfig, load_checkpoint,
                                                load_pretrained_body, prepare,
                                                save_checkpoint)
from sampleeff_shared.models import create_attention_mask


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "runs/nca-100k-regimes"
ROOT = REPO / "runs/1m-nca-transfer-v1"
BANDS = ("20_30", "30_40", "40_50", "50_plus")
TINY_DATASET = "roneneldan/TinyStories"
TINY_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENIZER = "gpt2"
LANGUAGE_TOKENS = 5_000_000
VALIDATION_TOKENS = 262_144
CONTEXT = 256
INITIALIZATION_SEED = 11
ORDER_SEED = 17
TRAINING_SEED = 23
EVAL_NCA_WORLDS = 256
NCA_PHYSICAL_BATCH = 8
LANGUAGE_PHYSICAL_BATCH = 4
EFFECTIVE_BATCH = 8
LR = 3e-4
WARMUP_STEPS = 100
CHECKPOINT_STEPS = 500
NCA_MILESTONES = tuple(range(0, 20_000_001, 2_000_000))
LANGUAGE_MILESTONES = (0, 250_000, 500_000, 1_000_000,
                       2_000_000, 3_000_000, 5_000_000)
NCA_MODEL = ModelConfig(vocab_size=10_002, context_length=CONTEXT,
                        layers=4, heads=2, embedding_dim=20, dropout=0.0,
                        tie_embeddings=True)
TEXT_MODEL = ModelConfig(vocab_size=50_257, context_length=CONTEXT,
                         layers=4, heads=2, embedding_dim=20, dropout=0.0,
                         tie_embeddings=True)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hash_.update(block)
    return hash_.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def configuration() -> dict:
    source = json.loads((SOURCE / "manifest.json").read_text())
    if (source["accepted"] != {band: 25_000 for band in BANDS} or
            source["duplicate_pairs"] != 0 or source["config"]["patch_size"] != 2 or
            source["config"]["num_colors"] != 10 or
            source["config"]["frames"] != 27 or source["config"]["grid_size"] != 12):
        raise ValueError("source NCA corpus is incomplete or incompatible")
    train_counts = {band: source["splits"][band]["train"] for band in BANDS}
    train_worlds = min(train_counts.values())
    packed_per_world = 27 * ((12 // 2) ** 2 + 2)
    if packed_per_world != 1026:
        raise ValueError("unexpected packed trajectory length")
    # Four 256-token windows use targets 1..1024. The first 37 targets are
    # masked, and the final packed token is outside those four windows.
    usable_per_world = CONTEXT * 4 - (38 - 1)
    if usable_per_world != 987:
        raise AssertionError("NCA token accounting changed")
    config = {
        "name": "1m-nca-transfer-v1", "one_seed_exploratory": True,
        "source_files_sha256": {
            str(path.relative_to(REPO)): digest(path)
            for path in (
                Path(__file__).resolve(), REPO / "uv.lock",
                REPO / "packages/shared/src/sampleeff_shared/models.py",
                REPO / "packages/pre-pre-training/src/sampleeff_nca_pretraining/model.py",
                REPO / "packages/pre-pre-training/src/sampleeff_nca_pretraining/packing.py",
                REPO / "packages/pre-pre-training/src/sampleeff_nca_pretraining/training.py",
            )
        },
        "nca_source": str(SOURCE), "nca_source_config_sha256": source["config_sha256"],
        "nca_manifest_sha256": digest(SOURCE / "manifest.json"),
        "source_train_worlds_by_band": train_counts,
        "nca_train_worlds_per_band": train_worlds,
        "nca_all_packed_tokens_per_band": packed_per_world * 25_000,
        "nca_usable_training_tokens_per_world": usable_per_world,
        "nca_usable_training_tokens_per_band": train_worlds * usable_per_world,
        "nca_validation_worlds_per_band": EVAL_NCA_WORLDS,
        "language_tokens": LANGUAGE_TOKENS,
        "language_validation_tokens": VALIDATION_TOKENS,
        "tiny_dataset": TINY_DATASET, "tiny_revision": TINY_REVISION,
        "text_tokenizer": TOKENIZER, "text_eot_token": 50_256,
        "nca_model": asdict(NCA_MODEL), "text_model": asdict(TEXT_MODEL),
        "initialization_seed": INITIALIZATION_SEED,
        "order_seed": ORDER_SEED, "training_seed": TRAINING_SEED,
        "nca_physical_batch": NCA_PHYSICAL_BATCH,
        "language_physical_batch": LANGUAGE_PHYSICAL_BATCH,
        "effective_batch": EFFECTIVE_BATCH,
        "optimizer": "Adam", "learning_rate": LR, "weight_decay": 0.0,
        "scheduler": "cosine with linear warmup", "warmup_steps": WARMUP_STEPS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "nca_milestones": list(NCA_MILESTONES),
        "language_milestones": list(LANGUAGE_MILESTONES),
        "backend": "mps" if torch.backends.mps.is_available() else "cpu",
    }
    return config


def init_root(config: dict) -> None:
    path = ROOT / "config/experiment.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("experiment configuration differs; refusing to mix runs")
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(path, config)
    if config["backend"] != "mps":
        raise RuntimeError("this benchmarked run requires the available MPS backend")


def stage_done(directory: Path, config: dict, outputs: tuple[Path, ...]) -> bool:
    marker = directory / "done.json"
    if not marker.exists():
        return False
    value = json.loads(marker.read_text())
    if value.get("config_sha256") != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest():
        raise ValueError(f"completed stage has a different configuration: {directory}")
    if any(not output.is_file() for output in outputs):
        raise ValueError(f"completed stage is missing output: {directory}")
    return True


def mark_done(directory: Path, config: dict, **details) -> None:
    atomic_json(directory / "done.json", {
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "completed_at_utc": utc(), **details,
    })


def prepare_tiny(config: dict) -> None:
    started = time.monotonic()
    directory = ROOT / "data/tinystories"
    train_path, val_path = directory / "train.bin", directory / "val.bin"
    metadata_path = directory / "metadata.json"
    if stage_done(directory, config, (train_path, val_path, metadata_path)):
        metadata = json.loads(metadata_path.read_text())
        if (metadata["train_sha256"] != digest(train_path) or
                metadata["validation_sha256"] != digest(val_path)):
            raise ValueError("TinyStories token files changed")
        print("SKIP TinyStories data preparation", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    encoder = tiktoken.get_encoding(TOKENIZER)
    details = {}
    for split, output, target in (("train", train_path, LANGUAGE_TOKENS + 1),
                                  ("validation", val_path, VALIDATION_TOKENS + 1)):
        temporary = output.with_suffix(".bin.tmp")
        temporary.unlink(missing_ok=True)
        written = rows_seen = 0
        rows = load_dataset(TINY_DATASET, split=split, streaming=True,
                            revision=TINY_REVISION)
        with temporary.open("wb") as stream:
            for row in rows:
                tokens = encoder.encode_ordinary(row["text"])
                tokens.append(encoder.eot_token)
                take = min(len(tokens), target - written)
                np.asarray(tokens[:take], dtype=np.uint16).tofile(stream)
                written += take
                rows_seen += 1
                if written == target:
                    break
            stream.flush()
            os.fsync(stream.fileno())
        if written != target:
            raise ValueError(f"TinyStories {split} exhausted before {target} tokens")
        os.replace(temporary, output)
        details[split] = {"rows_seen": rows_seen, "tokens_saved": written,
                          "sha256": digest(output)}
        print(f"DATA {split} rows={rows_seen} tokens={written}", flush=True)
    atomic_json(metadata_path, {
        "source": f"https://huggingface.co/datasets/{TINY_DATASET}",
        "revision": TINY_REVISION, "tokenizer": TOKENIZER,
        "train_sha256": details["train"]["sha256"],
        "validation_sha256": details["validation"]["sha256"],
        "splits": details,
    })
    mark_done(directory, config, train_sha256=details["train"]["sha256"],
              validation_sha256=details["validation"]["sha256"],
              runtime_seconds=time.monotonic()-started)


def receipts_for(band: str):
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        receipt = json.loads(path.read_text())
        rows = [row for row in receipt["accepted"] if row["band"] == band]
        if rows:
            yield rows


def pack_band(band: str, config: dict) -> None:
    directory = ROOT / "data/nca" / band
    packed_path = directory / "packed.pt"
    if stage_done(directory, config, (packed_path,)):
        dataset = PackedDataset.load(packed_path)
        if len(dataset) != 25_000 or dataset.sequences.shape[1] != 1026:
            raise ValueError(f"packed band is incompatible: {band}")
        print(f"SKIP packed {band}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    sequences = torch.empty((25_000, 1026), dtype=torch.int32)
    targets = torch.empty_like(sequences)
    split_indices = {split: [] for split in ("train", "validation", "test")}
    seen_seeds = {split: set() for split in split_indices}
    index = 0
    started = time.monotonic()
    for rows in receipts_for(band):
        dataset_path = SOURCE / rows[0]["dataset"]
        raw = list(TrajectoryDataset.load(dataset_path))
        for row in rows:
            record = raw[row["row"]]
            if (record["rule_seed"].tolist() != row["rule_seed"] or
                    record["initial_seed"].tolist() != row["initial_seed"] or
                    hashlib.sha256(record["states"].tobytes()).hexdigest() != row["states_sha256"]):
                raise ValueError(f"source trajectory mismatch: {row['trajectory_id']}")
            seq, target = pack_one(record["states"], num_colors=10, patch_size=2)
            sequences[index] = seq
            targets[index] = target
            split_indices[row["split"]].append(index)
            seed = tuple(row["rule_seed"])
            if any(seed in existing for existing in seen_seeds.values()):
                raise ValueError(f"duplicate rule across splits in {band}: {seed}")
            seen_seeds[row["split"]].add(seed)
            index += 1
        if index and index % 1000 < len(rows):
            print(f"PACK {band} {index}/25000 elapsed={time.monotonic()-started:.1f}s", flush=True)
    if index != 25_000:
        raise ValueError(f"packed {index} worlds for {band}; expected 25000")
    if len(split_indices["train"]) < config["nca_train_worlds_per_band"]:
        raise ValueError(f"insufficient training worlds for {band}")
    metadata = {
        "source_directory": str(SOURCE),
        "source_manifest_sha256": config["nca_manifest_sha256"],
        "grid_len": 38, "num_colors": 10, "patch_size": 2,
        "packing_config": {"num_colors": 10, "patch_size": 2},
        "split_indices": split_indices,
        "packed_tokens": int(sequences.numel()),
        "usable_training_tokens": len(split_indices["train"]) * 987,
        "selection": "first N training worlds in receipt/attempt order",
    }
    temporary = packed_path.with_suffix(".pt.tmp")
    temporary.unlink(missing_ok=True)
    PackedDataset(sequences, targets, metadata).save(temporary)
    os.replace(temporary, packed_path)
    mark_done(directory, config, packed_sha256=digest(packed_path),
              split_counts={key: len(value) for key, value in split_indices.items()},
              elapsed_seconds=time.monotonic()-started)
    print(f"PACK DONE {band} seconds={time.monotonic()-started:.1f}", flush=True)


class NCAWindows:
    def __init__(self, packed: PackedDataset, split: str, worlds: int):
        self.sequences = packed.sequences
        self.indices = np.asarray(packed.metadata["split_indices"][split][:worlds], dtype=np.int64)
        self.size = len(self.indices) * 4
        self.valid = np.tile(np.asarray([219, 256, 256, 256], dtype=np.int64), len(self.indices))

    def batch(self, positions: np.ndarray):
        worlds = self.indices[positions // 4]
        offsets = positions % 4 * CONTEXT
        x = torch.stack([self.sequences[int(world), int(start):int(start)+CONTEXT]
                         for world, start in zip(worlds, offsets)]).long()
        y = torch.stack([self.sequences[int(world), int(start)+1:int(start)+CONTEXT+1]
                         for world, start in zip(worlds, offsets)]).long()
        for row, start in enumerate(offsets):
            if start == 0:
                y[row, :37] = -100
        return x, y


class TextWindows:
    def __init__(self, path: Path, token_count: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.tokens) != token_count + 1:
            raise ValueError(f"text token file has wrong length: {path}")
        self.size = math.ceil(token_count / CONTEXT)
        self.valid = np.full(self.size, CONTEXT, dtype=np.int64)
        self.valid[-1] = token_count - (self.size-1)*CONTEXT

    def batch(self, positions: np.ndarray):
        x = torch.full((len(positions), CONTEXT), 50_256, dtype=torch.long)
        y = torch.full((len(positions), CONTEXT), -100, dtype=torch.long)
        for row, index in enumerate(positions):
            start = int(index) * CONTEXT
            count = int(self.valid[index])
            x[row, :count] = torch.from_numpy(self.tokens[start:start+count].astype(np.int64))
            y[row, :count] = torch.from_numpy(self.tokens[start+1:start+count+1].astype(np.int64))
        return x, y


def ordered(dataset, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(dataset.size)
    prefix = np.concatenate(([0], np.cumsum(dataset.valid[order], dtype=np.int64)))
    return order, prefix


def evaluate_model(model, dataset, batch_size: int, mask, limit: int | None = None) -> float:
    model.eval()
    total = 0.0
    count = 0
    size = dataset.size if limit is None else min(dataset.size, limit)
    with torch.no_grad():
        for start in range(0, size, batch_size):
            positions = np.arange(start, min(start+batch_size, size))
            x, y = dataset.batch(positions)
            x, y = x.to("mps"), y.to("mps")
            logits = model(x, attention_mask=mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), ignore_index=-100, reduction="sum")
            total += float(loss.detach().cpu())
            count += int((y != -100).sum().item())
    return total / count


def update_step(state, dataset, positions: np.ndarray, microbatch: int, mask) -> float:
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    count = int(dataset.valid[positions].sum())
    total = 0.0
    for start in range(0, len(positions), microbatch):
        x, y = dataset.batch(positions[start:start+microbatch])
        x, y = x.to(state.device), y.to(state.device)
        logits = state.model(x, attention_mask=mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1), ignore_index=-100, reduction="sum") / count
        if not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        total += float(loss.detach().cpu())
    state.optimizer.step()
    state.scheduler.step()
    torch.mps.synchronize()
    return total


def benchmark(config: dict) -> dict:
    started = time.monotonic()
    directory = ROOT / "config"
    output = directory / "benchmark.json"
    if stage_done(directory / "benchmark", config, (output,)):
        print("SKIP benchmark", flush=True)
        return json.loads(output.read_text())
    result = {"device": "mps", "context_length": CONTEXT,
              "effective_batch": EFFECTIVE_BATCH,
              "mps_recommended_max_memory_bytes": torch.mps.recommended_max_memory(),
              "mps_allocated_before_bytes": torch.mps.current_allocated_memory(),
              "measured_at_utc": utc()}
    for name, model_config, physical in (("nca", NCA_MODEL, NCA_PHYSICAL_BATCH),
                                         ("language", TEXT_MODEL, LANGUAGE_PHYSICAL_BATCH)):
        state = prepare(create_model(model_config, INITIALIZATION_SEED), model_config,
                        TrainingConfig(steps=100, batch_size=EFFECTIVE_BATCH,
                                       learning_rate=LR, warmup_steps=WARMUP_STEPS,
                                       device="mps", seed=TRAINING_SEED))
        mask = create_attention_mask(CONTEXT, additive=True).to("mps")
        vocab = model_config.vocab_size
        x = torch.randint(0, vocab, (EFFECTIVE_BATCH, CONTEXT), dtype=torch.long)
        y = x.clone()
        class FixedBatch:
            valid = np.full(EFFECTIVE_BATCH, CONTEXT, dtype=np.int64)
            def batch(self, indices):
                return x[indices], y[indices]
        fixed = FixedBatch()
        times = []
        for _ in range(12):
            started = time.monotonic()
            update_step(state, fixed, np.arange(EFFECTIVE_BATCH), physical, mask)
            times.append(time.monotonic()-started)
        stable = times[3:]
        rate = EFFECTIVE_BATCH * CONTEXT / (sum(stable) / len(stable))
        result[name] = {
            "parameters": trainable_parameters(state.model),
            "physical_batch": physical, "effective_batch": EFFECTIVE_BATCH,
            "context_length": CONTEXT,
            "tokens_per_second": rate,
            "steps_per_second": len(stable)/sum(stable),
            "mps_allocated_bytes": torch.mps.current_allocated_memory(),
        }
        print(f"BENCHMARK {name} {rate:.0f} tokens/s", flush=True)
        del state, mask
        torch.mps.empty_cache()
    nca_total = config["nca_usable_training_tokens_per_band"] * 4
    language_total = LANGUAGE_TOKENS * 5
    result["estimated_seconds"] = {
        "one_nca": config["nca_usable_training_tokens_per_band"] / result["nca"]["tokens_per_second"],
        "all_nca": nca_total / result["nca"]["tokens_per_second"],
        "one_language": LANGUAGE_TOKENS / result["language"]["tokens_per_second"],
        "all_language": language_total / result["language"]["tokens_per_second"],
        "training_total": nca_total / result["nca"]["tokens_per_second"] +
                          language_total / result["language"]["tokens_per_second"],
    }
    atomic_json(output, result)
    mark_done(directory / "benchmark", config, benchmark_sha256=digest(output),
              runtime_seconds=time.monotonic()-started)
    return result


def base_initialization(config: dict) -> Path:
    directory = ROOT / "config/base_random_init"
    output = directory / "base_random_init.pth"
    if stage_done(directory, config, (output,)):
        print("SKIP base_random_init", flush=True)
        return output
    directory.mkdir(parents=True, exist_ok=True)
    model = create_model(TEXT_MODEL, INITIALIZATION_SEED)
    if trainable_parameters(model) != 1_081_177:
        raise ValueError("model parameter count changed")
    temporary = output.with_suffix(".pth.tmp")
    torch.save({"model": model.state_dict(), "model_config": asdict(TEXT_MODEL),
                "initialization_seed": INITIALIZATION_SEED}, temporary)
    os.replace(temporary, output)
    mark_done(directory, config, sha256=digest(output))
    return output


def latest_checkpoint(directory: Path) -> Path | None:
    paths = sorted(directory.glob("model_*_iter.pth"),
                   key=lambda path: int(path.stem.split("_")[1]))
    for path in reversed(paths):
        if path.with_suffix(".json").is_file():
            return path
    return None


def checkpoint(state, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint.tmp-", dir=directory))
    try:
        source = save_checkpoint(state, temporary)
        final = directory / source.name
        os.replace(source, final)
        os.replace(source.with_suffix(".json"), final.with_suffix(".json"))
        return final
    finally:
        shutil.rmtree(temporary)


def record_metrics(directory: Path, state) -> None:
    atomic_json(directory / "metrics.json", {"history": state.history})


def train_stage(name: str, config: dict, dataset, validation,
                model_config: ModelConfig, milestones: tuple[int, ...],
                physical: int, initial_checkpoint: Path,
                nca_checkpoint: Path | None = None) -> Path:
    directory = ROOT / name
    checkpoint_dir = directory / "checkpoints"
    done = directory / "done.json"
    order, prefix = ordered(dataset, ORDER_SEED)
    budget = int(prefix[-1])
    steps = math.ceil(dataset.size / EFFECTIVE_BATCH)
    if stage_done(directory, config, (done,)):
        value = json.loads(done.read_text())
        final = directory / value["final_checkpoint"]
        if not final.is_file() or value["tokens_consumed"] != budget:
            raise ValueError(f"completed training stage invalid: {name}")
        print(f"SKIP {name} complete", flush=True)
        return final
    directory.mkdir(parents=True, exist_ok=True)
    train_config = TrainingConfig(steps=steps, batch_size=EFFECTIVE_BATCH,
                                  learning_rate=LR, weight_decay=0.0,
                                  warmup_steps=WARMUP_STEPS, device="mps",
                                  seed=TRAINING_SEED, eval_interval=CHECKPOINT_STEPS)
    prior = latest_checkpoint(checkpoint_dir) if checkpoint_dir.exists() else None
    if prior:
        state = load_checkpoint(prior)
        if state.model_config != model_config or state.config != train_config:
            raise ValueError(f"checkpoint configuration mismatch: {name}")
        print(f"RESUME {name} from step {state.step}", flush=True)
    else:
        model = create_model(model_config, INITIALIZATION_SEED)
        if name.startswith("nca/"):
            load_pretrained_body(model, initial_checkpoint)
        else:
            payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"])
            if nca_checkpoint is not None:
                load_pretrained_body(model, nca_checkpoint)
        state = prepare(model, model_config, train_config,
                        dataset_metadata={"stage": name, "config": config},
                        model_seed=INITIALIZATION_SEED)
    with (directory / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps({"started_at_utc": utc(), "pid": os.getpid(),
                                 "resumed_from_step": state.step}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    mask = create_attention_mask(CONTEXT, additive=True).to(state.device)
    stage_started = time.monotonic()
    interval_started = stage_started
    progress_path = directory / "progress.json"
    previous_runtime = 0.0
    if prior and progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress["step"] == state.step:
            previous_runtime = progress.get("runtime_seconds", 0.0)
            if progress.get("runtime_semantics") != "cumulative":
                elapsed = 0.0
                last_tokens = 0
                for row in state.history:
                    if row["tokens"] and row["tokens_per_second"]:
                        elapsed += (row["tokens"]-last_tokens) / row["tokens_per_second"]
                        last_tokens = row["tokens"]
                    row["elapsed_seconds"] = elapsed
                previous_runtime += elapsed
    previous_tokens = int(prefix[min(state.step*EFFECTIVE_BATCH, dataset.size)])
    next_milestone = next((x for x in milestones if x > previous_tokens), None)
    if state.step == 0 and not state.history:
        value = evaluate_model(state.model, validation, physical, mask)
        state.history.append({"step": 0, "tokens": 0, "train_loss": None,
                              "validation_loss": value,
                              "perplexity": math.exp(value),
                              "learning_rate": state.optimizer.param_groups[0]["lr"],
                              "tokens_per_second": None, "elapsed_seconds": 0.0})
        record_metrics(directory, state)
        checkpoint(state, checkpoint_dir)
        print(f"EVAL {name} tokens=0 loss={value:.5f}", flush=True)
    accumulated_loss = 0.0
    accumulated_steps = 0
    for step in range(state.step + 1, steps + 1):
        start_index = (step-1)*EFFECTIVE_BATCH
        positions = order[start_index:min(start_index+EFFECTIVE_BATCH, dataset.size)]
        train_loss = update_step(state, dataset, positions, physical, mask)
        state.step = step
        accumulated_loss += train_loss
        accumulated_steps += 1
        consumed = int(prefix[min(step*EFFECTIVE_BATCH, dataset.size)])
        need_eval = (next_milestone is not None and consumed >= next_milestone) or step == steps
        need_save = need_eval or step % CHECKPOINT_STEPS == 0
        if need_eval:
            val_loss = evaluate_model(state.model, validation, physical, mask)
            if not math.isfinite(val_loss):
                raise FloatingPointError(f"non-finite validation loss: {name}")
            elapsed = time.monotonic()-interval_started
            delta_tokens = consumed-previous_tokens
            row = {"step": step, "tokens": consumed,
                   "train_loss": accumulated_loss/max(accumulated_steps,1),
                   "validation_loss": val_loss,
                   "perplexity": math.exp(val_loss),
                   "learning_rate": state.optimizer.param_groups[0]["lr"],
                   "tokens_per_second": delta_tokens/max(elapsed,1e-9),
                   "elapsed_seconds": previous_runtime+time.monotonic()-stage_started}
            state.history.append(row)
            print(f"EVAL {name} step={step} tokens={consumed} "
                  f"train={row['train_loss']:.5f} val={val_loss:.5f} "
                  f"tokens_per_sec={row['tokens_per_second']:.0f}", flush=True)
            previous_tokens = consumed
            interval_started = time.monotonic()
            accumulated_loss = 0.0
            accumulated_steps = 0
            next_milestone = next((x for x in milestones if x > consumed), None)
        elif step % 100 == 0:
            print(f"PROGRESS {name} step={step}/{steps} tokens={consumed} "
                  f"train_loss={train_loss:.5f} lr={state.optimizer.param_groups[0]['lr']:.8f}", flush=True)
        if need_save:
            final = checkpoint(state, checkpoint_dir)
            record_metrics(directory, state)
            atomic_json(directory / "progress.json", {"step": step, "tokens": consumed,
                        "checkpoint": str(final.relative_to(directory)), "updated_at_utc": utc(),
                        "runtime_seconds": previous_runtime+time.monotonic()-stage_started,
                        "runtime_semantics": "cumulative"})
    final = latest_checkpoint(checkpoint_dir)
    if final is None:
        raise RuntimeError(f"no checkpoint for {name}")
    mark_done(directory, config, final_checkpoint=str(final.relative_to(directory)),
              tokens_consumed=budget, final_validation_loss=state.history[-1]["validation_loss"],
              final_perplexity=state.history[-1]["perplexity"],
              completed_steps=steps, runtime_seconds=previous_runtime+time.monotonic()-stage_started)
    print(f"DONE {name} tokens={budget} checkpoint={final}", flush=True)
    return final


def final_outputs(config: dict, benchmark_result: dict) -> None:
    directory = ROOT / "summary"
    output = directory / "summary.json"
    if stage_done(directory, config, (output, ROOT / "plots/nca_validation.png",
                                       ROOT / "plots/tinystories_validation.png")):
        print("SKIP final outputs", flush=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    histories = {}
    for name in [f"nca/{band}" for band in BANDS] + ["downstream/scratch"] + [f"downstream/nca_{band}" for band in BANDS]:
        histories[name] = json.loads((ROOT / name / "metrics.json").read_text())["history"]
    def plot(names, field, title, ylabel, filename):
        fig, axis = plt.subplots(figsize=(8, 5))
        for name in names:
            rows = histories[name]
            axis.plot([row["tokens"] for row in rows],
                      [row[field] for row in rows], marker="o", label=name.split("/")[-1])
        axis.set(title=title, xlabel="Tokens consumed", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(ROOT / "plots" / filename, dpi=150)
        plt.close(fig)
    plot([f"nca/{band}" for band in BANDS], "validation_loss",
         "Held-out-rule NCA validation", "Cross-entropy loss", "nca_validation.png")
    names = ["downstream/scratch"] + [f"downstream/nca_{band}" for band in BANDS]
    plot(names, "validation_loss", "TinyStories validation", "Cross-entropy loss",
         "tinystories_validation.png")
    plot(names, "perplexity", "TinyStories validation perplexity", "Perplexity",
         "tinystories_perplexity.png")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                             stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        git_commit = None
    scratch = {row["tokens"]: row["validation_loss"] for row in histories["downstream/scratch"]}
    attempts = {name: [json.loads(line) for line in (ROOT / name / "attempts.jsonl").read_text().splitlines()]
                for name in histories}
    stage_runtimes = {name: json.loads((ROOT / name / "done.json").read_text())["runtime_seconds"]
                      for name in histories}
    setup_runtimes = {"tinystories": json.loads((ROOT / "data/tinystories/done.json").read_text())["runtime_seconds"],
                      "benchmark": json.loads((ROOT / "config/benchmark/done.json").read_text())["runtime_seconds"],
                      **{f"pack_{band}": json.loads((ROOT / "data/nca" / band / "done.json").read_text())["elapsed_seconds"]
                         for band in BANDS}}
    summary = {
        "one_seed_exploratory": True, "config": config,
        "text_trainable_parameters": benchmark_result["language"]["parameters"],
        "nca_trainable_parameters": benchmark_result["nca"]["parameters"],
        "benchmark": benchmark_result,
        "git_commit": git_commit,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "torch": torch.__version__, "jax": jax.__version__, "backend": "mps"},
        "stages": {name: json.loads((ROOT / name / "done.json").read_text()) for name in histories},
        "stage_runtimes_seconds": stage_runtimes,
        "setup_runtimes_seconds": setup_runtimes,
        "total_recorded_runtime_seconds": sum(stage_runtimes.values())+sum(setup_runtimes.values()),
        "histories": histories,
        "loss_delta_from_scratch_at_matching_tokens": {
            name: {str(row["tokens"]): row["validation_loss"]-scratch[row["tokens"]]
                   for row in histories[name] if row["tokens"] in scratch}
            for name in names if name != "downstream/scratch"
        },
        "failed_or_retried_stages": {name: records for name, records in attempts.items()
                                     if len(records) > 1},
    }
    atomic_json(output, summary)
    rows = []
    for name in names:
        final = histories[name][-1]
        rows.append((name, final["validation_loss"], final["perplexity"]))
    with (directory / "README.md").open("w") as stream:
        stream.write("# One-seed exploratory NCA → TinyStories transfer\n\n")
        stream.write(f"Model: {benchmark_result['language']['parameters']:,} trainable parameters; "
                     f"{asdict(TEXT_MODEL)}.\n\n")
        stream.write(f"NCA budget: {config['nca_usable_training_tokens_per_band']:,} usable labels "
                     f"per regime. TinyStories: {LANGUAGE_TOKENS:,} training tokens per condition.\n\n")
        stream.write("| Condition | Final validation loss | Final perplexity |\n| --- | ---: | ---: |\n")
        for name, loss, ppl in rows:
            stream.write(f"| {name} | {loss:.5f} | {ppl:.2f} |\n")
        stream.write("\nSee summary.json for milestones, throughput, runtime, and loss differences. "
                     "This one-seed run does not establish statistical significance.\n")
    mark_done(directory, config, summary_sha256=digest(output))
    print("QUEUE COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true",
                        help="prepare inputs and benchmark, then stop before training")
    options = parser.parse_args()
    config = configuration()
    init_root(config)
    print(f"START {utc()} pid={os.getpid()} root={ROOT}", flush=True)
    prepare_tiny(config)
    base = base_initialization(config)
    benchmark_result = benchmark(config)
    for band in BANDS:
        pack_band(band, config)
    if options.setup:
        print("SETUP COMPLETE", flush=True)
        return
    nca_checkpoints = {}
    for band in BANDS:
        packed = PackedDataset.load(ROOT / "data/nca" / band / "packed.pt")
        train = NCAWindows(packed, "train", config["nca_train_worlds_per_band"])
        validation = NCAWindows(packed, "validation", EVAL_NCA_WORLDS)
        if int(train.valid.sum()) != config["nca_usable_training_tokens_per_band"]:
            raise ValueError(f"NCA token budget differs in {band}")
        nca_checkpoints[band] = train_stage(f"nca/{band}", config, train, validation,
                                             NCA_MODEL, NCA_MILESTONES,
                                             NCA_PHYSICAL_BATCH, base)
        del packed, train, validation
    train_text = TextWindows(ROOT / "data/tinystories/train.bin", LANGUAGE_TOKENS)
    validation_text = TextWindows(ROOT / "data/tinystories/val.bin", VALIDATION_TOKENS)
    for name, prior in [("scratch", None)] + [(f"nca_{band}", nca_checkpoints[band])
                                              for band in BANDS]:
        train_stage(f"downstream/{name}", config, train_text, validation_text,
                    TEXT_MODEL, LANGUAGE_MILESTONES, LANGUAGE_PHYSICAL_BATCH,
                    base, prior)
    final_outputs(config, benchmark_result)


if __name__ == "__main__":
    main()
