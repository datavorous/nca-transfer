"""Frozen seed-0 10M-model transfer queue; use --smoke before --full."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from examples import transfer_1m as old
from sampleeff_nca_pretraining.model import ModelConfig, create_model, trainable_parameters
from sampleeff_nca_pretraining.packing import PackedDataset
from sampleeff_nca_pretraining.training import TrainingConfig, prepare
from sampleeff_shared.models import create_attention_mask

ROOT = REPO / "runs/scale-transfer-10m"
SOURCE = REPO / "runs/1m-nca-transfer-v2"
ORDINARY = SOURCE / "data/nca/40_50/packed.pt"
DESIGNED = REPO / "runs/nca-data-search/boosted-nca/sweep-a/top_10/packed.pt"
DESIGNED_RECEIPT = DESIGNED.parent / "done.json"
TOP10_CUTOFFS = (0.034107767225, 0.025746229122, 0.017753705187,
                 0.012177637072, 0.008144630498)
SEED = 0
PASSES = 10
NCA_LABELS = 19_715_325 * PASSES
TEXT_TOKENS = 100_000_000
VAL_TOKENS = 262_144
NCA_MODEL = ModelConfig(vocab_size=10_002, context_length=256, layers=8,
                        heads=4, embedding_dim=144, dropout=0.0, tie_embeddings=True)
TEXT_MODEL = ModelConfig(vocab_size=50_257, context_length=256, layers=8,
                         heads=4, embedding_dim=144, dropout=0.0, tie_embeddings=True)
NCA_MILESTONES = tuple(range(0, 200_000_001, 20_000_000))
TEXT_MILESTONES = (0, 250_000, 500_000, 1_000_000, 2_000_000,
                   3_000_000, 5_000_000, 7_500_000, 10_000_000,
                   15_000_000, 20_000_000, 30_000_000, 50_000_000,
                   75_000_000, 100_000_000)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(stage: str, **fields) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"utc": utc(), "stage": stage, **fields}, sort_keys=True) + "\n")
    print(f"STAGE {stage} {fields}", flush=True)


def configure_old() -> None:
    old.ROOT = ROOT
    old.INITIALIZATION_SEED = SEED
    old.ORDER_SEED = 17
    old.TRAINING_SEED = 23
    old.NCA_MODEL = NCA_MODEL
    old.TEXT_MODEL = TEXT_MODEL
    old.NCA_PHYSICAL_BATCH = 4
    old.LANGUAGE_PHYSICAL_BATCH = 2
    old.EFFECTIVE_BATCH = 8
    old.CHECKPOINT_STEPS = 5_000


def frozen_config() -> dict:
    receipt = json.loads(DESIGNED_RECEIPT.read_text())
    assert receipt["tier"] == "top_10"
    assert tuple(receipt["cutoffs_by_bin"]) == TOP10_CUTOFFS
    assert receipt["train_worlds"] == 19_975
    assert receipt["validation_worlds"] == 256
    assert receipt["train_usable_tokens"] == 19_715_325
    assert sha(DESIGNED) == receipt["packed_sha256"]
    decision = json.loads((DESIGNED.parents[2] / "decision.json").read_text())
    assert decision["selected_tier"] == "top_10"
    original_data = json.loads((SOURCE / "data/tinystories/metadata.json").read_text())
    assert original_data["revision"] == old.TINY_REVISION
    assert original_data["tokenizer"] == "gpt2"
    assert sha(SOURCE / "data/tinystories/val.bin") == original_data["validation_sha256"]
    config = {
        "schema": "sampleeff-10x-scale-transfer-v1",
        "question": "Does the frozen 1M-model Top-10 recipe outperform ordinary NCA at 10M parameters?",
        "seed": SEED, "conditions": ["scratch", "ordinary_nca", "designed_nca"],
        "model_nca": asdict(NCA_MODEL), "model_text": asdict(TEXT_MODEL),
        "text_trainable_parameters": 9_943_921,
        "nca_passes": PASSES, "nca_scored_labels_per_arm": NCA_LABELS,
        "nca_corpus_policy": "ten shuffled passes over each frozen 19,975-world train split",
        "ordinary_packed_sha256": sha(ORDINARY),
        "designed_packed_sha256": receipt["packed_sha256"],
        "designed_receipt_sha256": sha(DESIGNED_RECEIPT),
        "frozen_tier_cutoffs_by_gzip_bin": TOP10_CUTOFFS,
        "gzip_bins": [(0.40, 0.42), (0.42, 0.44), (0.44, 0.46),
                      (0.46, 0.48), (0.48, 0.50)],
        "tiny_dataset": old.TINY_DATASET, "tiny_revision": old.TINY_REVISION,
        "tokenizer": "gpt2", "validation_sha256": original_data["validation_sha256"],
        "original_10m_train_sha256": original_data["train_sha256"],
        "language_scored_tokens_per_condition": TEXT_TOKENS,
        "validation_scored_tokens": VAL_TOKENS,
        "optimizer": "Adam", "learning_rate": old.LR, "weight_decay": 0.0,
        "warmup_steps": old.WARMUP_STEPS,
        "scheduler": "linear warmup then cosine over each full stage",
        "effective_batch": old.EFFECTIVE_BATCH,
        "nca_physical_batch": old.NCA_PHYSICAL_BATCH,
        "language_physical_batch": old.LANGUAGE_PHYSICAL_BATCH,
        "order_seed": old.ORDER_SEED, "training_seed": old.TRAINING_SEED,
        "nca_milestones": NCA_MILESTONES,
        "language_milestones": TEXT_MILESTONES,
        "checkpoint_steps": old.CHECKPOINT_STEPS,
        "code_sha256": sha(Path(__file__)),
        "source_code_sha256": {
            str(path.relative_to(REPO)): sha(path)
            for path in (REPO / "examples/transfer_1m.py",
                         REPO / "packages/shared/src/sampleeff_shared/models.py",
                         REPO / "packages/pre-pre-training/src/sampleeff_nca_pretraining/packing.py",
                         REPO / "packages/pre-pre-training/src/sampleeff_nca_pretraining/training.py",
                         REPO / "uv.lock")
        },
    }
    return config


def save_config(config: dict) -> None:
    path = ROOT / "config.json"
    if path.is_file():
        if json.loads(path.read_text()) != json.loads(json.dumps(config)):
            raise ValueError("frozen scale-transfer config differs from existing run")
    else:
        old.atomic_json(path, config)


class RepeatedNCA:
    def __init__(self, base: old.NCAWindows, passes: int):
        self.base = base
        self.size = base.size * passes
        self.valid = np.tile(base.valid, passes)

    def batch(self, positions: np.ndarray):
        return self.base.batch(positions % self.base.size)


def prepare_text(config: dict) -> Path:
    from datasets import load_dataset
    import tiktoken

    directory = ROOT / "data/tinystories"
    path = directory / "train.bin"
    metadata_path = directory / "metadata.json"
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if (metadata["tokens_saved"] != TEXT_TOKENS + 1 or
                path.stat().st_size != 2 * (TEXT_TOKENS + 1) or
                sha(path) != metadata["sha256"]):
            raise ValueError("saved 100M TinyStories stream is invalid")
        return path
    directory.mkdir(parents=True, exist_ok=True)
    temp = directory / "train.bin.tmp"
    temp.unlink(missing_ok=True)
    event("prepare_tinystories_start")
    encoder = tiktoken.get_encoding("gpt2")
    rows_seen = written = 0
    rows = load_dataset(old.TINY_DATASET, split="train", streaming=True,
                        revision=old.TINY_REVISION)
    with temp.open("wb") as stream:
        for row in rows:
            tokens = encoder.encode_ordinary(row["text"])
            tokens.append(encoder.eot_token)
            take = min(len(tokens), TEXT_TOKENS + 1 - written)
            np.asarray(tokens[:take], dtype=np.uint16).tofile(stream)
            written += take
            rows_seen += 1
            if rows_seen % 50_000 == 0:
                event("prepare_tinystories_progress", rows=rows_seen, tokens=written)
            if written == TEXT_TOKENS + 1:
                break
        stream.flush()
        os.fsync(stream.fileno())
    if written != TEXT_TOKENS + 1:
        raise ValueError(f"TinyStories exhausted after {written} tokens")
    original_bytes = 2 * (10_000_000 + 1)
    h = hashlib.sha256()
    with temp.open("rb") as stream:
        left = original_bytes
        while left:
            block = stream.read(min(1 << 20, left))
            if not block:
                raise ValueError("missing original 10M-token prefix")
            h.update(block)
            left -= len(block)
    assert h.hexdigest() == config["original_10m_train_sha256"], "TinyStories prefix drift"
    os.replace(temp, path)
    old.atomic_json(metadata_path, {"source": old.TINY_DATASET,
        "revision": old.TINY_REVISION, "tokenizer": "gpt2", "rows_seen": rows_seen,
        "tokens_saved": written, "sha256": sha(path),
        "original_10m_prefix_sha256": h.hexdigest()})
    event("prepare_tinystories_done", rows=rows_seen, tokens=written)
    return path


def base_initialization(config: dict) -> Path:
    output = ROOT / "config/base_random_init.pth"
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        assert payload["model_config"] == asdict(TEXT_MODEL)
        assert payload["initialization_seed"] == SEED
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    model = create_model(TEXT_MODEL, SEED)
    assert trainable_parameters(model) == config["text_trainable_parameters"]
    temp = output.with_suffix(".pth.tmp")
    torch.save({"model": model.state_dict(), "model_config": asdict(TEXT_MODEL),
                "initialization_seed": SEED}, temp)
    os.replace(temp, output)
    event("base_initialization_done", parameters=trainable_parameters(model), sha256=sha(output))
    return output


class Telemetry:
    """Best-effort system observations; battery power is never equated with AC system power."""

    def __init__(self):
        self.stop = threading.Event()
        self.stage = "setup"
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self) -> None:
        import csv
        pid = os.getpid()
        ps = subprocess.run(["ps", "-p", str(pid), "-o", "rss=,%cpu="],
                            capture_output=True, text=True, timeout=10)
        parts = ps.stdout.split()
        battery = subprocess.run(["ioreg", "-r", "-c", "AppleSmartBattery", "-d", "1"],
                                 capture_output=True, text=True, timeout=10).stdout
        def field(name):
            m = re.search(r'^\s*"' + re.escape(name) + r'" = (.+)$', battery, re.M)
            return m.group(1).strip() if m else ""
        amperage = field("Amperage")
        if amperage.isdigit() and int(amperage) >= 2**63:
            amperage = str(int(amperage) - 2**64)
        row = [utc(), time.monotonic(), self.stage, pid,
               parts[0] if len(parts) > 0 else "", parts[1] if len(parts) > 1 else "",
               field("ExternalConnected"), field("IsCharging"), field("Voltage"),
               amperage, field("CurrentCapacity")]
        path = ROOT / "telemetry.csv"
        with path.open("a", newline="") as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(["utc", "monotonic_seconds", "stage", "pid", "rss_kib",
                                 "cpu_percent", "ac_connected", "charging", "battery_mv",
                                 "battery_ma", "battery_percent"])
            writer.writerow(row)

    def loop(self) -> None:
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception as exc:
                event("telemetry_error", error=repr(exc))
            self.stop.wait(60)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=12)


def smoke(config: dict) -> None:
    from datasets import load_dataset
    import tiktoken

    assert torch.backends.mps.is_available(), "MPS backend required"
    assert trainable_parameters(create_model(TEXT_MODEL)) == 9_943_921
    row = next(iter(load_dataset(old.TINY_DATASET, split="train", streaming=True,
                                 revision=old.TINY_REVISION)))
    assert tiktoken.get_encoding("gpt2").encode_ordinary(row["text"])
    with tempfile.TemporaryDirectory(prefix="sampleeff-scale-smoke-") as tmp:
        base = Path(tmp) / "init.pth"
        model = create_model(TEXT_MODEL, SEED)
        torch.save({"model": model.state_dict(), "model_config": asdict(TEXT_MODEL),
                    "initialization_seed": SEED}, base)
        text = old.TextWindows(SOURCE / "data/tinystories/train.bin", 10_000_000)
        text_val = old.TextWindows(SOURCE / "data/tinystories/val.bin", VAL_TOKENS)
        mask = create_attention_mask(256, additive=True).to("mps")
        results = {}
        nca_smoke_checkpoints = {}
        for arm, packed_path in (("ordinary", ORDINARY), ("designed", DESIGNED)):
            packed = PackedDataset.load(packed_path)
            train = old.NCAWindows(packed, "train", 19_975)
            val = old.NCAWindows(packed, "validation", 256)
            assert int(train.valid.sum()) == 19_715_325
            repeated = RepeatedNCA(train, PASSES)
            assert int(repeated.valid.sum()) == NCA_LABELS
            nca = create_model(NCA_MODEL, SEED)
            old.load_pretrained_body(nca, base)
            state = prepare(nca, NCA_MODEL, TrainingConfig(steps=2, batch_size=8,
                learning_rate=old.LR, warmup_steps=0, device="mps", seed=23))
            start = time.monotonic()
            for _ in range(2):
                old.update_step(state, repeated, np.arange(8), 4, mask)
            val_loss = old.evaluate_model(state.model, val, 4, mask, limit=4)
            assert math.isfinite(val_loss)
            results[arm + "_nca"] = {"two_steps_seconds": time.monotonic()-start,
                                     "validation_loss": val_loss}
            nca_smoke_checkpoints[arm] = Path(tmp) / f"{arm}_nca.pth"
            torch.save({"model": state.model.state_dict()}, nca_smoke_checkpoints[arm])
            del state, nca, packed, train, val
            torch.mps.empty_cache()
        for arm in ("scratch", "ordinary", "designed"):
            language_model = create_model(TEXT_MODEL, SEED)
            language_model.load_state_dict(torch.load(base, map_location="cpu", weights_only=False)["model"])
            if arm != "scratch":
                old.load_pretrained_body(language_model, nca_smoke_checkpoints[arm])
            state = prepare(language_model, TEXT_MODEL,
                            TrainingConfig(steps=2, batch_size=8,
                                learning_rate=old.LR, warmup_steps=0,
                                device="mps", seed=23))
            start = time.monotonic()
            for _ in range(2):
                old.update_step(state, text, np.arange(8), 2, mask)
            val_loss = old.evaluate_model(state.model, text_val, 2, mask, limit=2)
            assert math.isfinite(val_loss)
            results[arm + "_language"] = {"two_steps_seconds": time.monotonic()-start,
                                          "validation_loss": val_loss}
            del state
            torch.mps.empty_cache()
    ROOT.mkdir(parents=True, exist_ok=True)
    old.atomic_json(ROOT / "smoke.json", {"completed_at_utc": utc(),
        "schema": "sampleeff-scale-smoke-v1", "model_parameters": 9_943_921,
        "result": results})
    print(json.dumps(results, indent=2), flush=True)


def full(config: dict) -> None:
    assert (ROOT / "smoke.json").is_file(), "run --smoke first"
    assert torch.backends.mps.is_available(), "MPS backend required"
    save_config(config)
    telemetry = Telemetry()
    telemetry.start()
    try:
        telemetry.stage = "prepare_tinystories"
        train_path = prepare_text(config)
        base = base_initialization(config)
        ordinary_packed = PackedDataset.load(ORDINARY)
        designed_packed = PackedDataset.load(DESIGNED)
        nca_checkpoints = {}
        for arm, packed in (("ordinary", ordinary_packed), ("designed", designed_packed)):
            telemetry.stage = "nca_" + arm
            event(telemetry.stage + "_start")
            train = RepeatedNCA(old.NCAWindows(packed, "train", 19_975), PASSES)
            val = old.NCAWindows(packed, "validation", 256)
            assert int(train.valid.sum()) == NCA_LABELS
            nca_checkpoints[arm] = old.train_stage("nca/" + arm, config, train, val,
                NCA_MODEL, NCA_MILESTONES, 4, base)
            event(telemetry.stage + "_done")
            del train, val
            torch.mps.empty_cache()
        del ordinary_packed, designed_packed
        text = old.TextWindows(train_path, TEXT_TOKENS)
        val = old.TextWindows(SOURCE / "data/tinystories/val.bin", VAL_TOKENS)
        for arm in ("scratch", "ordinary", "designed"):
            telemetry.stage = "downstream_" + arm
            event(telemetry.stage + "_start")
            checkpoint = nca_checkpoints.get(arm)
            old.train_stage("downstream/" + arm, config, text, val,
                TEXT_MODEL, TEXT_MILESTONES, 2, base, checkpoint)
            event(telemetry.stage + "_done")
            torch.mps.empty_cache()
        telemetry.stage = "complete"
        event("complete")
        subprocess.run([sys.executable, str(Path(__file__).with_name("summary.py"))],
                       cwd=REPO, check=True)
    except BaseException as exc:
        event("failed", error=repr(exc))
        raise
    finally:
        telemetry.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    configure_old()
    config = frozen_config()
    if args.smoke:
        smoke(config)
    else:
        full(config)


if __name__ == "__main__":
    main()
