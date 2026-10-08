"""Matched NCA stage and 1M-token downstream probe for change-rate cohorts.

Run from repo root: uv run --all-packages python progress/nca_search/probe_low_change.py
The downstream optimizer schedule remains the original 10M-token schedule.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from examples import transfer_1m as v1
from sampleeff_nca_pretraining.model import create_model
from sampleeff_nca_pretraining.packing import PackedDataset
from sampleeff_nca_pretraining.training import (TrainingConfig, load_checkpoint,
                                                load_pretrained_body, prepare)
from sampleeff_shared.models import create_attention_mask


parser = argparse.ArgumentParser()
variant = parser.add_mutually_exclusive_group()
variant.add_argument("--high", action="store_true")
variant.add_argument("--spatial-high", action="store_true")
variant.add_argument("--patch-high", action="store_true")
variant.add_argument("--patch-low", action="store_true")
variant.add_argument("--transition-high", action="store_true")
variant.add_argument("--replication-arm", choices=("high", "unfiltered", "low"))
parser.add_argument("--block", choices=("a", "b", "c"))
parser.add_argument("--seed", type=int, choices=range(5), default=0)
args = parser.parse_args()
if (args.replication_arm is None) != (args.block is None):
    parser.error("--replication-arm and --block must be supplied together")
HIGH = args.high
SPATIAL = args.spatial_high
TRANSITION = args.transition_high
REPLICATION = args.replication_arm is not None
PATCH_LOW = args.patch_low
PATCH = args.patch_high or PATCH_LOW
SEED = args.seed
CORPUS_ROOT = REPO / (f"runs/nca-data-search/patch-replication/block-{args.block}/{args.replication_arm}" if REPLICATION else
                     "runs/nca-data-search/iteration-06-high-transition-mi" if TRANSITION else
                     "runs/nca-data-search/iteration-05-low-patch-copy" if PATCH_LOW else
                     "runs/nca-data-search/iteration-04-high-patch-copy" if PATCH else
                     "runs/nca-data-search/iteration-03-high-spatial" if SPATIAL else
                     "runs/nca-data-search/iteration-02-high-change" if HIGH else
                     "runs/nca-data-search/iteration-01-low-change")
ROOT = CORPUS_ROOT / "seeds" / f"seed_{SEED}" if REPLICATION or SEED != 0 else CORPUS_ROOT
SOURCE = REPO / "runs/1m-nca-transfer-v2"
PROBE_TOKENS = 1_000_000
SOURCE_SEED = SOURCE / "seeds" / f"seed_{SEED}"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def setup_config() -> dict:
    if REPLICATION:
        block_root = CORPUS_ROOT.parent
        audit_path = block_root / "audit.json"
        assert audit_path.is_file(), f"run audit_patch_replication.py --block {args.block} first"
        audit = json.loads(audit_path.read_text())
        assert audit["earlier_rule_overlap"] == 0 and audit["selected_rules"] == 3 * 20_231
        assert sha(block_root / "rule_index.jsonl") == audit["rule_index_sha256"]
        assert sha(block_root / "sampling_config.json") == audit["sampling_config_sha256"]
    sampling_done = json.loads((CORPUS_ROOT / "done.json").read_text())
    packed_path = CORPUS_ROOT / "packed.pt"
    assert sha(packed_path) == sampling_done["packed_sha256"]
    reference_config = json.loads((SOURCE_SEED / "config/experiment.json").read_text())
    reference = SOURCE_SEED / "downstream/nca_40_50/metrics.json"
    baseline_history = json.loads(reference.read_text())["history"]
    baseline = next(row for row in baseline_history if row["tokens"] >= PROBE_TOKENS)
    assert baseline["tokens"] == 1_001_472
    config = {**reference_config,
              "name": (f"nca-patch-replication-block-{args.block}-{args.replication_arm}-probe" if REPLICATION else
                       "nca-high-transition-mi-iteration-06-probe" if TRANSITION else
                       "nca-low-patch-copy-iteration-05-probe" if PATCH_LOW else
                       "nca-high-patch-copy-iteration-04-probe" if PATCH else
                       "nca-high-spatial-iteration-03-probe" if SPATIAL else
                       "nca-high-change-iteration-02-probe" if HIGH else
                       "nca-low-change-iteration-01-probe"),
              "sampling_change": (f"fresh block {args.block}, hash-assigned {args.replication_arm} arm; frozen patch-copy threshold and matched gzip sub-bin quotas"
                                  if REPLICATION else
                                  "cell-transition mutual information >= 40–50 reference median with matched gzip sub-bin quotas"
                                  if TRANSITION else
                                  "horizontal patch-copy excess <= 40–50 reference median with matched gzip sub-bin quotas"
                                  if PATCH_LOW else
                                  "horizontal patch-copy excess >= 40–50 reference median with matched gzip sub-bin quotas"
                                  if PATCH else
                                  "spatial excess agreement >= 40–50 reference median with matched gzip sub-bin quotas"
                                  if SPATIAL else
                                  "mean cell-change fraction >= 40–50 reference median with matched gzip sub-bin quotas"
                                  if HIGH else "mean cell-change fraction <= 40–50 reference median with matched gzip sub-bin quotas"),
              "candidate_packed_sha256": sha(packed_path),
              "reference_seed": SEED,
              "reference_metrics_sha256": sha(reference),
              "probe_tokens": PROBE_TOKENS,
              "probe_checkpoint_tokens": baseline["tokens"],
              "baseline_validation_loss": baseline["validation_loss"]}
    config_path = ROOT / "probe_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    else:
        v1.atomic_json(config_path, config)
    return config


def train_nca(config: dict) -> Path:
    v1.ROOT = ROOT
    v1.INITIALIZATION_SEED = SEED
    v1.ORDER_SEED = 17
    v1.TRAINING_SEED = 23
    packed = PackedDataset.load(CORPUS_ROOT / "packed.pt")
    train = v1.NCAWindows(packed, "train", config["nca_train_worlds_per_band"])
    validation = v1.NCAWindows(packed, "validation", v1.EVAL_NCA_WORLDS)
    assert int(train.valid.sum()) == 19_715_325
    assert len(validation.indices) == 256
    base = SOURCE_SEED / "config/base_random_init/base_random_init.pth"
    assert base.is_file()
    stage = (f"nca/replication_{args.replication_arm}" if REPLICATION else
             "nca/high_transition_mi" if TRANSITION else
             "nca/low_patch_copy" if PATCH_LOW else "nca/high_patch_copy" if PATCH else
             "nca/high_spatial" if SPATIAL else
             "nca/high_change" if HIGH else "nca/low_change")
    return v1.train_stage(stage, config, train, validation,
                          v1.NCA_MODEL, v1.NCA_MILESTONES,
                          v1.NCA_PHYSICAL_BATCH, base)


def train_probe(config: dict, nca_checkpoint: Path) -> dict:
    directory = ROOT / "downstream/probe_1m"
    done_path = directory / "done.json"
    reference = json.loads((SOURCE_SEED / "downstream/nca_40_50/metrics.json").read_text())
    baseline = next(row for row in reference["history"] if row["tokens"] >= PROBE_TOKENS)
    if done_path.is_file():
        done = json.loads(done_path.read_text())
        assert done["config_sha256"] == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert (directory / done["final_checkpoint"]).is_file()
        assert done["tokens_consumed"] == baseline["tokens"]
        return done
    directory.mkdir(parents=True, exist_ok=True)
    data = v1.TextWindows(SOURCE / "data/tinystories/train.bin", 10_000_000)
    validation = v1.TextWindows(SOURCE / "data/tinystories/val.bin", v1.VALIDATION_TOKENS)
    order, prefix = v1.ordered(data, 17)
    full_steps = math.ceil(data.size / v1.EFFECTIVE_BATCH)
    stop_step = int(np.searchsorted(prefix, PROBE_TOKENS, side="left"))
    stop_step = math.ceil(stop_step / v1.EFFECTIVE_BATCH)
    assert int(prefix[min(stop_step*v1.EFFECTIVE_BATCH, data.size)]) == baseline["tokens"]
    train_config = TrainingConfig(steps=full_steps, batch_size=v1.EFFECTIVE_BATCH,
                                  learning_rate=v1.LR, weight_decay=0.0,
                                  warmup_steps=v1.WARMUP_STEPS, device="mps",
                                  seed=23, eval_interval=v1.CHECKPOINT_STEPS)
    checkpoint_dir = directory / "checkpoints"
    prior = v1.latest_checkpoint(checkpoint_dir) if checkpoint_dir.exists() else None
    if prior:
        state = load_checkpoint(prior)
        assert state.model_config == v1.TEXT_MODEL and state.config == train_config
    else:
        model = create_model(v1.TEXT_MODEL, SEED)
        base = SOURCE_SEED / "config/base_random_init/base_random_init.pth"
        model.load_state_dict(torch.load(base, map_location="cpu", weights_only=False)["model"])
        load_pretrained_body(model, nca_checkpoint)
        state = prepare(model, v1.TEXT_MODEL, train_config,
                        dataset_metadata={"stage": "downstream/probe_1m", "config": config},
                        model_seed=SEED)
    mask = create_attention_mask(v1.CONTEXT, additive=True).to(state.device)
    milestones = (0, 250_000, 500_000, 1_000_000)
    if state.step == 0 and not state.history:
        value = v1.evaluate_model(state.model, validation, v1.LANGUAGE_PHYSICAL_BATCH, mask)
        state.history.append({"step": 0, "tokens": 0, "train_loss": None,
                              "validation_loss": value, "perplexity": math.exp(value),
                              "learning_rate": state.optimizer.param_groups[0]["lr"]})
        v1.record_metrics(directory, state)
        v1.checkpoint(state, checkpoint_dir)
        print(f"PROBE EVAL tokens=0 val={value:.5f}", flush=True)
    next_milestone = next((m for m in milestones if m > int(prefix[min(state.step*8, data.size)])), None)
    loss_sum, loss_steps = 0., 0
    for step in range(state.step+1, stop_step+1):
        positions = order[(step-1)*v1.EFFECTIVE_BATCH:min(step*v1.EFFECTIVE_BATCH, data.size)]
        loss_sum += v1.update_step(state, data, positions, v1.LANGUAGE_PHYSICAL_BATCH, mask)
        loss_steps += 1
        state.step = step
        consumed = int(prefix[min(step*v1.EFFECTIVE_BATCH, data.size)])
        if next_milestone is not None and consumed >= next_milestone:
            val = v1.evaluate_model(state.model, validation, v1.LANGUAGE_PHYSICAL_BATCH, mask)
            row = {"step": step, "tokens": consumed, "train_loss": loss_sum/loss_steps,
                   "validation_loss": val, "perplexity": math.exp(val),
                   "learning_rate": state.optimizer.param_groups[0]["lr"]}
            state.history.append(row)
            v1.record_metrics(directory, state)
            v1.checkpoint(state, checkpoint_dir)
            print(f"PROBE EVAL tokens={consumed} val={val:.5f} baseline={baseline['validation_loss']:.5f}", flush=True)
            loss_sum, loss_steps = 0., 0
            next_milestone = next((m for m in milestones if m > consumed), None)
        elif step % 100 == 0:
            print(f"PROBE step={step}/{stop_step} tokens={consumed}", flush=True)
    final = state.history[-1]
    assert final["tokens"] == baseline["tokens"]
    checkpoint = v1.latest_checkpoint(checkpoint_dir)
    assert checkpoint is not None
    v1.mark_done(directory, config, final_checkpoint=str(checkpoint.relative_to(directory)),
                 tokens_consumed=final["tokens"], final_validation_loss=final["validation_loss"],
                 final_perplexity=final["perplexity"], baseline_validation_loss=baseline["validation_loss"],
                 gain_vs_reference=baseline["validation_loss"]-final["validation_loss"],
                 completed_steps=state.step, full_schedule_steps=full_steps)
    return json.loads(done_path.read_text())


def main() -> None:
    config = setup_config()
    checkpoint = train_nca(config)
    result = train_probe(config, checkpoint)
    print(json.dumps({"candidate_loss": result["final_validation_loss"],
                      "reference_loss": result["baseline_validation_loss"],
                      "gain": result["gain_vs_reference"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
