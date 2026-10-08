"""Fixed NCA stage and resumable short/1M TinyStories probe for vertical lag-6 arms.

Run from repo root: uv run --all-packages python progress/nca_search/probe_vertical_lag6.py --block d --arm high --seed 0
Requires a completed audit_vertical_lag6.py --block d receipt first.
The downstream optimizer schedule remains the original 10M-token schedule.
Use --stop-tokens 500000 for 100K/250K/500K triage before any 1M continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

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
parser.add_argument("--arm", choices=("high", "unfiltered", "low"), required=True)
parser.add_argument("--block", choices=("d", "e", "f"), required=True)
parser.add_argument("--seed", type=int, choices=range(5), default=0)
parser.add_argument("--stop-tokens", type=int,
                    choices=(100_000, 250_000, 500_000, 1_000_000),
                    default=1_000_000,
                    help="Stop early without marking the 1M probe complete; resume in the same output directory")
args = parser.parse_args()
SEED = args.seed
BLOCK_ROOT = REPO / "runs/nca-data-search/vertical-lag6" / f"block-{args.block}"
CORPUS_ROOT = BLOCK_ROOT / args.arm
ROOT = CORPUS_ROOT / "seeds" / f"seed_{SEED}"
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
    audit_path = BLOCK_ROOT / "audit.json"
    assert audit_path.is_file(), f"run audit_vertical_lag6.py --block {args.block} first"
    audit = json.loads(audit_path.read_text())
    assert audit["schema"] == "nca-vertical-lag6-audit-v1" and audit["block"] == args.block
    assert audit["earlier_rule_overlap"] == 0 and audit["selected_rules"] == 3 * 20_231
    assert sha(BLOCK_ROOT / "rule_index.jsonl") == audit["rule_index_sha256"]
    assert sha(BLOCK_ROOT / "sampling_config.json") == audit["sampling_config_sha256"]
    sampling_done = json.loads((CORPUS_ROOT / "done.json").read_text())
    assert sampling_done["schema"] == "nca-vertical-lag6-arm-v1"
    assert sampling_done["arm"] == args.arm and sampling_done["block"] == f"block-{args.block}"
    assert sampling_done["train_usable_tokens"] == 19_715_325
    packed_path = CORPUS_ROOT / "packed.pt"
    assert sha(packed_path) == sampling_done["packed_sha256"]
    assert audit["packed_sha256_by_arm"][args.arm] == sampling_done["packed_sha256"]
    reference_config = json.loads((SOURCE_SEED / "config/experiment.json").read_text())
    reference = SOURCE_SEED / "downstream/nca_40_50/metrics.json"
    baseline_history = json.loads(reference.read_text())["history"]
    baseline = next(row for row in baseline_history if row["tokens"] >= PROBE_TOKENS)
    assert baseline["tokens"] == 1_001_472
    config = {**reference_config,
              "name": f"nca-vertical-lag6-block-{args.block}-{args.arm}-probe",
              "sampling_change": f"fresh block {args.block}, hash-assigned {args.arm} arm; frozen vertical lag-6 excess threshold and matched gzip sub-bin quotas",
              "sampling_config_sha256": sha(BLOCK_ROOT / "sampling_config.json"),
              "audit_sha256": sha(audit_path),
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
    stage = f"nca/vertical_lag6_{args.arm}"
    return v1.train_stage(stage, config, train, validation,
                          v1.NCA_MODEL, v1.NCA_MILESTONES,
                          v1.NCA_PHYSICAL_BATCH, base)


def short_result(directory: Path, history: list[dict], target: int) -> dict:
    if target == 100_000:
        receipt = json.loads((directory / "early_100k.json").read_text())
    else:
        receipt = next(row for row in history if row["tokens"] >= target)
    return {"tokens_consumed": receipt["tokens"],
            "final_validation_loss": receipt["validation_loss"],
            "complete_1m": False}


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
    stop_step = int(np.searchsorted(prefix, args.stop_tokens, side="left"))
    stop_step = math.ceil(stop_step / v1.EFFECTIVE_BATCH)
    assert stop_step <= 489
    if args.stop_tokens == PROBE_TOKENS:
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
    consumed_before = int(prefix[min(state.step*v1.EFFECTIVE_BATCH, data.size)])
    if args.stop_tokens < PROBE_TOKENS and state.step >= stop_step:
        return short_result(directory, state.history, args.stop_tokens)
    next_milestone = next((m for m in milestones if m > consumed_before), None)
    loss_sum, loss_steps = 0., 0
    for step in range(state.step+1, stop_step+1):
        positions = order[(step-1)*v1.EFFECTIVE_BATCH:min(step*v1.EFFECTIVE_BATCH, data.size)]
        loss_sum += v1.update_step(state, data, positions, v1.LANGUAGE_PHYSICAL_BATCH, mask)
        loss_steps += 1
        state.step = step
        consumed = int(prefix[min(step*v1.EFFECTIVE_BATCH, data.size)])
        if consumed >= 100_000 and not (directory / "early_100k.json").is_file():
            val = v1.evaluate_model(state.model, validation, v1.LANGUAGE_PHYSICAL_BATCH, mask)
            v1.atomic_json(directory / "early_100k.json", {
                "step": step, "tokens": consumed, "validation_loss": val,
                "perplexity": math.exp(val), "checkpoint_history": "separate from canonical 1M milestones"})
            v1.checkpoint(state, checkpoint_dir)
            print(f"PROBE EARLY tokens={consumed} val={val:.5f}", flush=True)
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
    if args.stop_tokens < PROBE_TOKENS:
        assert state.step == stop_step
        assert int(prefix[min(state.step*v1.EFFECTIVE_BATCH, data.size)]) >= args.stop_tokens
        return short_result(directory, state.history, args.stop_tokens)
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
    if not result.get("complete_1m", True):
        print(json.dumps(result, indent=2), flush=True)
        return
    from diagnose_vertical_lag6_nca import analyze
    diagnostic = analyze(args.block, args.arm, SEED, checkpoint)
    print(json.dumps({"candidate_loss": result["final_validation_loss"],
                      "reference_loss": result["baseline_validation_loss"],
                      "gain": result["gain_vs_reference"],
                      "nca_visible_lag6_loss": diagnostic["validation_loss"]["visible_vertical"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
