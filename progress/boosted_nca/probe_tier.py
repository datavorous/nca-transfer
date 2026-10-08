"""Full NCA stage and fixed-schedule 1M TinyStories probe for one sweep tier.

Run from repo root: uv run --all-packages python progress/boosted_nca/probe_tier.py --tier top_25
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
parser.add_argument("--tier", choices=("top_25", "top_10", "top_5"), required=True)
args = parser.parse_args()
CORPUS = REPO / "runs/nca-data-search/boosted-nca/sweep-a" / args.tier
ROOT = CORPUS / "seeds/seed_0"
SOURCE = REPO / "runs/1m-nca-transfer-v2"
REFERENCE = REPO / "runs/nca-data-search/patch-replication/block-a/high/seeds/seed_0"
SEED = 0
TOKENS = 1_001_472


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_config() -> dict:
    audit_path = CORPUS.parent / "audit.json"
    assert audit_path.is_file(), "run audit_sweep.py after both corpora finish"
    audit = json.loads(audit_path.read_text())
    assert audit["schema"] == "nca-boosted-sweep-audit-v1"
    receipt = json.loads((CORPUS / "done.json").read_text())
    assert receipt["tier"] == args.tier and receipt["schema"] == "nca-boosted-sweep-arm-v1"
    assert receipt["train_worlds"] == 19_975 and receipt["validation_worlds"] == 256
    assert receipt["train_usable_tokens"] == 19_715_325
    assert sha(CORPUS / "packed.pt") == receipt["packed_sha256"]
    assert sha(CORPUS.parent / "sampling_config.json") == receipt["sampling_config_sha256"]
    assert audit["packed_sha256_by_tier"][args.tier] == receipt["packed_sha256"]
    assert audit["sampling_config_sha256"] == receipt["sampling_config_sha256"]
    reference_metrics = REFERENCE / "downstream/probe_1m/metrics.json"
    reference_history = json.loads(reference_metrics.read_text())["history"]
    assert [row["tokens"] for row in reference_history] == [0, 251_904, 501_760, TOKENS]
    source_config = json.loads((SOURCE / "seeds/seed_0/config/experiment.json").read_text())
    config = {**source_config,
              "name": f"nca-boosted-{args.tier}-block-a-seed-0-probe",
              "sampling_change": "stricter horizontal patch-copy excess within original gzip-bin quotas, same block-A high-arm rule stream",
              "candidate_packed_sha256": receipt["packed_sha256"],
              "candidate_receipt_sha256": sha(CORPUS / "done.json"),
              "candidate_audit_sha256": sha(audit_path),
              "sampling_config_sha256": receipt["sampling_config_sha256"],
              "reference_metrics_sha256": sha(reference_metrics),
              "probe_checkpoint_tokens": TOKENS,
              "probe_tokens": 1_000_000}
    path = ROOT / "probe_config.json"
    if path.exists():
        assert json.loads(path.read_text()) == config
    else:
        v1.atomic_json(path, config)
    return config


def train_nca(config: dict) -> Path:
    v1.ROOT = ROOT
    v1.INITIALIZATION_SEED = SEED
    v1.ORDER_SEED = 17
    v1.TRAINING_SEED = 23
    packed = PackedDataset.load(CORPUS / "packed.pt")
    training = v1.NCAWindows(packed, "train", config["nca_train_worlds_per_band"])
    validation = v1.NCAWindows(packed, "validation", v1.EVAL_NCA_WORLDS)
    assert int(training.valid.sum()) == 19_715_325
    assert len(validation.indices) == 256
    base = SOURCE / "seeds/seed_0/config/base_random_init/base_random_init.pth"
    assert base.is_file()
    return v1.train_stage(f"nca/boosted_{args.tier}", config, training, validation,
                          v1.NCA_MODEL, v1.NCA_MILESTONES,
                          v1.NCA_PHYSICAL_BATCH, base)


def train_probe(config: dict, nca_checkpoint: Path) -> dict:
    directory = ROOT / "downstream/probe_1m"
    done_path = directory / "done.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        assert done["config_sha256"] == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert (directory / done["final_checkpoint"]).is_file()
        assert done["tokens_consumed"] == TOKENS
        return done
    directory.mkdir(parents=True, exist_ok=True)
    data = v1.TextWindows(SOURCE / "data/tinystories/train.bin", 10_000_000)
    validation = v1.TextWindows(SOURCE / "data/tinystories/val.bin", v1.VALIDATION_TOKENS)
    order, prefix = v1.ordered(data, 17)
    full_steps = math.ceil(data.size / v1.EFFECTIVE_BATCH)
    stop_step = math.ceil(int(np.searchsorted(prefix, 1_000_000, side="left")) / v1.EFFECTIVE_BATCH)
    assert stop_step == 489 and int(prefix[stop_step * v1.EFFECTIVE_BATCH]) == TOKENS
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
        base = SOURCE / "seeds/seed_0/config/base_random_init/base_random_init.pth"
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
        print(f"PROBE {args.tier} tokens=0 val={value:.5f}", flush=True)
    consumed_before = int(prefix[min(state.step*v1.EFFECTIVE_BATCH, data.size)])
    next_milestone = next((m for m in milestones if m > consumed_before), None)
    loss_sum, loss_steps = 0., 0
    for step in range(state.step+1, stop_step+1):
        positions = order[(step-1)*v1.EFFECTIVE_BATCH:min(step*v1.EFFECTIVE_BATCH, data.size)]
        loss_sum += v1.update_step(state, data, positions, v1.LANGUAGE_PHYSICAL_BATCH, mask)
        loss_steps += 1
        state.step = step
        consumed = int(prefix[min(step*v1.EFFECTIVE_BATCH, data.size)])
        if next_milestone is not None and consumed >= next_milestone:
            value = v1.evaluate_model(state.model, validation, v1.LANGUAGE_PHYSICAL_BATCH, mask)
            state.history.append({"step": step, "tokens": consumed,
                                  "train_loss": loss_sum/loss_steps,
                                  "validation_loss": value, "perplexity": math.exp(value),
                                  "learning_rate": state.optimizer.param_groups[0]["lr"]})
            v1.record_metrics(directory, state)
            v1.checkpoint(state, checkpoint_dir)
            print(f"PROBE {args.tier} tokens={consumed} val={value:.5f}", flush=True)
            loss_sum, loss_steps = 0., 0
            next_milestone = next((m for m in milestones if m > consumed), None)
        elif step % 100 == 0:
            print(f"PROBE {args.tier} step={step}/{stop_step} tokens={consumed}", flush=True)
    final = state.history[-1]
    assert final["tokens"] == TOKENS
    checkpoint = v1.latest_checkpoint(checkpoint_dir)
    assert checkpoint is not None
    v1.mark_done(directory, config, final_checkpoint=str(checkpoint.relative_to(directory)),
                 tokens_consumed=final["tokens"], final_validation_loss=final["validation_loss"],
                 final_perplexity=final["perplexity"], completed_steps=state.step,
                 full_schedule_steps=full_steps)
    return json.loads(done_path.read_text())


def main() -> None:
    config = setup_config()
    checkpoint = train_nca(config)
    result = train_probe(config, checkpoint)
    print(json.dumps({"tier": args.tier, "validation_loss": result["final_validation_loss"],
                      "tokens": result["tokens_consumed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
