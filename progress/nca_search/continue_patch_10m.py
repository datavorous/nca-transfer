"""Resume a successful patch-copy 1M probe through the original 10M schedule.

Run from repo root: uv run --all-packages python progress/nca_search/continue_patch_10m.py --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from examples import transfer_1m as v1
from sampleeff_nca_pretraining.training import TrainingConfig, load_checkpoint
from sampleeff_shared.models import create_attention_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=range(5))
    seed = parser.parse_args().seed
    corpus = REPO / "runs/nca-data-search/iteration-04-high-patch-copy"
    root = corpus if seed == 0 else corpus / "seeds" / f"seed_{seed}"
    reference = REPO / "runs/1m-nca-transfer-v2/seeds" / f"seed_{seed}"
    probe = root / "downstream/probe_1m"
    output = root / "downstream/full_10m"
    checkpoint_dir = output / "checkpoints"
    probe_config = json.loads((root / "probe_config.json").read_text())
    probe_done = json.loads((probe / "done.json").read_text())
    assert probe_done["completed_steps"] == 489
    assert probe_done["tokens_consumed"] == 1_001_472
    assert probe_done["config_sha256"] == hashlib.sha256(
        json.dumps(probe_config, sort_keys=True).encode()).hexdigest()
    config = {**probe_config, "name": "nca-high-patch-copy-iteration-04-full-10m",
              "continued_from": str(probe / probe_done["final_checkpoint"]),
              "full_tokens": 10_000_000}
    config_path = root / "full_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    else:
        v1.atomic_json(config_path, config)
    done_path = output / "done.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        assert done["config_sha256"] == hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert done["tokens_consumed"] == 10_000_000
        print(f"SKIP seed={seed} full 10M complete", flush=True)
        return

    data = v1.TextWindows(REPO / "runs/1m-nca-transfer-v2/data/tinystories/train.bin", 10_000_000)
    validation = v1.TextWindows(REPO / "runs/1m-nca-transfer-v2/data/tinystories/val.bin",
                                v1.VALIDATION_TOKENS)
    order, prefix = v1.ordered(data, 17)
    full_steps = math.ceil(data.size / v1.EFFECTIVE_BATCH)
    train_config = TrainingConfig(steps=full_steps, batch_size=v1.EFFECTIVE_BATCH,
                                  learning_rate=v1.LR, weight_decay=0.0,
                                  warmup_steps=v1.WARMUP_STEPS, device="mps", seed=23,
                                  eval_interval=v1.CHECKPOINT_STEPS)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    prior = v1.latest_checkpoint(checkpoint_dir)
    state = load_checkpoint(prior or probe / probe_done["final_checkpoint"])
    assert state.model_config == v1.TEXT_MODEL and state.config == train_config
    assert state.step >= 489
    reference_history = json.loads((reference / "downstream/nca_40_50/metrics.json").read_text())["history"]
    for current, baseline in zip(state.history, reference_history):
        assert (current["step"], current["tokens"], current["learning_rate"]) == (
            baseline["step"], baseline["tokens"], baseline["learning_rate"])
    if prior is None:
        v1.record_metrics(output, state)
        v1.checkpoint(state, checkpoint_dir)
    mask = create_attention_mask(v1.CONTEXT, additive=True).to(state.device)
    milestones = (2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000)
    consumed = int(prefix[min(state.step * v1.EFFECTIVE_BATCH, data.size)])
    next_milestone = next((m for m in milestones if m > consumed), None)
    loss_sum, loss_steps = 0.0, 0
    for step in range(state.step + 1, full_steps + 1):
        positions = order[(step - 1) * v1.EFFECTIVE_BATCH:
                          min(step * v1.EFFECTIVE_BATCH, data.size)]
        loss_sum += v1.update_step(state, data, positions, v1.LANGUAGE_PHYSICAL_BATCH, mask)
        loss_steps += 1
        state.step = step
        consumed = int(prefix[min(step * v1.EFFECTIVE_BATCH, data.size)])
        if next_milestone is not None and consumed >= next_milestone:
            value = v1.evaluate_model(state.model, validation,
                                      v1.LANGUAGE_PHYSICAL_BATCH, mask)
            row = {"step": step, "tokens": consumed, "train_loss": loss_sum / loss_steps,
                   "validation_loss": value, "perplexity": math.exp(value),
                   "learning_rate": state.optimizer.param_groups[0]["lr"]}
            baseline = reference_history[len(state.history)]
            assert (row["step"], row["tokens"], row["learning_rate"]) == (
                baseline["step"], baseline["tokens"], baseline["learning_rate"])
            state.history.append(row)
            v1.record_metrics(output, state)
            v1.checkpoint(state, checkpoint_dir)
            print(f"seed={seed} EVAL tokens={consumed} loss={value:.5f} "
                  f"reference={baseline['validation_loss']:.5f}", flush=True)
            loss_sum, loss_steps = 0.0, 0
            next_milestone = next((m for m in milestones if m > consumed), None)
        elif step % 500 == 0:
            print(f"seed={seed} step={step}/{full_steps} tokens={consumed}", flush=True)
    final = state.history[-1]
    assert final["tokens"] == 10_000_000 and len(state.history) == len(reference_history)
    checkpoint = v1.latest_checkpoint(checkpoint_dir)
    assert checkpoint is not None
    v1.mark_done(output, config, final_checkpoint=str(checkpoint.relative_to(output)),
                 tokens_consumed=final["tokens"], final_validation_loss=final["validation_loss"],
                 final_perplexity=final["perplexity"],
                 baseline_validation_loss=reference_history[-1]["validation_loss"],
                 gain_vs_reference=reference_history[-1]["validation_loss"]-final["validation_loss"],
                 completed_steps=state.step)
    print(f"seed={seed} DONE 10M candidate={final['validation_loss']:.5f} "
          f"reference={reference_history[-1]['validation_loss']:.5f}", flush=True)


if __name__ == "__main__":
    main()
