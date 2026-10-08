"""Resume only the frozen dose-sweep winner from 1M to 10M TinyStories tokens."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from examples import transfer_1m as v1
from progress.boosted_nca.rank_sweep import HISTORIES, choose, load
from sampleeff_nca_pretraining.training import TrainingConfig, load_checkpoint
from sampleeff_shared.models import create_attention_mask


SOURCE = REPO / "runs/1m-nca-transfer-v2"
ROOT = REPO / "runs/nca-data-search/boosted-nca"
OUTPUT = ROOT / "final-10m"
MILESTONES = (2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    decision_path = ROOT / "decision.json"
    decision = json.loads(decision_path.read_text())
    assert decision["schema"] == "nca-boosted-dose-decision-v1"
    names = list(decision["probe_metrics_sha256_by_tier"])
    assert all(sha(HISTORIES[name]) == digest for name, digest in
               decision["probe_metrics_sha256_by_tier"].items())
    gate, winner = choose([load(name) for name in names])
    assert winner == decision["selected_tier"] and gate == decision["top_5_gate_passed"]
    probe = REPO / decision["selected_probe_root"]
    source_checkpoint = REPO / decision["selected_checkpoint"]
    assert probe == HISTORIES[winner].parent
    assert sha(source_checkpoint) == decision["selected_checkpoint_sha256"]
    probe_done = json.loads((probe / "done.json").read_text())
    probe_history = json.loads((probe / "metrics.json").read_text())["history"]
    assert probe_done["completed_steps"] == 489 and probe_done["tokens_consumed"] == 1_001_472
    assert source_checkpoint == probe / probe_done["final_checkpoint"]
    assert source_checkpoint.with_suffix(".json").is_file()
    assert [row["tokens"] for row in probe_history] == [0, 251_904, 501_760, 1_001_472]

    config = {
        "schema": "nca-boosted-winner-10m-v1", "selected_tier": winner,
        "decision_sha256": sha(decision_path),
        "source_checkpoint": decision["selected_checkpoint"],
        "source_checkpoint_sha256": decision["selected_checkpoint_sha256"],
        "tinystories_train_sha256": "fbed8efccdbd42cd9c803f770a6e18da40c3be37461da54106fbac2c7b140ef3",
        "tinystories_validation_sha256": "851de2118d56178237f2b87270d02803433dd7fb6b4b5c841bb05afb4a5385e5",
        "full_tokens": 10_000_000, "model_seed": 0,
    }
    assert sha(SOURCE / "data/tinystories/train.bin") == config["tinystories_train_sha256"]
    assert sha(SOURCE / "data/tinystories/val.bin") == config["tinystories_validation_sha256"]
    config_path = OUTPUT / "config.json"
    if config_path.is_file():
        assert json.loads(config_path.read_text()) == config
    else:
        v1.atomic_json(config_path, config)
    done_path = OUTPUT / "done.json"
    if done_path.is_file():
        done = json.loads(done_path.read_text())
        assert done["config_sha256"] == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert done["tokens_consumed"] == 10_000_000
        assert (OUTPUT / done["final_checkpoint"]).is_file()
        print(f"SKIP {winner}: final 10M complete", flush=True)
        return

    data = v1.TextWindows(SOURCE / "data/tinystories/train.bin", 10_000_000)
    validation = v1.TextWindows(SOURCE / "data/tinystories/val.bin", v1.VALIDATION_TOKENS)
    order, prefix = v1.ordered(data, 17)
    full_steps = math.ceil(data.size / v1.EFFECTIVE_BATCH)
    assert full_steps == probe_done["full_schedule_steps"] == 4883
    train_config = TrainingConfig(steps=full_steps, batch_size=v1.EFFECTIVE_BATCH,
                                  learning_rate=v1.LR, weight_decay=0.0,
                                  warmup_steps=v1.WARMUP_STEPS, device="mps", seed=23,
                                  eval_interval=v1.CHECKPOINT_STEPS)
    reference_history = json.loads((SOURCE / "seeds/seed_0/downstream/nca_40_50/metrics.json").read_text())["history"]
    assert [(r["step"], r["tokens"], r["learning_rate"]) for r in probe_history] == [
        (r["step"], r["tokens"], r["learning_rate"]) for r in reference_history[:4]]

    checkpoint_dir = OUTPUT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prior = v1.latest_checkpoint(checkpoint_dir)
    state = load_checkpoint(prior or source_checkpoint)
    assert state.model_config == v1.TEXT_MODEL and state.config == train_config
    assert state.model_seed == 0 and state.history[:4] == probe_history
    assert 489 <= state.step <= full_steps
    for current, baseline in zip(state.history, reference_history):
        assert (current["step"], current["tokens"], current["learning_rate"]) == (
            baseline["step"], baseline["tokens"], baseline["learning_rate"])
    if prior is None:
        v1.checkpoint(state, checkpoint_dir)
    v1.record_metrics(OUTPUT, state)
    mask = create_attention_mask(v1.CONTEXT, additive=True).to(state.device)
    consumed = int(prefix[min(state.step*v1.EFFECTIVE_BATCH, data.size)])
    next_milestone = next((m for m in MILESTONES if m > consumed), None)
    loss_sum, loss_steps = 0., 0
    for step in range(state.step+1, full_steps+1):
        positions = order[(step-1)*v1.EFFECTIVE_BATCH:min(step*v1.EFFECTIVE_BATCH, data.size)]
        loss_sum += v1.update_step(state, data, positions, v1.LANGUAGE_PHYSICAL_BATCH, mask)
        loss_steps += 1
        state.step = step
        consumed = int(prefix[min(step*v1.EFFECTIVE_BATCH, data.size)])
        if next_milestone is not None and consumed >= next_milestone:
            value = v1.evaluate_model(state.model, validation, v1.LANGUAGE_PHYSICAL_BATCH, mask)
            row = {"step": step, "tokens": consumed, "train_loss": loss_sum/loss_steps,
                   "validation_loss": value, "perplexity": math.exp(value),
                   "learning_rate": state.optimizer.param_groups[0]["lr"]}
            baseline = reference_history[len(state.history)]
            assert (row["step"], row["tokens"], row["learning_rate"]) == (
                baseline["step"], baseline["tokens"], baseline["learning_rate"])
            state.history.append(row)
            v1.checkpoint(state, checkpoint_dir)
            v1.record_metrics(OUTPUT, state)
            print(f"{winner} EVAL tokens={consumed} val={value:.5f}", flush=True)
            loss_sum, loss_steps = 0., 0
            next_milestone = next((m for m in MILESTONES if m > consumed), None)
        elif step % v1.CHECKPOINT_STEPS == 0:
            v1.checkpoint(state, checkpoint_dir)
            v1.record_metrics(OUTPUT, state)
            print(f"{winner} step={step}/{full_steps} tokens={consumed}", flush=True)
    final = state.history[-1]
    assert final["tokens"] == 10_000_000 and len(state.history) == len(reference_history)
    checkpoint = v1.latest_checkpoint(checkpoint_dir)
    assert checkpoint is not None
    v1.mark_done(OUTPUT, config, final_checkpoint=str(checkpoint.relative_to(OUTPUT)),
                 tokens_consumed=final["tokens"], final_validation_loss=final["validation_loss"],
                 final_perplexity=final["perplexity"], completed_steps=state.step)
    print(f"{winner} DONE 10M loss={final['validation_loss']:.5f}", flush=True)


if __name__ == "__main__":
    main()
