"""Continue one gated A/B/C patch-copy arm from 1M to the fixed 10M schedule.

The 45-run block-level gate is checked before creating output. This script never
retrains NCA or the first 1M language tokens. Run only after the active 1M
runner has finished, via run_patch_replication_10m.sh.
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
from progress.nca_search.gate_patch_replication_10m import gate, load_results, run_root
from sampleeff_nca_pretraining.training import TrainingConfig, load_checkpoint
from sampleeff_shared.models import create_attention_mask

SOURCE = REPO / "runs/1m-nca-transfer-v2"
MILESTONES = (2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices="abc", required=True)
    parser.add_argument("--arm", choices=("high", "unfiltered", "low"), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    if not gate(load_results()):
        parser.error("the prospective 45-run 1M gate failed")

    root = run_root(args.block, args.arm, args.seed)
    probe = root / "downstream/probe_1m"
    output = root / "downstream/full_10m"
    checkpoint_dir = output / "checkpoints"
    config_1m = json.loads((root / "probe_config.json").read_text())
    probe_done = json.loads((probe / "done.json").read_text())
    probe_history = json.loads((probe / "metrics.json").read_text())["history"]
    source_checkpoint = probe / probe_done["final_checkpoint"]
    assert probe_done["completed_steps"] == 489
    assert probe_done["tokens_consumed"] == 1_001_472
    assert probe_done["config_sha256"] == hashlib.sha256(
        json.dumps(config_1m, sort_keys=True).encode()).hexdigest()
    assert source_checkpoint.is_file() and source_checkpoint.with_suffix(".json").is_file()
    config = {**config_1m,
              "name": f"nca-patch-replication-block-{args.block}-{args.arm}-full-10m",
              "continued_from": str(source_checkpoint.relative_to(root)),
              "continued_from_sha256": sha(source_checkpoint),
              "full_tokens": 10_000_000}
    config_path = root / "full_config.json"
    if config_path.is_file():
        assert json.loads(config_path.read_text()) == config
    else:
        v1.atomic_json(config_path, config)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    done_path = output / "done.json"
    if done_path.is_file():
        done = json.loads(done_path.read_text())
        assert done["config_sha256"] == config_hash
        assert done["tokens_consumed"] == 10_000_000
        assert (output / done["final_checkpoint"]).is_file()
        print(f"SKIP block={args.block} arm={args.arm} seed={args.seed}: 10M complete", flush=True)
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
    reference_history = json.loads((SOURCE / "seeds" / f"seed_{args.seed}" /
                                    "downstream/nca_40_50/metrics.json").read_text())["history"]
    assert len(reference_history) == len(probe_history) + len(MILESTONES)
    assert [(r["step"], r["tokens"], r["learning_rate"]) for r in probe_history] == [
        (r["step"], r["tokens"], r["learning_rate"]) for r in reference_history[:4]]

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    prior = v1.latest_checkpoint(checkpoint_dir)
    state = load_checkpoint(prior or source_checkpoint)
    assert state.model_config == v1.TEXT_MODEL and state.config == train_config
    assert state.model_seed == args.seed
    assert state.history[:4] == probe_history
    assert 489 <= state.step <= full_steps
    assert state.history[-1]["step"] <= state.step
    assert len(state.history) <= len(reference_history)
    for current, baseline in zip(state.history, reference_history):
        assert (current["step"], current["tokens"], current["learning_rate"]) == (
            baseline["step"], baseline["tokens"], baseline["learning_rate"])
    if prior is None:
        v1.checkpoint(state, checkpoint_dir)
    v1.record_metrics(output, state)
    mask = create_attention_mask(v1.CONTEXT, additive=True).to(state.device)
    consumed = int(prefix[min(state.step * v1.EFFECTIVE_BATCH, data.size)])
    assert consumed >= state.history[-1]["tokens"]
    next_milestone = next((m for m in MILESTONES if m > consumed), None)
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
            v1.checkpoint(state, checkpoint_dir)
            v1.record_metrics(output, state)
            print(f"block={args.block} arm={args.arm} seed={args.seed} "
                  f"EVAL tokens={consumed} loss={value:.5f}", flush=True)
            loss_sum, loss_steps = 0.0, 0
            next_milestone = next((m for m in MILESTONES if m > consumed), None)
        elif step % v1.CHECKPOINT_STEPS == 0:
            v1.checkpoint(state, checkpoint_dir)
            v1.record_metrics(output, state)
            print(f"block={args.block} arm={args.arm} seed={args.seed} "
                  f"step={step}/{full_steps} tokens={consumed}", flush=True)
    final = state.history[-1]
    assert final["tokens"] == 10_000_000 and len(state.history) == len(reference_history)
    checkpoint = v1.latest_checkpoint(checkpoint_dir)
    assert checkpoint is not None
    v1.mark_done(output, config, final_checkpoint=str(checkpoint.relative_to(output)),
                 tokens_consumed=final["tokens"], final_validation_loss=final["validation_loss"],
                 final_perplexity=final["perplexity"],
                 completed_steps=state.step)
    print(f"block={args.block} arm={args.arm} seed={args.seed} "
          f"DONE 10M loss={final['validation_loss']:.5f}", flush=True)


if __name__ == "__main__":
    main()
