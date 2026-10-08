"""Paper-ratio seed-0 NCA warm-up test using the frozen scale-transfer inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from examples import transfer_1m as train
from progress.scale_transfer_10m import run as frozen
from progress.scale_transfer_10m import summarize_replication as calc
from sampleeff_nca_pretraining.packing import PackedDataset

ROOT = frozen.ROOT / "paper-ratio" / "seed_0"
NCA_TARGETS = (2_000_000, 4_000_000)
NCA_ACTUAL = (2_001_179, 4_001_087)
NCA_STEPS = (1_013, 2_025)
TEXT_STOP = 10_000_000
TEXT_ACTUAL = 10_000_384
EARLY_POINTS = (1_001_472, 2_000_896, 3_000_320, 5_001_216, 7_501_824, TEXT_ACTUAL)


class PauseAfterCheckpoint(Exception):
    pass


def source_audit() -> dict:
    frozen.configure_old()
    reference = json.loads((frozen.ROOT / "config.json").read_text())
    assert json.loads(json.dumps(frozen.frozen_config())) == reference
    assert torch.backends.mps.is_available()
    for name in ("scratch", "ordinary", "designed"):
        assert (frozen.ROOT / "downstream" / name / "done.json").is_file()
    for arm in ("ordinary", "designed"):
        source = frozen.ROOT / "nca" / arm / "checkpoints/model_0_iter.pth"
        assert source.is_file() and source.with_suffix(".json").is_file()
        files = sorted((source.parent).glob("model_*_iter.pth"),
                       key=lambda path: int(path.stem.split("_")[1]))
        assert len(files) == 30 and int(files[1].stem.split("_")[1]) == 5_000
    assert frozen.sha(frozen.ROOT / "data/tinystories/train.bin") == json.loads(
        (frozen.ROOT / "data/tinystories/metadata.json").read_text())["sha256"]
    assert frozen.sha(frozen.SOURCE / "data/tinystories/val.bin") == reference["validation_sha256"]
    return reference


def configure() -> None:
    frozen.configure_old()
    train.ROOT = ROOT


def nca_data(arm: str):
    path = frozen.ORDINARY if arm == "ordinary" else frozen.DESIGNED
    packed = PackedDataset.load(path)
    data = frozen.RepeatedNCA(train.NCAWindows(packed, "train", 19_975), frozen.PASSES)
    val = train.NCAWindows(packed, "validation", 256)
    _, prefix = train.ordered(data, train.ORDER_SEED)
    for target, expected_step, actual in zip(NCA_TARGETS, NCA_STEPS, NCA_ACTUAL):
        step = math.ceil(int(np.searchsorted(prefix, target, side="left")) /
                         train.EFFECTIVE_BATCH)
        assert step == expected_step
        assert int(prefix[min(step * train.EFFECTIVE_BATCH, data.size)]) == actual
    return packed, data, val


def text_data():
    text = train.TextWindows(frozen.ROOT / "data/tinystories/train.bin", frozen.TEXT_TOKENS)
    val = train.TextWindows(frozen.SOURCE / "data/tinystories/val.bin", frozen.VAL_TOKENS)
    _, prefix = train.ordered(text, train.ORDER_SEED)
    step = math.ceil(int(np.searchsorted(prefix, TEXT_STOP, side="left")) /
                     train.EFFECTIVE_BATCH)
    assert step == 4_883
    assert int(prefix[min(step * train.EFFECTIVE_BATCH, text.size)]) == TEXT_ACTUAL
    return text, val


def nca_config(reference: dict, arm: str) -> dict:
    return {**reference, "schema": "sampleeff-paper-ratio-nca-v1", "arm": arm,
            "source_zero_checkpoint_sha256": frozen.sha(
                frozen.ROOT / "nca" / arm / "checkpoints/model_0_iter.pth"),
            "source_config_sha256": frozen.sha(frozen.ROOT / "config.json"),
            "checkpoint_only_targets": list(NCA_TARGETS)}


def downstream_config(reference: dict, arm: str, dose: int, checkpoint: Path) -> dict:
    return {**reference, "schema": "sampleeff-paper-ratio-downstream-v1",
            "arm": arm, "nca_scored_labels": NCA_ACTUAL[dose // 2 - 1],
            "nca_checkpoint_sha256": frozen.sha(checkpoint),
            "source_config_sha256": frozen.sha(frozen.ROOT / "config.json"),
            "full_downstream_tokens": frozen.TEXT_TOKENS}


def pause_stage(name: str, config: dict, dataset, validation, model_config,
                milestones: tuple[int, ...], physical: int, initial: Path,
                nca_checkpoint: Path | None, stop_step: int, expected_tokens: int,
                checkpoint_only_labels: dict[int, int] | None = None) -> None:
    """Use the original training loop/schedule; pause before the update after its saved milestone."""
    original = train.update_step

    def guarded(state, data, positions, batch, mask):
        if state.step >= stop_step:
            raise PauseAfterCheckpoint
        next_step = state.step + 1
        loss = original(state, data, positions, batch, mask)
        if checkpoint_only_labels and next_step in checkpoint_only_labels:
            state.step = next_step
            directory = ROOT / name
            saved = train.checkpoint(state, directory / "checkpoints")
            train.atomic_json(directory / f"checkpoint_only_{next_step}.json", {
                "step": next_step, "scored_labels": checkpoint_only_labels[next_step],
                "checkpoint": str(saved.relative_to(directory)),
                "validation_evaluation_added": False})
        return loss

    train.update_step = guarded
    try:
        train.train_stage(name, config, dataset, validation, model_config,
                          milestones, physical, initial, nca_checkpoint)
    except PauseAfterCheckpoint:
        pass
    finally:
        train.update_step = original
    directory = ROOT / name
    checkpoint = directory / "checkpoints" / f"model_{stop_step}_iter.pth"
    assert checkpoint.is_file() and checkpoint.with_suffix(".json").is_file()
    if checkpoint_only_labels:
        receipt = json.loads((directory / f"checkpoint_only_{stop_step}.json").read_text())
        assert receipt["scored_labels"] == expected_tokens
    else:
        history = json.loads((directory / "metrics.json").read_text())["history"]
        assert any(row["step"] == stop_step and row["tokens"] == expected_tokens for row in history)
    print(f"PAUSED {name} at {expected_tokens:,} scored tokens", flush=True)
    torch.mps.empty_cache()


def link_zero(arm: str) -> None:
    source = frozen.ROOT / "nca" / arm / "checkpoints/model_0_iter.pth"
    destination = ROOT / "nca" / arm / "checkpoints/model_0_iter.pth"
    destination.parent.mkdir(parents=True, exist_ok=True)
    for origin, target in ((source, destination),
                           (source.with_suffix(".json"), destination.with_suffix(".json"))):
        if target.is_symlink():
            assert target.resolve() == origin.resolve()
        else:
            assert not target.exists()
            target.symlink_to(origin)


def nca_checkpoint(arm: str, dose: int) -> Path:
    step = NCA_STEPS[dose // 2 - 1]
    path = ROOT / "nca" / arm / "checkpoints" / f"model_{step}_iter.pth"
    assert path.is_file() and path.with_suffix(".json").is_file()
    return path


def warm_nca(arm: str, dose: int, reference: dict) -> Path:
    assert dose in (2, 4)
    configure()
    link_zero(arm)
    packed, data, validation = nca_data(arm)
    target_index = dose // 2 - 1
    checkpoint = ROOT / "nca" / arm / "checkpoints" / f"model_{NCA_STEPS[target_index]}_iter.pth"
    if checkpoint.is_file():
        print(f"REUSE {arm} {dose}M NCA checkpoint", flush=True)
    else:
        print(f"NCA {arm}: resume original step-0 checkpoint to {NCA_ACTUAL[target_index]:,} labels",
              flush=True)
        pause_stage("nca/" + arm, nca_config(reference, arm), data, validation,
                    frozen.NCA_MODEL, frozen.NCA_MILESTONES, train.NCA_PHYSICAL_BATCH,
                    frozen.ROOT / "config/base_random_init.pth", None,
                    NCA_STEPS[target_index], NCA_ACTUAL[target_index],
                    dict(zip(NCA_STEPS, NCA_ACTUAL)))
    del packed, data, validation
    torch.mps.empty_cache()
    return nca_checkpoint(arm, dose)


def downstream_name(arm: str, dose: int) -> str:
    return f"downstream/{arm}_{dose}m"


def downstream_probe(arm: str, dose: int, reference: dict, checkpoint: Path) -> None:
    configure()
    name = downstream_name(arm, dose)
    directory = ROOT / name
    marker = directory / "probe_10m_done.json"
    if marker.is_file():
        assert json.loads(marker.read_text())["nca_checkpoint_sha256"] == frozen.sha(checkpoint)
        print(f"REUSE {name} 10M probe", flush=True)
        return
    text, validation = text_data()
    print(f"TEXT {name}: probe to {TEXT_ACTUAL:,} tokens on full 100M schedule", flush=True)
    pause_stage(name, downstream_config(reference, arm, dose, checkpoint), text, validation,
                frozen.TEXT_MODEL, frozen.TEXT_MILESTONES, train.LANGUAGE_PHYSICAL_BATCH,
                frozen.ROOT / "config/base_random_init.pth", checkpoint, 4_883, TEXT_ACTUAL)
    train.atomic_json(marker, {"schema": "paper-ratio-10m-probe-v1", "arm": arm,
        "dose_labels": NCA_ACTUAL[dose // 2 - 1],
        "nca_checkpoint_sha256": frozen.sha(checkpoint),
        "downstream_tokens": TEXT_ACTUAL})


def downstream_full(arm: str, dose: int, reference: dict, checkpoint: Path) -> None:
    configure()
    name = downstream_name(arm, dose)
    text, validation = text_data()
    print(f"TEXT {name}: resume through 100M", flush=True)
    train.train_stage(name, downstream_config(reference, arm, dose, checkpoint),
                      text, validation, frozen.TEXT_MODEL, frozen.TEXT_MILESTONES,
                      train.LANGUAGE_PHYSICAL_BATCH,
                      frozen.ROOT / "config/base_random_init.pth", checkpoint)
    torch.mps.empty_cache()


def history(arm: str, dose: int) -> list[dict]:
    return json.loads((ROOT / downstream_name(arm, dose) / "metrics.json").read_text())["history"]


def scratch() -> list[dict]:
    return json.loads((frozen.ROOT / "downstream/scratch/metrics.json").read_text())["history"]


def early_decision() -> dict:
    saved = ROOT / "early_decision.json"
    if saved.is_file():
        return json.loads(saved.read_text())
    h = {dose: history("designed", dose) for dose in (2, 4)}
    s = scratch()
    values = {dose: [next(row["validation_loss"] for row in h[dose]
                          if row["tokens"] == token) for token in EARLY_POINTS] for dose in (2, 4)}
    scratch_values = [next(row["validation_loss"] for row in s if row["tokens"] == token)
                      for token in EARLY_POINTS]
    areas = {dose: calc.auc(h[dose], 0, TEXT_STOP) for dose in (2, 4)}
    selected = [2, 4]
    for weaker, stronger in ((2, 4), (4, 2)):
        if (all(values[weaker][i] >= values[stronger][i] + .03 and
                values[weaker][i] >= scratch_values[i] + .03
                for i in range(len(EARLY_POINTS))) and
                areas[weaker] >= areas[stronger] + .03):
            selected = [stronger]
            break
        if (sum(values[weaker][i] >= values[stronger][i] + .02
                for i in range(len(EARLY_POINTS))) >= 5 and
                values[weaker][-1] >= values[stronger][-1] + .02 and
                areas[weaker] >= areas[stronger] + .02):
            selected = [stronger]
            break
    receipt = {"schema": "paper-ratio-early-decision-v1", "selected_for_100m": selected,
               "rule": "Extend both unless one is worse than the other at five of six "
                       "early checkpoints by at least .02, and worse at the 10M endpoint "
                       "and 0–10M AUC by at least .02; or worse than both other and Scratch "
                       "at every checkpoint by .03 with AUC worse by .03.",
               "checkpoints": list(EARLY_POINTS), "losses": values,
               "scratch_losses": scratch_values, "auc": areas}
    train.atomic_json(saved, receipt)
    print(f"EARLY DECISION: extend Designed {selected}", flush=True)
    return receipt


def final_decision(selected: list[int]) -> dict:
    saved = ROOT / "final_decision.json"
    if saved.is_file():
        return json.loads(saved.read_text())
    ranked = []
    for dose in selected:
        h = history("designed", dose)
        assert h[-1]["tokens"] == 100_000_000
        ranked.append((dose, h[-1]["validation_loss"], calc.auc(h, 0, 100_000_000)))
    best_loss = min(row[1] for row in ranked)
    near = [row for row in ranked if row[1] <= best_loss + .01]
    winner = min(near, key=lambda row: (row[2], row[0]))[0]
    receipt = {"schema": "paper-ratio-final-decision-v1", "matched_ordinary_dose": winner,
               "rule": "Lowest Designed 100M validation loss; within .01 loss, lower "
                       "0–100M AUC; remaining tie favors lower NCA dose.",
               "ranked": [{"dose": d, "loss": loss, "auc": area} for d, loss, area in ranked]}
    train.atomic_json(saved, receipt)
    print(f"FINAL DECISION: matched Ordinary dose {winner}M", flush=True)
    return receipt


def report() -> None:
    subprocess.run([sys.executable, str(Path(__file__).with_name("paper_ratio_report.py"))],
                   cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.audit == args.run:
        parser.error("choose --audit or --run")
    reference = source_audit()
    print("Existing: NCA step-0 and 9.87M+ checkpoints; seed-0 Scratch, 9.87M, "
          "197.15M histories; old sweep artifacts preserved.", flush=True)
    print("New NCA: Designed 4,001,087 labels, then Ordinary 2,001,179 or "
          "4,001,087 after selection. New downstream branches: Designed 2M/4M, "
          "one matched Ordinary; first two stop at 10M text.", flush=True)
    if args.audit:
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    designed_4 = warm_nca("designed", 4, reference)
    del designed_4
    for dose in (2, 4):
        downstream_probe("designed", dose, reference, nca_checkpoint("designed", dose))
    decision = early_decision()
    report()
    for dose in decision["selected_for_100m"]:
        downstream_full("designed", dose, reference, nca_checkpoint("designed", dose))
    final = final_decision(decision["selected_for_100m"])
    dose = final["matched_ordinary_dose"]
    ordinary = warm_nca("ordinary", dose, reference)
    downstream_full("ordinary", dose, reference, ordinary)
    report()
    print("PAPER-RATIO DIAGNOSTIC COMPLETE", flush=True)


if __name__ == "__main__":
    main()
