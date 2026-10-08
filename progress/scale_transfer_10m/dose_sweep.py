"""Seed-0 downstream dose diagnostic from frozen 10M-model NCA checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from examples import transfer_1m as train
from progress.scale_transfer_10m import run as frozen
from progress.scale_transfer_10m import summarize_replication as metrics
from sampleeff_nca_pretraining.packing import PackedDataset

ROOT = frozen.ROOT / "dose-sweep" / "seed_0"
TARGET_DOSES = (10, 25, 50, 100, 150, 200)
NEW_DOSES = TARGET_DOSES[:-1]
ARMS = ("ordinary", "designed")
PROBE_TOKENS = 10_000_000
TEXT_TOKENS = frozen.TEXT_TOKENS


class ProbeComplete(Exception):
    """Pause the original full-schedule training loop after its 10M checkpoint."""


def audit() -> tuple[dict, dict[str, dict[int, dict]]]:
    frozen.configure_old()
    reference = json.loads((frozen.ROOT / "config.json").read_text())
    assert json.loads(json.dumps(frozen.frozen_config())) == reference
    assert torch.backends.mps.is_available()
    for name in ("scratch", *ARMS):
        assert (frozen.ROOT / "downstream" / name / "done.json").is_file()
    corpus_paths = {"ordinary": frozen.ORDINARY, "designed": frozen.DESIGNED}
    result = {}
    for arm in ARMS:
        packed = PackedDataset.load(corpus_paths[arm])
        repeated = frozen.RepeatedNCA(train.NCAWindows(packed, "train", 19_975), frozen.PASSES)
        _, prefix = train.ordered(repeated, train.ORDER_SEED)
        directory = frozen.ROOT / "nca" / arm / "checkpoints"
        rows = []
        for path in directory.glob("model_*_iter.pth"):
            step = int(path.stem.split("_")[1])
            assert path.with_suffix(".json").is_file()
            rows.append({"step": step,
                         "labels": int(prefix[min(step * train.EFFECTIVE_BATCH, repeated.size)]),
                         "path": path})
        rows.sort(key=lambda row: row["step"])
        assert len(rows) == 30 and rows[0]["labels"] == 0
        assert rows[-1]["labels"] == frozen.NCA_LABELS
        # The saved 9.87M checkpoint serves 10M. No distinct 5M checkpoint exists.
        selected = {target: min(rows, key=lambda row: abs(row["labels"] - target * 1_000_000))
                    for target in TARGET_DOSES}
        assert len({selected[target]["step"] for target in TARGET_DOSES}) == len(TARGET_DOSES)
        assert selected[10]["labels"] == 9_869_186
        assert selected[25]["labels"] == 29_602_341
        assert selected[50]["labels"] == 49_347_373
        assert selected[100]["labels"] == 100_001_521
        assert selected[150]["labels"] == 148_042_896
        assert selected[200]["labels"] == 197_153_250
        result[arm] = selected
        del packed, repeated
    return reference, result


def stage_name(arm: str, target: int) -> str:
    return f"downstream/{arm}_dose_{target}m"


def stage_config(reference: dict, arm: str, target: int, checkpoint: dict) -> dict:
    config = dict(reference)
    config.update({
        "schema": "sampleeff-scale-dose-diagnostic-v1",
        "reference_seed0_config_sha256": frozen.sha(frozen.ROOT / "config.json"),
        "dose_arm": arm, "requested_nca_millions": target,
        "actual_nca_scored_labels": checkpoint["labels"],
        "nca_checkpoint_step": checkpoint["step"],
        "nca_checkpoint_sha256": frozen.sha(checkpoint["path"]),
        "first_pass_stop_tokens": PROBE_TOKENS,
        "full_downstream_budget": TEXT_TOKENS,
    })
    return config


def configure_training() -> tuple[train.TextWindows, train.TextWindows, Path, int]:
    frozen.configure_old()
    train.ROOT = ROOT
    text = train.TextWindows(frozen.ROOT / "data/tinystories/train.bin", TEXT_TOKENS)
    validation = train.TextWindows(frozen.SOURCE / "data/tinystories/val.bin", frozen.VAL_TOKENS)
    _, prefix = train.ordered(text, train.ORDER_SEED)
    stop_step = math.ceil(int(np.searchsorted(prefix, PROBE_TOKENS, side="left")) /
                          train.EFFECTIVE_BATCH)
    assert int(prefix[min(stop_step * train.EFFECTIVE_BATCH, text.size)]) == 10_000_384
    base = frozen.ROOT / "config/base_random_init.pth"
    assert base.is_file()
    return text, validation, base, stop_step


def probe(arm: str, target: int, reference: dict, checkpoint: dict,
          text: train.TextWindows, validation: train.TextWindows,
          base: Path, stop_step: int) -> None:
    name = stage_name(arm, target)
    directory = ROOT / name
    config = stage_config(reference, arm, target, checkpoint)
    marker = directory / "probe_10m_done.json"
    if marker.is_file():
        receipt = json.loads(marker.read_text())
        assert receipt["nca_checkpoint_sha256"] == config["nca_checkpoint_sha256"]
        assert receipt["downstream_tokens"] == 10_000_384
        print(f"SKIP {name}: 10M probe complete", flush=True)
        return
    print(f"START {name}: {checkpoint['labels']:,} NCA labels, seed 0, full 100M schedule",
          flush=True)
    original = train.update_step

    def pause_after_checkpoint(state, dataset, positions, physical, mask):
        if state.step >= stop_step:
            raise ProbeComplete
        return original(state, dataset, positions, physical, mask)

    train.update_step = pause_after_checkpoint
    try:
        train.train_stage(name, config, text, validation, frozen.TEXT_MODEL,
                          frozen.TEXT_MILESTONES, train.LANGUAGE_PHYSICAL_BATCH,
                          base, checkpoint["path"])
    except ProbeComplete:
        pass
    finally:
        train.update_step = original
    saved = train.latest_checkpoint(directory / "checkpoints")
    assert saved is not None and int(saved.stem.split("_")[1]) == stop_step
    history = json.loads((directory / "metrics.json").read_text())["history"]
    assert history[-1]["tokens"] == 10_000_384
    train.atomic_json(marker, {"seed": 0, "arm": arm,
        "requested_nca_millions": target,
        "nca_scored_labels": checkpoint["labels"],
        "nca_checkpoint": str(checkpoint["path"]),
        "nca_checkpoint_sha256": config["nca_checkpoint_sha256"],
        "downstream_tokens": history[-1]["tokens"],
        "last_validation_loss": history[-1]["validation_loss"],
        "downstream_checkpoint": str(saved)})
    print(f"DONE {name}: 10M probe", flush=True)
    torch.mps.empty_cache()


def choose_extensions() -> dict[str, list[int]]:
    """At most two shorter doses per arm; fixed rule, evaluated only after all probes."""
    decision = {}
    for arm in ARMS:
        anchor = json.loads((frozen.ROOT / "downstream" / arm / "metrics.json").read_text())["history"]
        anchor_loss = metrics.at(anchor, 10_000_000)
        anchor_auc = metrics.auc(anchor, 0, 10_000_000)
        candidates = []
        for target in NEW_DOSES:
            path = ROOT / stage_name(arm, target) / "metrics.json"
            history = json.loads(path.read_text())["history"]
            loss = metrics.at(history, 10_000_000)
            area = metrics.auc(history, 0, 10_000_000)
            if loss <= anchor_loss - .01 or area <= anchor_auc - .01:
                candidates.append((target, loss, area))
        by_loss = sorted(candidates, key=lambda row: (row[1], row[0]))
        by_auc = sorted(candidates, key=lambda row: (row[2], row[0]))
        selected = []
        for ranked in (by_loss, by_auc):
            if ranked and ranked[0][0] not in selected:
                selected.append(ranked[0][0])
        decision[arm] = selected
    return decision


def extend(arm: str, target: int, reference: dict, checkpoint: dict,
           text: train.TextWindows, validation: train.TextWindows, base: Path) -> None:
    name = stage_name(arm, target)
    config = stage_config(reference, arm, target, checkpoint)
    assert (ROOT / name / "probe_10m_done.json").is_file()
    print(f"EXTEND {name}: resume same seed-0 model and optimizer to 100M", flush=True)
    train.train_stage(name, config, text, validation, frozen.TEXT_MODEL,
                      frozen.TEXT_MILESTONES, train.LANGUAGE_PHYSICAL_BATCH,
                      base, checkpoint["path"])
    torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="Read-only checkpoint audit")
    parser.add_argument("--run", action="store_true", help="Run the bounded seed-0 diagnostic")
    args = parser.parse_args()
    if args.audit == args.run:
        parser.error("choose --audit or --run")
    reference, selected = audit()
    for arm in ARMS:
        print(f"{arm}: " + ", ".join(
            f"{target}M→{selected[arm][target]['labels']:,}"
            for target in TARGET_DOSES), flush=True)
    print("Reuse: Scratch seed 0, Ordinary 197.15M seed 0, Designed 197.15M seed 0.",flush=True)
    print("New: 5 shorter doses × 2 arms = 10 seed-0 downstream probes to 10M; "
          "at most 2 continuations per arm to 100M. No NCA training.",flush=True)
    if args.audit:
        return
    text, validation, base, stop_step = configure_training()
    ROOT.mkdir(parents=True, exist_ok=True)
    for target in NEW_DOSES:
        for arm in ARMS:
            probe(arm, target, reference, selected[arm][target],
                  text, validation, base, stop_step)
    decision_path = ROOT / "extension_decision.json"
    decision = choose_extensions()
    if decision_path.is_file():
        assert json.loads(decision_path.read_text())["selected"] == decision
    else:
        train.atomic_json(decision_path, {"schema": "dose-extension-decision-v1",
            "rule": "Up to one best 10M endpoint dose and one best 0–10M AUC dose per arm, "
                    "among shorter doses beating the 197.15M anchor by at least 0.01 "
                    "on either metric; ties favor lower dose.",
            "selected": decision})
    subprocess.run([sys.executable, str(Path(__file__).with_name("dose_report.py"))],
                   cwd=REPO, check=True)
    for arm in ARMS:
        for target in decision[arm]:
            extend(arm, target, reference, selected[arm][target],text,validation,base)
    subprocess.run([sys.executable, str(Path(__file__).with_name("dose_report.py"))],
                   cwd=REPO, check=True)
    print("DOSE DIAGNOSTIC COMPLETE",flush=True)


if __name__ == "__main__":
    main()
