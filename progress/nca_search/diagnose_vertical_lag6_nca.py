"""Score trained NCA validation targets with a visible vertical lag-6 predecessor.

Run after the NCA checkpoint exists, from repo root:
uv run --all-packages python progress/nca_search/diagnose_vertical_lag6_nca.py --block d --arm high --seed 0
This loads model weights for inference only; it never trains or edits a checkpoint.
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from examples import transfer_1m as v1
from sampleeff_nca_pretraining.model import create_model
from sampleeff_nca_pretraining.packing import PackedDataset
from sampleeff_shared.models import create_attention_mask


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_masks(offsets: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scored patch and visible vertical target masks for 256-token windows."""
    target_positions = offsets[:, None] + 1 + torch.arange(v1.CONTEXT)[None, :]
    frame_positions = target_positions.remainder(38)
    patch = (y >= 0) & (y < 10_000) & (frame_positions >= 1) & (frame_positions <= 36)
    vertical_row = frame_positions >= 7  # 2x2 patch rows 1..5, columns 0..5.
    prior_index = target_positions - 6 - offsets[:, None]
    visible = (prior_index >= 0) & (prior_index < v1.CONTEXT)
    vertical = patch & vertical_row & visible
    rows, cols = torch.where(vertical)
    if len(rows):
        predecessor = x[rows, prior_index[rows, cols]]
        assert bool(((predecessor >= 0) & (predecessor < 10_000)).all())
    return patch, vertical


def analyze(block: str, arm: str, seed: int, checkpoint: Path | None = None) -> dict:
    root = REPO / "runs/nca-data-search/vertical-lag6" / f"block-{block}" / arm
    run = root / "seeds" / f"seed_{seed}"
    stage = run / "nca" / f"vertical_lag6_{arm}"
    receipt = json.loads((root / "done.json").read_text())
    stage_done = json.loads((stage / "done.json").read_text())
    packed_path = root / "packed.pt"
    assert receipt["packed_sha256"] == sha(packed_path)
    assert receipt["train_usable_tokens"] == 19_715_325
    checkpoint = checkpoint or stage / stage_done["final_checkpoint"]
    assert checkpoint == stage / stage_done["final_checkpoint"]
    assert checkpoint.is_file()
    output = stage / "visible_lag6_diagnostic.json"
    source_hashes = {"checkpoint_sha256": sha(checkpoint), "packed_sha256": receipt["packed_sha256"]}
    if output.is_file():
        previous = json.loads(output.read_text())
        assert all(previous[key] == value for key, value in source_hashes.items())
        return previous

    packed = PackedDataset.load(packed_path)
    validation = v1.NCAWindows(packed, "validation", v1.EVAL_NCA_WORLDS)
    assert len(validation.indices) == 256
    model = create_model(v1.NCA_MODEL)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model = model.to("mps").eval()
    attention_mask = create_attention_mask(v1.CONTEXT, additive=True).to("mps")
    totals = {key: 0.0 for key in ("all_scored", "patch", "visible_vertical")}
    counts = {key: 0 for key in totals}
    with torch.no_grad():
        for start in range(0, validation.size, v1.NCA_PHYSICAL_BATCH):
            positions = np.arange(start, min(start + v1.NCA_PHYSICAL_BATCH, validation.size))
            offsets = torch.from_numpy((positions % 4 * v1.CONTEXT).astype(np.int64))
            x, y = validation.batch(positions)
            patch, vertical = target_masks(offsets, x, y)
            logits = model(x.to("mps"), attention_mask=attention_mask)
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     y.to("mps").reshape(-1), ignore_index=-100,
                                     reduction="none").reshape_as(y)
            masks = {"all_scored": y != -100, "patch": patch, "visible_vertical": vertical}
            for key, target_mask in masks.items():
                counts[key] += int(target_mask.sum())
                totals[key] += float(losses[target_mask.to("mps")].sum().cpu())
    assert counts["all_scored"] == 256 * 987
    # The original NCA window protocol excludes frame 0's 36 patch targets.
    assert counts["patch"] == 256 * 26 * 36
    assert counts["visible_vertical"] == 256 * 765
    losses = {key: totals[key] / counts[key] for key in totals}
    assert all(math.isfinite(value) for value in losses.values())
    assert abs(losses["all_scored"] - stage_done["final_validation_loss"]) < 1e-4
    result = {"schema": "nca-visible-lag6-diagnostic-v1", "block": block, "arm": arm,
              "seed": seed, **source_hashes, "target_counts": counts,
              "validation_loss": losses,
              "visible_vertical_share_of_all_pairs": counts["visible_vertical"] / (256 * 27 * 30)}
    v1.atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("d", "e", "f"), required=True)
    parser.add_argument("--arm", choices=("high", "unfiltered", "low"), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    print(json.dumps(analyze(args.block, args.arm, args.seed), indent=2))
