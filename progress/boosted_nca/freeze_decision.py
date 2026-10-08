"""Freeze the winning completed dose tier; never launches training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import rank_sweep


ROOT = rank_sweep.REPO / "runs/nca-data-search/boosted-nca"
OUT = ROOT / "decision.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    names = [name for name, path in rank_sweep.HISTORIES.items() if path.is_file()]
    assert all(name in names for name in ("top_50_anchor", "top_25", "top_10"))
    results = [rank_sweep.load(name) for name in names]
    gate, winner = rank_sweep.choose(results)
    if winner is None:
        raise RuntimeError("top-5 gate passed; finish its one allowed probe first")
    path = rank_sweep.HISTORIES[winner]
    done = json.loads((path.parent / "done.json").read_text())
    checkpoint = path.parent / done["final_checkpoint"]
    assert done["tokens_consumed"] == 1_001_472 and done["completed_steps"] == 489
    assert checkpoint.is_file() and checkpoint.with_suffix(".json").is_file()
    selected = next(result for result in results if result["name"] == winner)
    receipt = {
        "schema": "nca-boosted-dose-decision-v1", "selected_tier": winner,
        "loss_targets": list(rank_sweep.TARGETS),
        "ranking": "first downward crossing, geometric mean tokens; within 1% choose less extreme tier",
        "top_5_gate_passed": gate,
        "crossing_tokens": selected["crossings"],
        "geometric_mean_tokens": selected["score"],
        "selected_1m_validation_loss": selected["loss_1m"],
        "selected_0_1m_auc": selected["auc_0_1m"],
        "selected_probe_root": str(path.parent.relative_to(rank_sweep.REPO)),
        "selected_checkpoint": str(checkpoint.relative_to(rank_sweep.REPO)),
        "selected_checkpoint_sha256": sha(checkpoint),
        "probe_metrics_sha256_by_tier": {name: sha(rank_sweep.HISTORIES[name]) for name in names},
        "protocol_sha256": sha(rank_sweep.REPO / "progress/boosted_nca/SWEEP_PROTOCOL.md"),
    }
    if OUT.exists():
        assert json.loads(OUT.read_text()) == receipt, "decision already frozen differently"
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUT.with_name(".decision.json.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temporary.replace(OUT)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
