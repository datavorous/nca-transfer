"""Read-only full-curve comparison after the single winner reaches 10M."""

from __future__ import annotations

import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/nca-data-search/boosted-nca"
REFERENCE = REPO / "runs/1m-nca-transfer-v2/seeds/seed_0/downstream"
TOKENS = (0, 251_904, 501_760, 1_001_472, 2_000_896, 3_000_320,
          5_001_088, 7_501_696, 10_000_000)


def load(path: Path) -> list[dict]:
    history = json.loads(path.read_text())["history"]
    assert tuple(row["tokens"] for row in history) == TOKENS
    return history


def crossing(history: list[dict], target: float) -> float | None:
    if history[0]["validation_loss"] <= target:
        return 0.0
    for a, b in zip(history, history[1:]):
        y0, y1 = a["validation_loss"], b["validation_loss"]
        if y0 > target >= y1:
            return a["tokens"] + (b["tokens"]-a["tokens"]) * (y0-target)/(y0-y1)
    return None


def auc(history: list[dict]) -> float:
    return sum((b["tokens"]-a["tokens"]) *
               (a["validation_loss"]+b["validation_loss"]) / 2
               for a, b in zip(history, history[1:])) / TOKENS[-1]


def main() -> None:
    decision = json.loads((RUN / "decision.json").read_text())
    done = json.loads((RUN / "final-10m/done.json").read_text())
    assert done["tokens_consumed"] == 10_000_000
    scratch = load(REFERENCE / "scratch/metrics.json")
    ordinary = load(REFERENCE / "nca_40_50/metrics.json")
    designed = load(RUN / "final-10m/metrics.json")
    assert [r["learning_rate"] for r in designed] == [r["learning_rate"] for r in ordinary]
    assert designed[-1]["validation_loss"] == done["final_validation_loss"]
    print(f"Winner: {decision['selected_tier']} (seed 0, resumed from 1M)")
    print("| TinyStories tokens | Scratch loss | Ordinary NCA loss | Designed NCA loss | Designed PPL |")
    print("|--:|--:|--:|--:|--:|")
    for a,b,c in zip(scratch, ordinary, designed):
        print(f"| {a['tokens']:,} | {a['validation_loss']:.5f} | "
              f"{b['validation_loss']:.5f} | {c['validation_loss']:.5f} | "
              f"{math.exp(c['validation_loss']):.2f} |")
    print("\n| Condition | 10M loss | 10M perplexity | Normalized 0–10M AUC |")
    print("|:--|--:|--:|--:|")
    for name, history in (("Scratch", scratch), ("Ordinary NCA", ordinary),
                          ("Designed NCA", designed)):
        print(f"| {name} | {history[-1]['validation_loss']:.5f} | "
              f"{math.exp(history[-1]['validation_loss']):.2f} | {auc(history):.5f} |")
    target = scratch[-1]["validation_loss"]
    ordinary_tokens = crossing(ordinary, target)
    designed_tokens = crossing(designed, target)
    print(f"\nFixed target: scratch seed-0 loss at 10M = {target:.5f}")
    print(f"Ordinary NCA crossing: {ordinary_tokens / 1e6:.3f}M tokens; "
          f"scratch/ordinary ratio: {10_000_000 / ordinary_tokens:.3f}x" if ordinary_tokens else
          "Ordinary NCA did not reach the target by 10M")
    print(f"Designed NCA crossing: {designed_tokens / 1e6:.3f}M tokens; "
          f"scratch/designed ratio: {10_000_000 / designed_tokens:.3f}x" if designed_tokens else
          "Designed NCA did not reach the target by 10M")
    print("Crossings use first downward linear interpolation of logged milestones; "
          "ratios exclude NCA pretraining compute and are not wall-time speedups.")


if __name__ == "__main__":
    main()
