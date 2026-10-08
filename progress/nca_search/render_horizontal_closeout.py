"""Print the complete horizontal 1M evidence tables from verified run receipts.

Run: uv run --all-packages python progress/nca_search/render_horizontal_closeout.py
Read-only. Refuses an incomplete 45-arm matrix; does not launch continuation.
"""

from __future__ import annotations

from math import exp
from pathlib import Path
from statistics import mean, stdev
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from progress.nca_search.gate_patch_replication_10m import load_results


BLOCKS = "abc"
SEEDS = range(5)
ARMS = ("high", "unfiltered", "low")


def number(value: float) -> str:
    return f"{value:+.5f}"


def main() -> None:
    # load_results verifies the source metrics, exact four-point schedules,
    # receipt/config hashes, packed corpus hashes, and the saved checkpoints.
    data = load_results()
    print("| Block | Seed | High loss | Unfiltered loss | Low loss | "
          "U−H loss | L−U loss | L−H loss | H AUC | U AUC | L AUC |")
    print("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for block in BLOCKS:
        for seed in SEEDS:
            h, ha = data[block, seed, "high"]
            u, ua = data[block, seed, "unfiltered"]
            l, la = data[block, seed, "low"]
            print(f"| {block.upper()} | {seed} | {h:.5f} | {u:.5f} | {l:.5f} | "
                  f"{number(u-h)} | {number(l-u)} | {number(l-h)} | "
                  f"{ha:.5f} | {ua:.5f} | {la:.5f} |")

    print("\nBlock summaries below average the five model seeds within each "
          "corpus. SD in parentheses is model-seed variation within that block.")
    print("\n| Block | High loss | Unfiltered loss | Low loss | "
          "U−H loss (seed SD) | L−U loss (seed SD) | L−H loss (seed SD) | "
          "U−H AUC (seed SD) | U−H endpoint wins | L−U endpoint wins |")
    print("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    blocks: dict[str, dict[str, float]] = {}
    for block in BLOCKS:
        losses = {arm: [data[block, seed, arm][0] for seed in SEEDS] for arm in ARMS}
        gains = [losses["unfiltered"][seed] - losses["high"][seed] for seed in SEEDS]
        penalties = [losses["low"][seed] - losses["unfiltered"][seed] for seed in SEEDS]
        low_high = [losses["low"][seed] - losses["high"][seed] for seed in SEEDS]
        auc_gains = [data[block, seed, "unfiltered"][1] -
                     data[block, seed, "high"][1] for seed in SEEDS]
        blocks[block] = {"high": mean(losses["high"]),
                         "unfiltered": mean(losses["unfiltered"]),
                         "low": mean(losses["low"]),
                         "gain": mean(gains), "penalty": mean(penalties),
                         "low_high": mean(low_high),
                         "auc_gain": mean(auc_gains),
                         "ppl_gain": mean(exp(losses["unfiltered"][seed]) -
                                          exp(losses["high"][seed]) for seed in SEEDS)}
        print(f"| {block.upper()} | {mean(losses['high']):.5f} | "
              f"{mean(losses['unfiltered']):.5f} | {mean(losses['low']):.5f} | "
              f"{number(mean(gains))} ({stdev(gains):.5f}) | "
              f"{number(mean(penalties))} ({stdev(penalties):.5f}) | "
              f"{number(mean(low_high))} ({stdev(low_high):.5f}) | "
              f"{number(mean(auc_gains))} ({stdev(auc_gains):.5f}) | "
              f"{sum(x > 0 for x in gains)}/5 | "
              f"{sum(x > 0 for x in penalties)}/5 |")

    print("\nAcross-block summaries treat the three corpus means as the independent "
          "units. These are mean ± sample SD across blocks, not across 15 seeds.")
    print("\n| Measure | Mean ± block SD | Positive blocks |")
    print("|:--|--:|--:|")
    for label, key in (("High 1M loss", "high"),
                       ("Unfiltered 1M loss", "unfiltered"),
                       ("Low 1M loss", "low"),
                       ("U−H 1M loss gain", "gain"),
                       ("L−U 1M loss penalty", "penalty"),
                       ("L−H 1M loss gap", "low_high"),
                       ("U−H 0–1M AUC gain", "auc_gain"),
                       ("U−H paired perplexity reduction", "ppl_gain")):
        values = [blocks[block][key] for block in BLOCKS]
        positive = "—" if key in ARMS else f"{sum(value > 0 for value in values)}/3"
        print(f"| {label} | {mean(values):+.5f} ± {stdev(values):.5f} | "
              f"{positive} |")
    print("\nWith three independent blocks, 3/3 same-sign outcomes have "
          "two-sided exact sign p = 0.25; this is a robustness screen, not a "
          "confirmatory significance result.")


if __name__ == "__main__":
    main()
