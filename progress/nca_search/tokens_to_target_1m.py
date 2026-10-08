"""Read-only, predeclared tokens-to-loss analysis for horizontal-copy probes.

Run: python progress/nca_search/tokens_to_target_1m.py [--require-complete] [--json]

The target grid was fixed on 2026-10-08 with 35/45 probes complete, before the
remaining block-C model seeds were observed. The five levels cover the common
0–1M validation-loss range; they were not selected for a favorable speedup.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


REPO = Path(__file__).resolve().parents[2]
INPUT = Path(__file__).with_name("replication_1m.csv")
RUNS = REPO / "runs/nca-data-search/patch-replication"
BLOCKS = ("a", "b", "c")
ARMS = ("high", "unfiltered", "low")
TARGETS = (10.0, 9.0, 8.0, 7.0, 6.25)
TOKENS = (0, 251_904, 501_760, 1_001_472)


def crossing(history: list[dict], target: float) -> float | None:
    """First downward crossing on saved evaluation points, without extrapolation."""
    points = [(int(row["tokens"]), float(row["validation_loss"])) for row in history]
    assert [token for token, _ in points] == list(TOKENS), points
    if points[0][1] <= target:
        return 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 > target >= y1:
            return x0 + (x1 - x0) * (y0 - target) / (y0 - y1)
    return None


def load() -> dict[tuple[str, int, str], dict]:
    data: dict[tuple[str, int, str], dict] = {}
    for row in csv.DictReader(INPUT.open(newline="")):
        block, seed, arm = row["block"], int(row["seed"]), row["arm"]
        key = block, seed, arm
        assert block in BLOCKS and seed in range(5) and arm in ARMS and key not in data
        directory = RUNS / f"block-{block}/{arm}/seeds/seed_{seed}/downstream/probe_1m"
        assert (directory / "done.json").is_file(), key
        history = json.loads((directory / "metrics.json").read_text())["history"]
        assert abs(float(history[-1]["validation_loss"]) - float(row["validation_loss"])) < 1e-10, key
        assert int(row["tinystories_tokens"]) == TOKENS[-1]
        data[key] = {"history": history, "crossings": {str(target): crossing(history, target)
                                                       for target in TARGETS}}
    return data


def analyze(data: dict, require_complete: bool) -> dict:
    if require_complete:
        expected = {(block, seed, arm) for block in BLOCKS for seed in range(5)
                    for arm in ARMS}
        assert set(data) == expected, f"Incomplete: {len(data)}/45 completed arms"
    rows = []
    for block in BLOCKS:
        for seed in range(5):
            if (block, seed, "high") not in data or (block, seed, "unfiltered") not in data:
                continue
            for target in TARGETS:
                high = data[block, seed, "high"]["crossings"][str(target)]
                unfiltered = data[block, seed, "unfiltered"]["crossings"][str(target)]
                low = (data[block, seed, "low"]["crossings"][str(target)]
                       if (block, seed, "low") in data else None)
                rows.append({"block": block, "seed": seed, "target_loss": target,
                             "high_tokens": high, "unfiltered_tokens": unfiltered,
                             "low_tokens": low,
                             "unfiltered_over_high": unfiltered / high
                             if high is not None and high > 0 and unfiltered is not None else None})
    summary = []
    for target in TARGETS:
        blocks = []
        for block in BLOCKS:
            selected = [row["unfiltered_over_high"] for row in rows
                        if row["target_loss"] == target and row["block"] == block
                        and row["unfiltered_over_high"] is not None]
            blocks.append({"block": block, "paired_seeds": len(selected),
                           "mean_ratio": mean(selected) if selected else None,
                           "model_seed_sd": stdev(selected) if len(selected) > 1 else None})
        valid = [block["mean_ratio"] for block in blocks if block["mean_ratio"] is not None]
        summary.append({"target_loss": target, "blocks": blocks,
                        "mean_block_ratio": mean(valid) if valid else None,
                        "block_sd": stdev(valid) if len(valid) > 1 else None,
                        "n_blocks": len(valid)})
    return {"targets_predeclared": list(TARGETS), "crossing_method": "first downward linear interpolation",
            "unit_of_replication": "corpus block; model seeds averaged within block",
            "rows": rows, "summary": summary}


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def fmt_tokens(value: float | None) -> str:
    return "NA" if value is None else f"{value / 1000:.0f}k"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze(load(), args.require_complete)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("Targets predeclared: " + ", ".join(map(str, TARGETS)))
    print("First downward linear crossing; no extrapolation. NA means a target was not reached.")
    print("Only four validation checkpoints are available; crossings are approximate interpolations.")
    print("Ratio = unfiltered TinyStories tokens / high-copy TinyStories tokens; >1 favors high.")
    print("\n| Block | Seed | Target loss | High tokens | Unfiltered tokens | Low tokens | Ratio |")
    print("|:--|--:|--:|--:|--:|--:|--:|")
    for row in result["rows"]:
        print(f"| {row['block'].upper()} | {row['seed']} | {row['target_loss']:g} | "
              f"{fmt_tokens(row['high_tokens'])} | {fmt_tokens(row['unfiltered_tokens'])} | "
              f"{fmt_tokens(row['low_tokens'])} | {fmt(row['unfiltered_over_high'])} |")
    print("\n| Target loss | A mean ± seed SD (n) | B mean ± seed SD (n) | "
          "C mean ± seed SD (n) | Mean ± block SD |")
    print("|--:|--:|--:|--:|--:|")
    for summary in result["summary"]:
        blocks = [f"{fmt(b['mean_ratio'])} ± {fmt(b['model_seed_sd'])} "
                  f"({b['paired_seeds']})" for b in summary["blocks"]]
        print(f"| {summary['target_loss']:g} | " + " | ".join(blocks) + " | "
              f"{fmt(summary['mean_block_ratio'])} ± {fmt(summary['block_sd'])} "
              f"({summary['n_blocks']} blocks) |")


if __name__ == "__main__":
    main()
