"""Summarize completed, paired patch-copy replication probes without pseudoreplication.

Run: python progress/nca_search/analyze_replication.py
Each block's interim summary uses only its completed three-arm seeds. Only model
seeds complete in every block enter the overall three-block comparison. More
model seeds refine each block, not the independent corpus-block count.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev


INPUT = Path(__file__).with_name("replication_1m.csv")
BLOCKS = ("a", "b", "c")
ARMS = ("high", "unfiltered", "low")
METRICS = ("validation_loss", "auc_0_1m")


def main() -> None:
    rows = list(csv.DictReader(INPUT.open(newline="")))
    data: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        key = row["block"], int(row["seed"]), row["arm"]
        assert row["block"] in BLOCKS and row["arm"] in ARMS and key not in data
        assert int(row["nca_tokens"]) == 19_715_325
        assert int(row["tinystories_tokens"]) == 1_001_472
        data[key] = row
    paired = {(block, seed) for block in BLOCKS for seed in range(5)
              if all((block, seed, arm) in data for arm in ARMS)}
    common_seeds = sorted(set.intersection(*(
        {seed for block_, seed in paired if block_ == block} for block in BLOCKS
    )))
    print(f"completed three-arm block/seed pairs: {len(paired)}; "
          f"model seeds common to all blocks: {common_seeds}")
    for metric in METRICS:
        print(f"\n{metric}: interim completed seeds within each block "
              "(positive gain means high-copy is better)")
        for block in BLOCKS:
            seeds = sorted(seed for block_, seed in paired if block_ == block)
            if not seeds:
                print(f"block {block}: no completed three-arm seed")
                continue
            gains = [float(data[block, seed, "unfiltered"][metric])
                     - float(data[block, seed, "high"][metric]) for seed in seeds]
            low_gains = [float(data[block, seed, "low"][metric])
                         - float(data[block, seed, "high"][metric]) for seed in seeds]
            spread = f"{stdev(gains):.6f}" if len(gains) > 1 else "n/a"
            print(f"block {block}, seeds {seeds}: unfiltered−high={mean(gains):+.6f}, "
                  f"seed SD={spread}, wins={sum(x > 0 for x in gains)}/{len(gains)}; "
                  f"low−high={mean(low_gains):+.6f}, "
                  f"wins={sum(x > 0 for x in low_gains)}/{len(low_gains)}")
    if not common_seeds:
        return
    for metric in METRICS:
        print(f"\n{metric}: balanced comparison using model seeds {common_seeds}")
        block_means = []
        for block in BLOCKS:
            gains = [float(data[block, seed, "unfiltered"][metric])
                     - float(data[block, seed, "high"][metric])
                     for seed in common_seeds]
            low_gains = [float(data[block, seed, "low"][metric])
                         - float(data[block, seed, "high"][metric])
                         for seed in common_seeds]
            block_means.append(mean(gains))
            spread = f"{stdev(gains):.6f}" if len(gains) > 1 else "n/a"
            print(f"block {block}: unfiltered−high={mean(gains):+.6f}, "
                  f"model-seed SD={spread}, wins={sum(x > 0 for x in gains)}/{len(gains)}; "
                  f"low−high={mean(low_gains):+.6f}, "
                  f"wins={sum(x > 0 for x in low_gains)}/{len(low_gains)}")
        print(f"across {len(BLOCKS)} corpus blocks: mean={mean(block_means):+.6f}, "
              f"block SD={stdev(block_means):.6f}, "
              f"positive blocks={sum(x > 0 for x in block_means)}/{len(block_means)}")
    if len(common_seeds) != 5:
        print("\nModel-seed follow-up incomplete; do not treat the current subset as final.")


if __name__ == "__main__":
    main()
