"""Generate fresh worlds from optimized original neural rules; no model training."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from nca import (GameConfig, NeuralRuleGenerator, RuleDesign, RuleSearch,
                 TrajectoryDataset, optimize_rule, reproduce_saved)

from compare_top10 import FIELDS, features, summary

ROOT = Path(__file__).resolve().parent
SEEDS = (17, 23, 47)
TARGETS = (("015", .015), ("040", .04), ("070", .07))


def main() -> None:
    results = {"rule_seeds": list(SEEDS), "worlds_per_rule": 8, "targets": {}}
    with TemporaryDirectory() as temporary:
        for tag, target in TARGETS:
            config = GameConfig(rule_count=len(SEEDS), trajectories_per_rule=8,
                                batch_size=8)
            rules = [optimize_rule(config, seed, RuleDesign(target), RuleSearch())
                     for seed in SEEDS]
            directory = NeuralRuleGenerator(config, rules).save(
                Path(temporary) / f"target-{tag}")
            dataset = TrajectoryDataset.load(directory)
            records = list(dataset)
            for index in (0, len(dataset) - 1):
                initial, states = reproduce_saved(directory, index)
                np.testing.assert_array_equal(initial, records[index]["initial_state"])
                np.testing.assert_array_equal(states, records[index]["states"])
            values = np.stack([features(row["states"]) for row in records])
            results["targets"][tag] = {
                "requested_horizontal_excess": target,
                "worlds": len(values),
                "gzip_40_50_worlds": int(((values[:, 0] >= .4) & (values[:, 0] < .5)).sum()),
                "features": {name: summary(values[:, i]) for i, name in enumerate(FIELDS)},
                "horizontal_by_rule": [summary(values[j*8:(j+1)*8, 1])
                                       for j in range(len(rules))],
            }
    path = ROOT / "neural_api_validation.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
