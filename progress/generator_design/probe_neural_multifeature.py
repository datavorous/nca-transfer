"""Small held-out check of multi-feature targets for the original neural NCA."""

import json
from pathlib import Path

import numpy as np

from nca import GameConfig, NeuralRuleGenerator, RuleDesign, RuleSearch, optimize_rule

from compare_top10 import FIELDS, features, frozen_features, summary


def main() -> None:
    config = GameConfig(rule_count=3, trajectories_per_rule=8, batch_size=8)
    design = RuleDesign(horizontal_excess=.033, vertical_excess=.012,
                        lag2=.349, transition_mi=.229, cell_change=.728,
                        patch_entropy=7.255)
    search = RuleSearch(generations=120, population=24)
    rules = [optimize_rule(config, seed, design, search) for seed in (17, 23, 47)]
    rows = np.stack([features(state) for batch in NeuralRuleGenerator(config, rules).generate()
                     for state in batch.states])
    _, _, frozen = frozen_features()
    result = {
        "rule_seeds": [17, 23, 47], "worlds_per_rule": 8,
        "design": design.__dict__, "search": search.__dict__,
        "gzip_40_50_worlds": int(((rows[:, 0] >= .4) & (rows[:, 0] < .5)).sum()),
        "generated_features": {name: summary(rows[:, i]) for i, name in enumerate(FIELDS)},
        "frozen_sample_features": {name: summary(frozen[:, i]) for i, name in enumerate(FIELDS)},
        "generated_horizontal_by_rule": [summary(rows[i*8:(i+1)*8, 1]) for i in range(3)],
        "optimized_heldout_by_rule": [rule.receipt["after_heldout"] for rule in rules],
    }
    path = Path(__file__).with_name("neural_multifeature_probe.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
