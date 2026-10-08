"""Feature checks for a completed matched-copy block; no model execution.

Run: uv run --all-packages python progress/nca_search/screen_matched_copy_arms.py --block g
Every 20th selected world within each split and 45-cell stratum is measured.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from sample_matched_copy import ARMS, BLOCK_SEEDS, REPO
from screen_attention_lags import lag_features
from screen_existing import features
from screen_transformer_targets import patch_grid, token_features
from screen_transition_info_target import measure


STRIDE = 20
KEYS = ("change", "color_entropy", "spatial_excess_agreement", "patch_entropy",
        "rare_patch_mass", "horizontal_copy_excess", "vertical_patch_copy",
        "temporal_copy_excess", "lag38_visible_excess", "lag76_visible_excess",
        "patch_markov_gain_bits", "patch_copy_gain_bits")


def audit(block: str) -> dict:
    root = REPO / "runs/nca-data-search/matched-copy" / f"block-{block}"
    assert (root / "audit.json").is_file(), "Run the exact rule/packing audit first"
    arms: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
    marginals = {arm: np.zeros(10_000, dtype=np.int64) for arm in ARMS}
    counters = np.zeros((2, 2, 5, 3, 3), dtype=np.int64)
    for path in sorted((root / "chunks").glob("chunk-*.npz")):
        with np.load(path) as data:
            for i in range(len(data["arm"])):
                arm, split, band, lag, mi = (int(data[key][i]) for key in
                                             ("arm", "split", "bin", "lag2_bin", "mi_bin"))
                ordinal = counters[arm, split, band, lag, mi]
                counters[arm, split, band, lag, mi] += 1
                if ordinal % STRIDE:
                    continue
                states = data["states"][i]
                ratio = float(data["gzip_ratio"][i])
                arms[ARMS[arm]].append({**features(states, ratio), **token_features(states),
                                        **lag_features(states), **measure(states, ratio)})
                marginals[ARMS[arm]] += np.bincount(patch_grid(states).ravel(), minlength=10_000)
    assert len(arms["high"]) == len(arms["low"])
    sampled = {}
    for arm, rows in arms.items():
        sampled[arm] = {
            "worlds": len(rows),
            "means": {key: float(np.mean([row[key] for row in rows])) for key in KEYS},
            "patch_token_marginal": (marginals[arm] / marginals[arm].sum()).tolist(),
        }
    out = {"schema": "nca-matched-copy-feature-audit-v1", "block": block,
           "sampling": "every 20th selected world per arm/split/gzip/lag2/MI cell",
           "sampled": sampled,
           "high_minus_low": {key: sampled["high"]["means"][key] - sampled["low"]["means"][key]
                              for key in KEYS}}
    (root / "feature_audit.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=BLOCK_SEEDS, required=True)
    print(json.dumps(audit(parser.parse_args().block)["high_minus_low"], indent=2))
