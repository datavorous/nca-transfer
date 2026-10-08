"""Read-only feature balance audit for one completed patch-copy block.

Run: uv run --all-packages python progress/nca_search/screen_replication_arms.py --block a
Samples every 20th selected world separately within each arm/split/gzip bin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from screen_existing import features
from screen_transformer_targets import token_features


ROOT = Path(__file__).resolve().parents[2] / "runs/nca-data-search/patch-replication"
ARMS = ("high", "unfiltered", "low")
STRIDE = 20
KEYS = ("gzip", "change", "transition_mutual_info", "color_entropy",
        "spatial_excess_agreement", "lag2_recurrence", "patch_entropy",
        "rare_patch_mass", "horizontal_copy_excess", "vertical_patch_copy",
        "temporal_copy_excess", "context_repeat_256", "late_patch_novelty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("a", "b", "c"), required=True)
    args = parser.parse_args()
    root = ROOT / f"block-{args.block}"
    rows: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
    full_gzip: dict[str, list[float]] = {arm: [] for arm in ARMS}
    counts = np.zeros((3, 2, 5), dtype=np.int64)
    for path in sorted((root / "chunks").glob("chunk-*.npz")):
        with np.load(path) as data:
            for i in range(len(data["arm"])):
                arm, split, bin_ = (int(data[key][i]) for key in ("arm", "split", "bin"))
                name = ARMS[arm]
                ratio = float(data["gzip_ratio"][i])
                full_gzip[name].append(ratio)
                ordinal = counts[arm, split, bin_]
                counts[arm, split, bin_] += 1
                if ordinal % STRIDE == 0:
                    states = data["states"][i]
                    rows[name].append({**features(states, ratio), **token_features(states)})
    assert len({len(v) for v in rows.values()}) == 1
    result = {"block": args.block, "sampled_worlds_per_arm": len(rows["high"]),
              "full_gzip": {}, "sampled_features": {}}
    print(f"block {args.block}: {len(rows['high'])} deterministic worlds per arm")
    for name in ARMS:
        gzip = np.asarray(full_gzip[name])
        result["full_gzip"][name] = {"n": len(gzip), "mean": float(gzip.mean()),
                                     "median": float(np.median(gzip))}
        result["sampled_features"][name] = {
            key: {"mean": float(np.mean([row[key] for row in rows[name]])),
                  "median": float(np.median([row[key] for row in rows[name]]))}
            for key in KEYS}
    print("property | high | unfiltered | low | high−unfiltered")
    for key in KEYS:
        values = [result["sampled_features"][arm][key]["mean"] for arm in ARMS]
        print(f"{key} | {values[0]:+.6f} | {values[1]:+.6f} | {values[2]:+.6f} | {values[0]-values[1]:+.6f}")
    path = root / "feature_audit.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
