"""Read-only feature audit for completed vertical lag-6 sampling arms.

Run: uv run --all-packages python progress/nca_search/screen_vertical_lag6_arms.py --block d
Select every twentieth world within each arm/split/gzip bin in chunk order.
Identical quotas give identical sample counts for all three arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from screen_existing import features as cell_features
from screen_transformer_targets import patch_grid, token_features
from screen_vertical_lag_target import features as vertical_features
from sample_vertical_lag6 import ARMS, BLOCK_SEEDS, REPO, THRESHOLD, quotas


STRIDE = 20
GZIP_CDF_EDGES = np.round(np.arange(.40, .50001, .005), 6)
QUANTILES = (.01, .05, .10, .25, .50, .75, .90, .95, .99)
KEYS = (
    "vertical_copy", "vertical_excess", "vertical_visible_excess",
    "vertical_visible_share", "lag76_visible_excess", "lag76_visible_share",
    "horizontal_copy_excess", "horizontal_global_excess", "change",
    "early_change", "late_change", "change_slope", "lag2_recurrence",
    "transition_mutual_info", "color_entropy", "patch_entropy",
    "effective_patch_vocab", "rare_patch_mass", "unique_patch_fraction",
    "context_repeat_256", "horizontal_bigram_repeat", "late_patch_novelty",
)


def actual_visible_features(states: np.ndarray) -> dict[str, float]:
    """Use the scored target's actual 256-token input, including its first token."""
    patches = patch_grid(states)
    frame_counts = np.stack([np.bincount(frame.ravel(), minlength=10_000)
                             for frame in patches]).astype(np.int64)
    frame_chance = (frame_counts * (frame_counts - 1)).sum(axis=1) / (36 * 35)
    positions = (np.arange(27)[:, None, None] * 38 + 1
                 + np.arange(6)[None, :, None] * 6 + np.arange(6)[None, None, :])
    positions = np.broadcast_to(positions, patches.shape)
    current = positions[:, 1:, :]
    prior = positions[:, :-1, :]
    window_start = ((current - 1) // 256) * 256
    visible = (current >= 38) & (current <= 1024) & (prior >= window_start)
    excess = ((patches[:, 1:, :] == patches[:, :-1, :])
              - np.broadcast_to(frame_chance[:, None, None], visible.shape))
    assert int(visible.sum()) == 765

    lag_current = positions[2:]
    lag_prior = positions[:-2]
    lag_start = ((lag_current - 1) // 256) * 256
    lag_visible = ((lag_current >= 38) & (lag_current <= 1024)
                   & (lag_prior >= lag_start))
    lag_chance = np.array([
        (frame_counts[f] * frame_counts[f - 2]).sum() / (36 * 36)
        for f in range(2, 27)
    ])
    lag_excess = ((patches[2:] == patches[:-2])
                  - np.broadcast_to(lag_chance[:, None, None], lag_visible.shape))
    return {
        "vertical_visible_excess": float(excess[visible].mean()),
        "vertical_visible_share": float(visible.mean()),
        "lag76_visible_excess": float(lag_excess[lag_visible].mean()),
        "lag76_visible_share": float(lag_visible.mean()),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "sd": float(array.std(ddof=1)),
            "p10": float(np.quantile(array, .1)),
            "p90": float(np.quantile(array, .9))}


def audit(block: str) -> dict:
    root = REPO / "runs/nca-data-search/vertical-lag6" / f"block-{block}"
    receipt = json.loads((root / "audit.json").read_text())
    assert receipt["schema"] == "nca-vertical-lag6-audit-v1"
    assert receipt["block"] == block
    assert receipt["earlier_rule_overlap"] == 0
    assert np.array_equal(receipt["counts_by_arm_split_bin"],
                          np.broadcast_to(quotas(), (3, 2, 5)))

    counts = np.zeros((3, 2, 5), dtype=np.int64)
    sampled = np.zeros_like(counts)
    rows: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
    gzip_values: dict[str, list[float]] = {arm: [] for arm in ARMS}
    patch_counts = {arm: np.zeros(10_000, dtype=np.int64) for arm in ARMS}
    for path in sorted((root / "chunks").glob("chunk-*.npz")):
        with np.load(path) as chunk:
            for i in range(len(chunk["arm"])):
                arm, split, bin_ = (int(chunk[key][i]) for key in ("arm", "split", "bin"))
                name = ARMS[arm]
                ratio = float(chunk["gzip_ratio"][i])
                gzip_values[name].append(ratio)
                ordinal = counts[arm, split, bin_]
                counts[arm, split, bin_] += 1
                if ordinal % STRIDE:
                    continue
                sampled[arm, split, bin_] += 1
                states = chunk["states"][i]
                vf = vertical_features(states, ratio)
                if abs(vf["vertical_excess"] - float(chunk["vertical_excess"][i])) > 1e-12:
                    raise AssertionError((path, i, "saved vertical score mismatch"))
                rows[name].append({**cell_features(states, ratio),
                                   **token_features(states), **vf,
                                   **actual_visible_features(states)})
                patch_counts[name] += np.bincount(patch_grid(states).ravel(), minlength=10_000)

    assert np.array_equal(counts, np.broadcast_to(quotas(), (3, 2, 5)))
    assert np.array_equal(sampled, (np.broadcast_to(quotas(), (3, 2, 5)) + STRIDE - 1) // STRIDE)
    assert len({len(v) for v in rows.values()}) == 1

    result = {
        "schema": "nca-vertical-lag6-feature-audit-v1", "block": block,
        "sampling": "every 20th selected world independently within arm/split/gzip bin in chunk order",
        "sampled_counts_by_arm_split_bin": sampled.tolist(),
        "sampled_worlds_per_arm": len(rows["high"]),
        "full_gzip": {}, "sampled_features": {}, "sampled_patch_frequencies": {},
        "mean_differences": {}, "manipulation_success": {},
        "source_rule_index_sha256": receipt["rule_index_sha256"],
    }
    for name in ARMS:
        gz = np.asarray(gzip_values[name], dtype=np.float64)
        assert len(gz) == 20_231 and np.all((gz >= .40) & (gz < .50))
        result["full_gzip"][name] = {
            "n": len(gz), "mean": float(gz.mean()),
            "quantiles": {str(q): float(np.quantile(gz, q)) for q in QUANTILES},
            "cdf_at_edges": {f"{edge:.3f}": float(np.mean(gz <= edge))
                             for edge in GZIP_CDF_EDGES},
        }
        result["sampled_features"][name] = {
            key: summarize([row[key] for row in rows[name]]) for key in KEYS}
        frequencies = patch_counts[name]
        top = np.argsort(-frequencies, kind="stable")[:10]
        result["sampled_patch_frequencies"][name] = {
            "total_tokens": int(frequencies.sum()),
            "observed_types": int(np.count_nonzero(frequencies)),
            "top_10": [{"patch_id": int(i), "count": int(frequencies[i]),
                        "share": float(frequencies[i] / frequencies.sum())} for i in top],
            "top_10_mass": float(frequencies[top].sum() / frequencies.sum()),
            "max_patch_share": float(frequencies.max() / frequencies.sum()),
            "tokens_from_types_with_count_at_most_2":
                float(frequencies[frequencies <= 2].sum() / frequencies.sum()),
        }
    for key in KEYS:
        high, unfiltered, low = (result["sampled_features"][arm][key]["mean"] for arm in ARMS)
        result["mean_differences"][key] = {
            "high_minus_unfiltered": high - unfiltered,
            "high_minus_low": high - low,
        }
    for key in ("vertical_excess", "vertical_visible_excess"):
        values = [result["sampled_features"][arm][key]["mean"] for arm in ARMS]
        result["manipulation_success"][key] = bool(values[0] > values[1] > values[2])
    result["manipulation_success"]["predeclared_gate"] = bool(
        all(result["manipulation_success"].values()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=BLOCK_SEEDS, required=True)
    args = parser.parse_args()
    result = audit(args.block)
    output = REPO / "runs/nca-data-search/vertical-lag6" / f"block-{args.block}" / "feature_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"{output}: {result['sampled_worlds_per_arm']} worlds/arm; "
          f"manipulation={result['manipulation_success']['predeclared_gate']}")
    print("feature | high | unfiltered | low | high-minus-unfiltered")
    for key in KEYS:
        values = [result["sampled_features"][arm][key]["mean"] for arm in ARMS]
        print(f"{key} | {values[0]:+.6f} | {values[1]:+.6f} | {values[2]:+.6f} | "
              f"{values[0] - values[1]:+.6f}")


if __name__ == "__main__":
    main()
