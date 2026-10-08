"""Descriptive, corpus-level audit of existing 1M transfer outcomes.

Run: uv run --all-packages python progress/nca_search/audit_transfer_predictors.py
No trajectory generation, model loading, or training is performed.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = ROOT / "runs/nca-data-search"
ARMS = ("high", "unfiltered", "low")
FEATURES = (
    ("horizontal_copy_excess", "Horizontal copy excess"),
    ("change", "Cell-change fraction"),
    ("patch_entropy", "Patch entropy"),
    ("lag2_recurrence", "Lag-2 recurrence"),
    ("transition_mutual_info", "Transition MI"),
    ("spatial_excess_agreement", "Spatial excess agreement"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def centered_correlation(
    blocks: list[str], feature: dict[tuple[str, str], float],
    loss: dict[tuple[str, str], float],
) -> float:
    x, y = [], []
    for block in blocks:
        fx = np.array([feature[block, arm] for arm in ARMS])
        ly = np.array([loss[block, arm] for arm in ARMS])
        x.extend(fx - fx.mean())
        y.extend(ly.mean() - ly)  # larger = better transfer
    return pearson(np.array(x), np.array(y))


def replication() -> list[str]:
    rows = read_csv(HERE / "replication_1m.csv")
    by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = row["block"], row["arm"], int(row["seed"])
        assert key not in by_key, f"duplicate replication result: {key}"
        by_key[key] = row
    blocks = [block for block in "abc" if all((block, arm, 0) in by_key for arm in ARMS)]
    assert blocks == list("abc"), "all three seed-0 triplets are required"
    loss = {(block, arm): float(by_key[block, arm, 0]["validation_loss"])
            for block in blocks for arm in ARMS}
    auc = {(block, arm): float(by_key[block, arm, 0]["auc_0_1m"])
           for block in blocks for arm in ARMS}
    feats: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for block in blocks:
        audit = json.loads((RUNS / "patch-replication" / f"block-{block}" /
                            "feature_audit.json").read_text())
        assert audit["sampled_worlds_per_arm"] == 1017
        for arm in ARMS:
            feats["gzip"][block, arm] = audit["full_gzip"][arm]["mean"]
            for key, _ in FEATURES:
                feats[key][block, arm] = audit["sampled_features"][arm][key]["mean"]

    out = [
        "# Corpus-level transfer predictor audit",
        "",
        "Reproduce: `uv run --all-packages python progress/nca_search/audit_transfer_predictors.py` ",
        "(this script rewrites only this report). Sources: `replication_1m.csv`, the three ",
        "`block-*/feature_audit.json` files, and `experiments.csv`. ",
        "Snapshot rule: use seed 0 in **every** fresh block; extra completed model seeds ",
        "are reported separately. The feature audits systematically ",
        "sample 1,017 worlds per arm; gzip is from all 20,231 selected worlds per arm.",
        "",
        "## Fresh independent rule blocks: A/B/C",
        "",
        "The 9 arm-level outcomes are three **blocks**, each with three disjoint corpora. ",
        "Repeated model seeds on a corpus and the 1,017 measured worlds are not independent ",
        "transfer observations. Each arm has the same 19,715,325 NCA labels, 1,001,472 ",
        "TinyStories tokens, and original five gzip-bin quotas. Gain is unfiltered loss minus ",
        "selected-arm loss, so positive is better.",
        "",
        "| Block | High gain | Low gain | High AUC gain | High−unfiltered copy | High−unfiltered gzip |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for block in blocks:
        out.append(f"| {block.upper()} | {loss[block,'unfiltered']-loss[block,'high']:+.5f} "
                   f"| {loss[block,'unfiltered']-loss[block,'low']:+.5f} "
                   f"| {auc[block,'unfiltered']-auc[block,'high']:+.5f} "
                   f"| {feats['horizontal_copy_excess'][block,'high']-feats['horizontal_copy_excess'][block,'unfiltered']:+.5f} "
                   f"| {feats['gzip'][block,'high']-feats['gzip'][block,'unfiltered']:+.6f} |")

    high_gains = np.array([loss[b,"unfiltered"]-loss[b,"high"] for b in blocks])
    low_gains = np.array([loss[b,"unfiltered"]-loss[b,"low"] for b in blocks])
    out += [
        "", f"Across blocks: high-copy gain {high_gains.mean():+.5f} ± "
        f"{high_gains.std(ddof=1):.5f}; low-copy gain {low_gains.mean():+.5f} ± "
        f"{low_gains.std(ddof=1):.5f} (mean ± sample SD, N=3 blocks). "
        "Leaving out A/B/C gives high-copy mean gains " + " / ".join(
            f"{np.mean([high_gains[i] for i,b in enumerate(blocks) if b != omitted]):+.5f}"
            for omitted in blocks) + ".",
        "",
        "For each feature, the correlation below centers the three arms **within each block** ",
        "then pools their 9 centered points. Positive means higher feature accompanies lower ",
        "1M loss. Leave-one-block-out values recompute that same descriptive correlation on ",
        "the two remaining blocks. They are sensitivity checks, **not** cross-validated ",
        "prediction accuracy or independent p values.",
        "",
        "| Feature | Block-centered r | Omit A | Omit B | Omit C | High−unfiltered feature shift A/B/C |",
        "|:--|--:|--:|--:|--:|:--|",
    ]
    for key, label in (*FEATURES, ("gzip", "Gzip ratio")):
        shifts = [feats[key][b, "high"] - feats[key][b, "unfiltered"] for b in blocks]
        r = centered_correlation(blocks, feats[key], loss)
        loo = [centered_correlation([b for b in blocks if b != omitted], feats[key], loss)
               for omitted in blocks]
        out.append(f"| {label} | {r:+.3f} | {loo[0]:+.3f} | {loo[1]:+.3f} "
                   f"| {loo[2]:+.3f} | {' / '.join(f'{v:+.5f}' for v in shifts)} |")

    out += ["", "### Paired direction check", "",
            "For each of the 3 blocks, compare high vs unfiltered and unfiltered vs low. ",
            "This produces 6 **dependent** directional contrasts. The count below asks whether ",
            "the arm with more of the feature also has lower loss; it is not a significance test.",
            "", "| Feature | Concordant contrasts |", "|:--|--:|"]
    for key, label in (*FEATURES, ("gzip", "Gzip ratio")):
        concordant = 0
        for block in blocks:
            for first, second in (("high", "unfiltered"), ("unfiltered", "low")):
                feature_difference = feats[key][block, first] - feats[key][block, second]
                gain = loss[block, second] - loss[block, first]
                concordant += feature_difference * gain > 0
        out.append(f"| {label} | {concordant}/6 |")

    seed_sets = {block: set.intersection(*[
        {seed for b, a, seed in by_key if b == block and a == arm} for arm in ARMS
    ]) for block in blocks}
    common_seeds = sorted(set.intersection(*seed_sets.values()))
    out += ["", "### Model-seed coverage", "",
            "Complete three-arm model seeds per block: " + ", ".join(
                f"{b.upper()}={sorted(seed_sets[b])}" for b in blocks) + ". "
            f"Common seeds across blocks: {common_seeds}. Extra seeds within one block test model "
            "initialization sensitivity there, not corpus-to-corpus uncertainty.", ""]
    for block in blocks:
        if len(seed_sets[block]) <= 1:
            continue
        gains = np.array([float(by_key[block,"unfiltered",seed]["validation_loss"])
                          - float(by_key[block,"high",seed]["validation_loss"])
                          for seed in sorted(seed_sets[block])])
        out.append(f"- Block {block.upper()}: high vs unfiltered mean paired 1M gain "
                   f"{gains.mean():+.5f}, range {gains.min():+.5f} to {gains.max():+.5f}; "
                   f"{sum(gains > 0)}/{len(gains)} model seeds win.")

    out += ["", "## Earlier exploratory selectors (iterations 01–06)", "",
            "Each row below is one selected **corpus** compared with the original 40–50 corpus. "
            "The initial corpora share rule streams and many rules, and the six selectors were "
            "chosen sequentially. The reported paired model-seed mean is an outcome summary, "
            "not six independent causal experiments. AUC is normalized 0–1M loss area; "
            "positive gain is better. These data cannot form a common multifeature "
            "correlation table because the same full-corpus features were not saved for "
            "every iteration.", "",
            "| Iteration | Selection | Model seeds | Mean endpoint gain | Mean AUC gain | "
            "Endpoint wins |", "|:--|:--|--:|--:|--:|--:|"]
    experiments = defaultdict(list)
    for row in read_csv(HERE / "experiments.csv"):
        if row["status"] == "complete":
            experiments[row["experiment"]].append(row)
    labels = {"01": "low cell change", "02": "high cell change",
              "03": "high spatial agreement", "04": "high horizontal copy",
              "05": "low horizontal copy", "06": "high transition MI"}
    for number in sorted(labels):
        rows = experiments[number]
        gains = np.array([float(row["reference_loss"])-float(row["candidate_loss"])
                          for row in rows])
        auc_gains = np.array([float(row["reference_auc_0_1m"])
                              - float(row["candidate_auc_0_1m"]) for row in rows])
        out.append(f"| {number} | {labels[number]} | {len(rows)} | {gains.mean():+.5f} "
                   f"| {auc_gains.mean():+.5f} | {sum(gains > 0)}/{len(rows)} |")

    out += ["", "## Interpretation and limits", "",
            "1. Across the three fresh blocks, seed-0 endpoint loss ranks high < unfiltered < low "
            "in every block. The high-copy selector beats its within-block unfiltered control "
            "by 0.04103–0.08426 loss units. The low-copy control loses in all three blocks. "
            "This supports the **selection policy** at the 1M checkpoint.",
            "2. Horizontal copy is a useful ranking marker here, but lag-2 recurrence and "
            "transition MI also shift with the selection in every block. All three have 6/6 "
            "directional concordance. These are engineered arm contrasts rather than nine "
            "independent random corpus draws, so the high correlations cannot identify which "
            "property causes transfer; a prospective matched-factor sampling test is required.",
            "3. Gzip means differ by less than 0.00014 between high and unfiltered arms, "
            "while the observed loss differences are much larger. Matched gzip-bin quotas "
            "limit coarse gzip confounding, but do not balance all token statistics.",
            "4. Cell change and patch entropy do not shift consistently in the same direction "
            "across blocks. The single-corpus high-change result is seed dependent; high MI "
            "and high spatial agreement lose in their one-seed exploratory probes.",
            "5. NCA validation loss was intentionally omitted from the predictor ranking. "
            "Each selected arm has its own filtered validation distribution. Those losses "
            "are comparable for training progress within an arm, but cannot serve as a "
            "shared cross-corpus representation-quality target. TinyStories validation uses "
            "the common downstream evaluation set.",
            "6. With only three fresh blocks and seed-0 outcomes used in the correlations, they are "
            "descriptive and high-variance. The selectors were discovered after earlier probes; "
            "no unadjusted p value or apparent leave-one-block stability is confirmatory.",
            "", "**Next decisions:** complete the queued A/B/C model-seed follow-up. "
            "For a distinct data target, test the frozen visible vertical lag-6 selector in "
            "`VERTICAL_LAG6_FROZEN_PROTOCOL.md`. For the horizontal-copy mechanism, later "
            "sample high/low copy arms matched on lag-2 recurrence and transition MI as "
            "scoped in `CONTROLLED_COPY_PROTOCOL.md`. The latter tests whether copy remains "
            "useful after its strongest measured co-movements are balanced.", ""]
    return out


def main() -> None:
    lines = replication()
    destination = HERE / "TRANSFER_PREDICTOR_AUDIT.md"
    destination.write_text("\n".join(lines))
    print(destination)


if __name__ == "__main__":
    main()
