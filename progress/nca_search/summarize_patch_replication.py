"""Rebuild the machine-readable 1M summary from completed replication logs.

Run: uv run --all-packages python progress/nca_search/summarize_patch_replication.py
Incomplete block/arm/seed runs are skipped, never imputed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "runs/nca-data-search/patch-replication"
OUTPUT = Path(__file__).with_name("replication_1m.csv")
ARMS = ("high", "unfiltered", "low")
FIELDS = ("block", "arm", "seed", "nca_tokens", "tinystories_tokens", "gzip_mean",
          "patch_copy_excess_mean", "validation_loss", "perplexity", "auc_0_1m")


def auc(history: list[dict]) -> float:
    return sum((b["tokens"] - a["tokens"]) * (a["validation_loss"] + b["validation_loss"]) / 2
               for a, b in zip(history, history[1:])) / history[-1]["tokens"]


def main() -> None:
    rows = []
    for block in ("a", "b", "c"):
        directory = ROOT / f"block-{block}"
        for seed in range(5):
            complete = []
            for arm in ARMS:
                arm_dir = directory / arm
                run = arm_dir / "seeds" / f"seed_{seed}"
                if not (run / "downstream/probe_1m/done.json").is_file():
                    continue
                receipt = json.loads((arm_dir / "done.json").read_text())
                assert receipt["train_usable_tokens"] == 19_715_325
                assert (run / f"nca/replication_{arm}/done.json").is_file()
                history = json.loads((run / "downstream/probe_1m/metrics.json").read_text())["history"]
                assert len(history) == 4 and history[-1]["tokens"] == 1_001_472
                complete.append((arm, history))
                rows.append({"block": block, "arm": arm, "seed": seed,
                             "nca_tokens": receipt["train_usable_tokens"],
                             "tinystories_tokens": history[-1]["tokens"],
                             "gzip_mean": receipt["gzip_ratio_mean"],
                             "patch_copy_excess_mean": receipt["patch_copy_excess_mean"],
                             "validation_loss": history[-1]["validation_loss"],
                             "perplexity": history[-1]["perplexity"],
                             "auc_0_1m": auc(history)})
            if len(complete) == 3:
                reference = [(x["step"], x["tokens"], x["learning_rate"]) for x in complete[0][1]]
                for _, history in complete[1:]:
                    assert [(x["step"], x["tokens"], x["learning_rate"]) for x in history] == reference
                value = {arm: history[-1]["validation_loss"] for arm, history in complete}
                print(f"block {block}, seed {seed}: unfiltered−high={value['unfiltered']-value['high']:+.6f}, "
                      f"low−high={value['low']-value['high']:+.6f}")
    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} completed runs to {OUTPUT}")


if __name__ == "__main__":
    main()
