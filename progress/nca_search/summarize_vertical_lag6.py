"""Rebuild the machine-readable 1M summary from completed vertical lag-6 logs.

Run: uv run --all-packages python progress/nca_search/summarize_vertical_lag6.py
Incomplete block/arm/seed runs are skipped, never imputed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "runs/nca-data-search/vertical-lag6"
OUTPUT = Path(__file__).with_name("vertical_lag6_1m.csv")
ARMS = ("high", "unfiltered", "low")
FIELDS = ("block", "arm", "seed", "nca_tokens", "tinystories_tokens", "gzip_mean",
          "vertical_excess_mean", "reference_validation_loss", "validation_loss",
          "perplexity", "auc_0_1m", "nca_validation_loss_all_scored",
          "nca_validation_loss_patch", "nca_validation_loss_visible_lag6",
          "nca_visible_lag6_target_count")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc(history: list[dict]) -> float:
    return sum((b["tokens"] - a["tokens"]) * (a["validation_loss"] + b["validation_loss"]) / 2
               for a, b in zip(history, history[1:])) / history[-1]["tokens"]


def main() -> None:
    rows = []
    for block in ("d", "e", "f"):
        directory = ROOT / f"block-{block}"
        audit_path = directory / "audit.json"
        for seed in range(5):
            complete = []
            for arm in ARMS:
                arm_dir = directory / arm
                run = arm_dir / "seeds" / f"seed_{seed}"
                done_path = run / "downstream/probe_1m/done.json"
                if not done_path.is_file():
                    continue
                receipt = json.loads((arm_dir / "done.json").read_text())
                assert receipt["schema"] == "nca-vertical-lag6-arm-v1"
                assert receipt["train_usable_tokens"] == 19_715_325
                audit = json.loads(audit_path.read_text())
                assert audit["schema"] == "nca-vertical-lag6-audit-v1"
                assert audit["block"] == block
                assert audit["packed_sha256_by_arm"][arm] == receipt["packed_sha256"]
                assert sha(arm_dir / "packed.pt") == receipt["packed_sha256"]
                stage = run / "nca" / f"vertical_lag6_{arm}"
                stage_done = json.loads((stage / "done.json").read_text())
                config = json.loads((run / "probe_config.json").read_text())
                done = json.loads(done_path.read_text())
                config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
                assert done["config_sha256"] == stage_done["config_sha256"] == config_hash
                assert config["audit_sha256"] == sha(audit_path)
                assert done["tokens_consumed"] == 1_001_472
                assert done["baseline_validation_loss"] == config["baseline_validation_loss"]
                assert (run / "downstream/probe_1m" / done["final_checkpoint"]).is_file()
                history = json.loads((run / "downstream/probe_1m/metrics.json").read_text())["history"]
                assert len(history) == 4 and history[-1]["tokens"] == 1_001_472
                assert history[-1]["validation_loss"] == done["final_validation_loss"]
                diagnostic_path = stage / "visible_lag6_diagnostic.json"
                diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.is_file() else None
                if diagnostic:
                    assert diagnostic["schema"] == "nca-visible-lag6-diagnostic-v1"
                    assert (diagnostic["block"], diagnostic["arm"], diagnostic["seed"]) == (block, arm, seed)
                    assert diagnostic["packed_sha256"] == receipt["packed_sha256"]
                    assert diagnostic["checkpoint_sha256"] == sha(stage / stage_done["final_checkpoint"])
                    assert diagnostic["target_counts"] == {
                        "all_scored": 256 * 987,
                        "patch": 256 * 26 * 36,
                        "visible_vertical": 256 * 765,
                    }
                complete.append((arm, history))
                rows.append({"block": block, "arm": arm, "seed": seed,
                             "nca_tokens": receipt["train_usable_tokens"],
                             "tinystories_tokens": history[-1]["tokens"],
                             "gzip_mean": receipt["gzip_ratio_mean"],
                             "vertical_excess_mean": receipt["vertical_excess_mean"],
                             "reference_validation_loss": config["baseline_validation_loss"],
                             "validation_loss": history[-1]["validation_loss"],
                             "perplexity": history[-1]["perplexity"],
                             "auc_0_1m": auc(history),
                             "nca_validation_loss_all_scored": diagnostic["validation_loss"]["all_scored"] if diagnostic else None,
                             "nca_validation_loss_patch": diagnostic["validation_loss"]["patch"] if diagnostic else None,
                             "nca_validation_loss_visible_lag6": diagnostic["validation_loss"]["visible_vertical"] if diagnostic else None,
                             "nca_visible_lag6_target_count": diagnostic["target_counts"]["visible_vertical"] if diagnostic else None})
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
