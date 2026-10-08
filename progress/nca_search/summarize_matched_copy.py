"""Summarize completed matched-copy 1M probes without imputing missing runs.

Run: uv run --all-packages python progress/nca_search/summarize_matched_copy.py
The high/low corpus pair within each block is the comparison unit.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "runs/nca-data-search/matched-copy"
OUTPUT = Path(__file__).with_name("matched_copy_1m.csv")
FIELDS = ("block", "arm", "seed", "nca_tokens", "tinystories_tokens", "gzip_mean",
          "patch_copy_excess_mean", "lag2_mean", "transition_mi_bits_mean",
          "reference_validation_loss", "validation_loss", "perplexity", "auc_0_1m")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc(history: list[dict]) -> float:
    assert len(history) == 4 and history[0]["tokens"] == 0
    assert history[-1]["tokens"] == 1_001_472
    assert all(a["tokens"] < b["tokens"] for a, b in zip(history, history[1:]))
    return sum((b["tokens"] - a["tokens"]) *
               (a["validation_loss"] + b["validation_loss"]) / 2
               for a, b in zip(history, history[1:])) / history[-1]["tokens"]


def read_run(block: str, arm: str, seed: int) -> tuple[dict, list[dict]] | None:
    block_root = ROOT / f"block-{block}"
    directory = block_root / arm
    run = directory / "seeds" / f"seed_{seed}"
    probe = run / "downstream/probe_1m"
    done_path = probe / "done.json"
    if not done_path.is_file():
        return None
    audit_path = block_root / "audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["schema"] == "nca-matched-copy-audit-v1" and audit["block"] == block
    assert audit["manipulation_check_passed"]
    assert audit["high_minus_low_patch_copy_excess"] > 0
    assert all(abs(value) < .10 for value in
               audit["lag2_and_mi_standardized_mean_differences"].values())
    receipt = json.loads((directory / "done.json").read_text())
    assert receipt["schema"] == "nca-matched-copy-arm-v1"
    assert (receipt["arm"], receipt["block"]) == (arm, f"block-{block}")
    assert receipt["selected"] == 20_231 and receipt["train_usable_tokens"] == 19_715_325
    assert audit["packed_sha256_by_arm"][arm] == receipt["packed_sha256"]
    assert sha(directory / "packed.pt") == receipt["packed_sha256"]
    config = json.loads((run / "probe_config.json").read_text())
    done = json.loads(done_path.read_text())
    stage_done = json.loads((run / "nca" / f"matched_copy_{arm}" / "done.json").read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    assert done["config_sha256"] == stage_done["config_sha256"] == config_hash
    assert config["audit_sha256"] == sha(audit_path)
    assert config["candidate_packed_sha256"] == receipt["packed_sha256"]
    assert config["reference_seed"] == seed
    assert done["tokens_consumed"] == 1_001_472
    assert done["baseline_validation_loss"] == config["baseline_validation_loss"]
    assert (probe / done["final_checkpoint"]).is_file()
    history = json.loads((probe / "metrics.json").read_text())["history"]
    assert history[-1]["validation_loss"] == done["final_validation_loss"]
    row = {"block": block, "arm": arm, "seed": seed,
           "nca_tokens": receipt["train_usable_tokens"],
           "tinystories_tokens": done["tokens_consumed"],
           "gzip_mean": receipt["gzip_ratio_mean"],
           "patch_copy_excess_mean": receipt["patch_copy_excess_mean"],
           "lag2_mean": receipt["lag2_mean"],
           "transition_mi_bits_mean": receipt["transition_mi_bits_mean"],
           "reference_validation_loss": config["baseline_validation_loss"],
           "validation_loss": done["final_validation_loss"],
           "perplexity": done["final_perplexity"],
           "auc_0_1m": auc(history)}
    return row, history


def summarize() -> tuple[list[dict], dict[str, list[dict]]]:
    rows = []
    pairs: dict[str, list[dict]] = {}
    for block in ("g", "h", "i"):
        pairs[block] = []
        for seed in range(5):
            high = read_run(block, "high", seed)
            low = read_run(block, "low", seed)
            for result in (high, low):
                if result is not None:
                    rows.append(result[0])
            if high is None or low is None:
                continue
            assert [(x["step"], x["tokens"], x["learning_rate"]) for x in high[1]] == [
                (x["step"], x["tokens"], x["learning_rate"]) for x in low[1]]
            assert high[0]["reference_validation_loss"] == low[0]["reference_validation_loss"]
            pairs[block].append({"seed": seed,
                                 "loss_gain_low_minus_high": low[0]["validation_loss"] - high[0]["validation_loss"],
                                 "auc_gain_low_minus_high": low[0]["auc_0_1m"] - high[0]["auc_0_1m"]})
    return rows, pairs


def main() -> None:
    rows, pairs = summarize()
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for block, paired in pairs.items():
        if not paired:
            print(f"block {block}: no completed high/low model-seed pair")
            continue
        gain = statistics.mean(row["loss_gain_low_minus_high"] for row in paired)
        auc_gain = statistics.mean(row["auc_gain_low_minus_high"] for row in paired)
        wins = sum(row["loss_gain_low_minus_high"] > 0 for row in paired)
        print(f"block {block}: {len(paired)}/5 paired seeds, {wins} high wins, "
              f"mean 1M loss gain={gain:+.6f}, mean AUC gain={auc_gain:+.6f}")
    print(f"wrote {len(rows)} completed runs to {OUTPUT}")


if __name__ == "__main__":
    main()
