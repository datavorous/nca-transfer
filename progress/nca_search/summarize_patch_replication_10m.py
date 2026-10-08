"""Audit and print the gated A/B/C horizontal-copy 10M results.

Run: uv run --all-packages python progress/nca_search/summarize_patch_replication_10m.py
This script reads completed logs only. It exits if any of the 45 curves is missing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from progress.nca_search.gate_patch_replication_10m import (
    ARMS, BLOCKS, SEEDS, SOURCE, gate, load_results, run_root,
)


TOKENS = (0, 251_904, 501_760, 1_001_472, 2_000_896, 3_000_320,
          5_001_088, 7_501_696, 10_000_000)
STEPS = (0, 123, 245, 489, 977, 1465, 2442, 3663, 4883)


def auc(history: list[dict]) -> float:
    return sum((b["tokens"] - a["tokens"]) *
               (a["validation_loss"] + b["validation_loss"]) / 2
               for a, b in zip(history, history[1:])) / history[-1]["tokens"]


def read_curve(block: str, arm: str, seed: int) -> tuple[list[dict], float]:
    root = run_root(block, arm, seed)
    output = root / "downstream/full_10m"
    done_path = output / "done.json"
    metrics_path = output / "metrics.json"
    config_path = root / "full_config.json"
    if not all(path.is_file() for path in (done_path, metrics_path, config_path)):
        raise RuntimeError(f"10M incomplete: block={block} arm={arm} seed={seed}")
    config = json.loads(config_path.read_text())
    done = json.loads(done_path.read_text())
    history = json.loads(metrics_path.read_text())["history"]
    probe = json.loads((root / "downstream/probe_1m/metrics.json").read_text())["history"]
    reference = json.loads((SOURCE / "seeds" / f"seed_{seed}" /
                            "downstream/nca_40_50/metrics.json").read_text())["history"]
    assert config["name"] == f"nca-patch-replication-block-{block}-{arm}-full-10m"
    assert config["reference_seed"] == seed
    assert config["full_tokens"] == 10_000_000
    assert done["config_sha256"] == hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()
    assert done["completed_steps"] == STEPS[-1]
    assert done["tokens_consumed"] == TOKENS[-1]
    assert (output / done["final_checkpoint"]).is_file()
    assert (output / done["final_checkpoint"]).with_suffix(".json").is_file()
    assert len(history) == len(reference) == len(TOKENS)
    assert history[:4] == probe
    assert [(r["step"], r["tokens"]) for r in history] == list(zip(STEPS, TOKENS))
    assert [(r["step"], r["tokens"], r["learning_rate"]) for r in history] == [
        (r["step"], r["tokens"], r["learning_rate"]) for r in reference]
    assert done["final_validation_loss"] == history[-1]["validation_loss"]
    assert done["final_perplexity"] == history[-1]["perplexity"]
    return history, auc(history)


def main() -> None:
    if not gate(load_results()):
        raise RuntimeError("the prospective 45-run 1M gate failed")
    curves = {(block, seed, arm): read_curve(block, arm, seed)
              for block in BLOCKS for seed in SEEDS for arm in ARMS}
    print("# Horizontal-copy A/B/C: completed 10M continuation")
    print("\nAll 45 done receipts, complete nine-point schedules, and exact four-row "
          "probe prefixes verified. Positive gain means lower loss/AUC for high-copy.")
    print("\n| Block | Seed | High loss | Unfiltered loss | Low loss | "
          "Unfiltered−high final | Low−high final | Unfiltered−high AUC | Low−high AUC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    block_endpoint = []
    block_auc = []
    for block in BLOCKS:
        endpoint = []
        area = []
        for seed in SEEDS:
            h, ha = curves[block, seed, "high"]
            u, ua = curves[block, seed, "unfiltered"]
            l, la = curves[block, seed, "low"]
            final_gain = u[-1]["validation_loss"] - h[-1]["validation_loss"]
            auc_gain = ua - ha
            endpoint.append(final_gain)
            area.append(auc_gain)
            print(f"| {block.upper()} | {seed} | {h[-1]['validation_loss']:.5f} | "
                  f"{u[-1]['validation_loss']:.5f} | {l[-1]['validation_loss']:.5f} | "
                  f"{final_gain:+.5f} | {l[-1]['validation_loss'] - h[-1]['validation_loss']:+.5f} | "
                  f"{auc_gain:+.5f} | {la - ha:+.5f} |")
        block_endpoint.append(mean(endpoint))
        block_auc.append(mean(area))
    print("\n| Block | Mean unfiltered−high final | Mean unfiltered−high AUC |")
    print("|---|---:|---:|")
    for block, final, area in zip(BLOCKS, block_endpoint, block_auc):
        print(f"| {block.upper()} | {final:+.5f} | {area:+.5f} |")
    print(f"| Block mean ± block SD | {mean(block_endpoint):+.5f} ± {stdev(block_endpoint):.5f} | "
          f"{mean(block_auc):+.5f} ± {stdev(block_auc):.5f} |")
    early = [mean(curves[b, s, "unfiltered"][0][3]["validation_loss"] -
                  curves[b, s, "high"][0][3]["validation_loss"] for s in SEEDS)
             for b in BLOCKS]
    print(f"\nAt 1M, {sum(x > 0 for x in early)}/3 block means favor high-copy; "
          f"at 10M, {sum(x > 0 for x in block_endpoint)}/3 do. "
          f"Full-curve AUC favors high-copy in {sum(x > 0 for x in block_auc)}/3 blocks.")
    print("Three corpus blocks bound inference; model seeds within a block do not "
          "count as independent corpus replicates.")


if __name__ == "__main__":
    main()
