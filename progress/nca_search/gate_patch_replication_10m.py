"""Read-only 45-run gate for the frozen horizontal-copy 10M continuation.

Run from repo root: uv run --all-packages python progress/nca_search/gate_patch_replication_10m.py
Exit 0 only when every 1M arm is complete and STOPPING_RULE.md's block gates pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from progress.nca_search.summarize_patch_replication import auc

ROOT = REPO / "runs/nca-data-search/patch-replication"
SOURCE = REPO / "runs/1m-nca-transfer-v2"
BLOCKS = "abc"
ARMS = ("high", "unfiltered", "low")
SEEDS = range(5)
TOKENS = (0, 251_904, 501_760, 1_001_472)
STEPS = (0, 123, 245, 489)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_root(block: str, arm: str, seed: int) -> Path:
    return ROOT / f"block-{block}" / arm / "seeds" / f"seed_{seed}"


def load_results() -> dict[tuple[str, int, str], tuple[float, float]]:
    data = {}
    missing = []
    packed_hashes = {}
    for block in BLOCKS:
        for seed in SEEDS:
            reference_path = SOURCE / "seeds" / f"seed_{seed}" / "downstream/nca_40_50/metrics.json"
            reference = json.loads(reference_path.read_text())["history"][:4]
            assert [(r["step"], r["tokens"]) for r in reference] == list(zip(STEPS, TOKENS))
            for arm in ARMS:
                root = run_root(block, arm, seed)
                corpus = ROOT / f"block-{block}" / arm
                probe = root / "downstream/probe_1m"
                done_path = probe / "done.json"
                metrics_path = probe / "metrics.json"
                config_path = root / "probe_config.json"
                if not all(path.is_file() for path in (done_path, metrics_path, config_path)):
                    missing.append((block, seed, arm))
                    continue
                config = json.loads(config_path.read_text())
                done = json.loads(done_path.read_text())
                history = json.loads(metrics_path.read_text())["history"]
                assert config["reference_seed"] == seed
                assert config["name"] == f"nca-patch-replication-block-{block}-{arm}-probe"
                assert config["reference_metrics_sha256"] == sha(reference_path)
                if (block, arm) not in packed_hashes:
                    packed_hashes[block, arm] = sha(corpus / "packed.pt")
                assert config["candidate_packed_sha256"] == packed_hashes[block, arm]
                assert (root / f"nca/replication_{arm}/done.json").is_file()
                assert done["config_sha256"] == hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()).hexdigest()
                assert done["completed_steps"] == STEPS[-1]
                assert done["tokens_consumed"] == TOKENS[-1]
                assert (probe / done["final_checkpoint"]).is_file()
                assert (probe / done["final_checkpoint"]).with_suffix(".json").is_file()
                assert len(history) == 4
                assert [(r["step"], r["tokens"]) for r in history] == list(zip(STEPS, TOKENS))
                assert [r["learning_rate"] for r in history] == [
                    r["learning_rate"] for r in reference]
                assert history[-1]["validation_loss"] == done["final_validation_loss"]
                assert all(r["validation_loss"] > 0 for r in history)
                data[block, seed, arm] = (history[-1]["validation_loss"], auc(history))
    if missing:
        raise RuntimeError(f"incomplete 1M replication: {len(data)}/45 runs; missing {missing}")
    assert len(data) == 45
    return data


def gate(data: dict[tuple[str, int, str], tuple[float, float]]) -> bool:
    positive_endpoint = positive_low = positive_auc = 0
    for block in BLOCKS:
        endpoint = mean(data[block, seed, "unfiltered"][0] - data[block, seed, "high"][0]
                        for seed in SEEDS)
        low = mean(data[block, seed, "low"][0] - data[block, seed, "high"][0]
                   for seed in SEEDS)
        area = mean(data[block, seed, "unfiltered"][1] - data[block, seed, "high"][1]
                    for seed in SEEDS)
        positive_endpoint += endpoint > 0
        positive_low += low > 0
        positive_auc += area > 0
        print(f"block {block}: unfiltered-high 1M={endpoint:+.6f}; "
              f"low-high 1M={low:+.6f}; unfiltered-high AUC={area:+.6f}")
    passed = positive_endpoint == 3 and positive_low >= 2 and positive_auc >= 2
    print(f"10M continuation gate: {'PASS' if passed else 'FAIL'} "
          f"(endpoint {positive_endpoint}/3, low {positive_low}/3, AUC {positive_auc}/3)")
    return passed


def main() -> None:
    try:
        data = load_results()
        passed = gate(data)
    except (AssertionError, KeyError, RuntimeError, ValueError) as error:
        print(f"10M continuation gate: NOT READY ({error})", file=sys.stderr)
        raise SystemExit(1) from error
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
