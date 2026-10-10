"""Train one audited corpus block/arm/model-seed through the frozen 10M benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from examples import transfer_1m as v1
from sampleeff_nca_pretraining.packing import PackedDataset

ROOT = REPO / "runs/nca-data-search/top10-confirmation"
SOURCE = REPO / "runs/1m-nca-transfer-v2"
MILESTONES = (0, 250_000, 500_000, 1_000_000, 2_000_000,
              3_000_000, 5_000_000, 7_500_000, 10_000_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices="abc", required=True)
    parser.add_argument("--arm", choices=("designed", "ordinary"), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    audit_path = ROOT / "audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["schema"] == "nca-top10-confirmation-audit-v1"
    assert audit["selected_total"] == 3 * 2 * 20_231
    corpus = ROOT / f"block-{args.block}" / args.arm
    receipt = json.loads((corpus / "done.json").read_text())
    packed = corpus / "packed.pt"
    assert v1.digest(packed) == receipt["packed_sha256"]
    assert audit["blocks"][args.block]["packed_sha256_by_arm"][args.arm] == receipt["packed_sha256"]
    seed_dir = SOURCE / "seeds" / f"seed_{args.seed}"
    original = json.loads((seed_dir / "config/experiment.json").read_text())
    assert original["initialization_seed"] == args.seed
    assert original["nca_train_worlds_per_band"] == 19_975
    assert original["language_tokens"] == 10_000_000
    assert original["language_milestones"] == list(MILESTONES)
    scratch = json.loads((seed_dir / "downstream/scratch/done.json").read_text())
    assert scratch["tokens_consumed"] == 10_000_000
    scratch_history = json.loads((seed_dir / "downstream/scratch/metrics.json").read_text())["history"]
    assert [row["tokens"] for row in scratch_history] == [
        0, 251_904, 501_760, 1_001_472, 2_000_896, 3_000_320,
        5_001_088, 7_501_696, 10_000_000]
    base = seed_dir / "config/base_random_init/base_random_init.pth"
    assert base.is_file()
    assert v1.digest(SOURCE / "data/tinystories/train.bin") == (
        "fbed8efccdbd42cd9c803f770a6e18da40c3be37461da54106fbac2c7b140ef3")
    assert v1.digest(SOURCE / "data/tinystories/val.bin") == (
        "851de2118d56178237f2b87270d02803433dd7fb6b4b5c841bb05afb4a5385e5")
    config = {**original,
              "name": f"top10-confirmation-block-{args.block}-{args.arm}-seed-{args.seed}",
              "confirmation_protocol_sha256": audit["protocol_sha256"],
              "corpus_audit_sha256": v1.digest(audit_path),
              "corpus_packed_sha256": v1.digest(packed),
              "base_initialization_sha256": v1.digest(base),
              "scratch_metrics_sha256": v1.digest(seed_dir / "downstream/scratch/metrics.json"),
              "confirmation_role": "fresh-corpus confirmation; no threshold or outcome selection"}
    run_root = corpus / "seeds" / f"seed_{args.seed}"
    config_path = run_root / "run_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    else:
        v1.atomic_json(config_path, config)
    v1.ROOT = run_root
    v1.INITIALIZATION_SEED = args.seed
    v1.ORDER_SEED = 17
    v1.TRAINING_SEED = 23
    data = PackedDataset.load(packed)
    nca_train = v1.NCAWindows(data, "train", 19_975)
    nca_val = v1.NCAWindows(data, "validation", 256)
    assert int(nca_train.valid.sum()) == 19_715_325
    assert len(nca_val.indices) == 256
    nca_checkpoint = v1.train_stage("nca/confirmation", config, nca_train, nca_val,
                                    v1.NCA_MODEL, v1.NCA_MILESTONES,
                                    v1.NCA_PHYSICAL_BATCH, base)
    del data, nca_train, nca_val
    text = v1.TextWindows(SOURCE / "data/tinystories/train.bin", 10_000_000)
    text_val = v1.TextWindows(SOURCE / "data/tinystories/val.bin", v1.VALIDATION_TOKENS)
    v1.train_stage("downstream/full_10m", config, text, text_val, v1.TEXT_MODEL,
                   MILESTONES, v1.LANGUAGE_PHYSICAL_BATCH, base, nca_checkpoint)
    print(f"DONE block={args.block} arm={args.arm} seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
