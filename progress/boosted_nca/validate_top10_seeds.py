"""Train frozen top-10 NCA on held-out model seeds 1–4 through 10M text tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from examples import transfer_1m as v1
from sampleeff_nca_pretraining.packing import PackedDataset


SOURCE = REPO / "runs/1m-nca-transfer-v2"
BOOSTED = REPO / "runs/nca-data-search/boosted-nca"
CORPUS = BOOSTED / "sweep-a/top_10"
MILESTONES = (0, 250_000, 500_000, 1_000_000, 2_000_000,
              3_000_000, 5_000_000, 7_500_000, 10_000_000)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup(seed: int) -> tuple[Path, Path, dict]:
    decision_path = BOOSTED / "decision.json"
    decision = json.loads(decision_path.read_text())
    assert decision["schema"] == "nca-boosted-dose-decision-v1"
    assert decision["selected_tier"] == "top_10"
    assert (BOOSTED / "final-10m/done.json").is_file(), "seed-0 continuation must finish first"
    audit = json.loads((CORPUS.parent / "audit.json").read_text())
    receipt = json.loads((CORPUS / "done.json").read_text())
    packed_path = CORPUS / "packed.pt"
    assert receipt["tier"] == "top_10"
    assert receipt["train_usable_tokens"] == 19_715_325
    assert sha(packed_path) == receipt["packed_sha256"]
    assert audit["packed_sha256_by_tier"]["top_10"] == receipt["packed_sha256"]

    source_seed = SOURCE / "seeds" / f"seed_{seed}"
    base = source_seed / "config/base_random_init/base_random_init.pth"
    assert base.is_file()
    for arm in ("scratch", "nca_40_50"):
        path = source_seed / "downstream" / arm
        done = json.loads((path / "done.json").read_text())
        assert done["tokens_consumed"] == 10_000_000
        assert (path / "metrics.json").is_file()

    original = json.loads((source_seed / "config/experiment.json").read_text())
    assert original["initialization_seed"] == seed
    assert original["nca_train_worlds_per_band"] == 19_975
    root = CORPUS / "seeds" / f"seed_{seed}"
    config = {**original,
              "name": f"nca-boosted-top_10-block-a-seed-{seed}-10m-validation",
              "selected_tier": "top_10",
              "selected_corpus_sha256": sha(packed_path),
              "selected_corpus_receipt_sha256": sha(CORPUS / "done.json"),
              "corpus_audit_sha256": sha(CORPUS.parent / "audit.json"),
              "dose_decision_sha256": sha(decision_path),
              "base_initialization_sha256": sha(base),
              "validation_scope": "held-out model initialization; same selected corpus, not a fresh corpus block",
              "language_tokens": 10_000_000,
              "language_milestones": list(MILESTONES)}
    path = root / "run_config.json"
    if path.is_file():
        assert json.loads(path.read_text()) == config
    else:
        v1.atomic_json(path, config)
    return root, base, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(1, 2, 3, 4), required=True)
    seed = parser.parse_args().seed
    root, base, config = setup(seed)
    v1.ROOT = root
    v1.INITIALIZATION_SEED = seed
    v1.ORDER_SEED = 17
    v1.TRAINING_SEED = 23

    packed = PackedDataset.load(CORPUS / "packed.pt")
    training = v1.NCAWindows(packed, "train", 19_975)
    validation = v1.NCAWindows(packed, "validation", 256)
    assert int(training.valid.sum()) == 19_715_325
    assert len(validation.indices) == 256
    nca_checkpoint = v1.train_stage("nca/boosted_top_10", config, training,
                                    validation, v1.NCA_MODEL, v1.NCA_MILESTONES,
                                    v1.NCA_PHYSICAL_BATCH, base)
    del packed, training, validation

    text = v1.TextWindows(SOURCE / "data/tinystories/train.bin", 10_000_000)
    text_validation = v1.TextWindows(SOURCE / "data/tinystories/val.bin",
                                     v1.VALIDATION_TOKENS)
    v1.train_stage("downstream/full_10m", config, text, text_validation,
                   v1.TEXT_MODEL, MILESTONES, v1.LANGUAGE_PHYSICAL_BATCH,
                   base, nca_checkpoint)
    print(f"DONE top_10 seed={seed} full NCA budget and 10M TinyStories tokens", flush=True)


if __name__ == "__main__":
    main()
