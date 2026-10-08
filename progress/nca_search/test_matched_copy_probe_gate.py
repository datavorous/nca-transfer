"""Small fixture for matched-copy audit and seed-expansion gates; no models run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.argv = [sys.argv[0], "--block", "g", "--arm", "high", "--seed", "0"]
from probe_matched_copy import (FREEZE, completed_seed_zero, require_completed_blocks, sha)
from sample_matched_copy import frozen


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def must_fail(action) -> None:
    try:
        action()
    except (AssertionError, FileNotFoundError):
        return
    raise AssertionError("expected gate rejection")


def main() -> None:
    _, quota = frozen()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        must_fail(lambda: require_completed_blocks(root, FREEZE))
        for block in "ghi":
            path = root / f"block-{block}"
            (path / "rule_index.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (path / "rule_index.jsonl").write_text("fixture\n")
            write_json(path / "sampling_config.json", {"fixture": True})
            audit = {"schema": "nca-matched-copy-audit-v1", "block": block,
                     "selected_rules": 40_462, "earlier_rule_overlap": 0,
                     "quota_freeze_sha256": sha(FREEZE),
                     "rule_index_sha256": sha(path / "rule_index.jsonl"),
                     "sampling_config_sha256": sha(path / "sampling_config.json"),
                     "counts_by_arm_split_gzip_lag2_mi": [quota.tolist()] * 2,
                     "packed_sha256_by_arm": {"high": "high-fixture", "low": "low-fixture"},
                     "high_minus_low_patch_copy_excess": .02,
                     "lag2_and_mi_standardized_mean_differences": {
                         "lag2": .03, "transition_mi_bits": -.02},
                     "manipulation_check_passed": True}
            write_json(path / "audit.json", audit)
        require_completed_blocks(root, FREEZE)
        audit_path = root / "block-h/audit.json"
        audit = json.loads(audit_path.read_text())
        audit["lag2_and_mi_standardized_mean_differences"]["lag2"] = .10
        write_json(audit_path, audit)
        must_fail(lambda: require_completed_blocks(root, FREEZE))
        audit["lag2_and_mi_standardized_mean_differences"]["lag2"] = .03
        audit["high_minus_low_patch_copy_excess"] = 0
        write_json(audit_path, audit)
        must_fail(lambda: require_completed_blocks(root, FREEZE))
        audit["high_minus_low_patch_copy_excess"] = .02
        write_json(audit_path, audit)
        require_completed_blocks(root, FREEZE)
        must_fail(lambda: completed_seed_zero(root))
        for block, gain in zip("ghi", (.03, -.01, .02)):
            for arm in ("high", "low"):
                run = root / f"block-{block}" / arm / "seeds/seed_0"
                probe = run / "downstream/probe_1m"
                probe.mkdir(parents=True)
                config = {"audit_sha256": sha(root / f"block-{block}" / "audit.json"),
                          "quota_freeze_sha256": sha(FREEZE),
                          "candidate_packed_sha256": f"{arm}-fixture", "reference_seed": 0}
                write_json(run / "probe_config.json", config)
                loss = 3.0 if arm == "high" else 3.0 + gain
                write_json(probe / "done.json", {
                    "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    "tokens_consumed": 1_001_472, "final_checkpoint": "final.pth",
                    "final_validation_loss": loss})
                (probe / "final.pth").touch()
                write_json(probe / "metrics.json", {"history": [
                    {"step": i, "tokens": tokens, "learning_rate": .001,
                     "validation_loss": loss if i == 3 else 4.0}
                    for i, tokens in enumerate((0, 250_000, 500_000, 1_001_472))]})
        gains = completed_seed_zero(root)
        assert all(abs(gains[block] - expected) < 1e-12 for block, expected in
                   {"g": .03, "h": -.01, "i": .02}.items())
        done_path = root / "block-i/low/seeds/seed_0/downstream/probe_1m/done.json"
        done = json.loads(done_path.read_text())
        done["final_validation_loss"] = 2.98
        write_json(done_path, done)
        metrics_path = done_path.with_name("metrics.json")
        metrics = json.loads(metrics_path.read_text())
        metrics["history"][-1]["validation_loss"] = 2.98
        write_json(metrics_path, metrics)
        must_fail(lambda: completed_seed_zero(root))
    print("matched-copy probe gate fixtures passed")


if __name__ == "__main__":
    main()
