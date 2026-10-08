"""Read-only summary of the completed block-D 500K TinyStories screen."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "runs/nca-data-search/vertical-lag6/block-d"
ARMS = ("high", "unfiltered", "low")


def load(arm: str) -> tuple[list[float], float, float]:
    directory = ROOT / arm / "seeds/seed_0/downstream/probe_1m"
    rows = json.loads((directory / "metrics.json").read_text())["history"]
    assert [row["tokens"] for row in rows] == [0, 251_904, 501_760], arm
    early = json.loads((directory / "early_100k.json").read_text())
    assert early["tokens"] == 100_352, arm
    checkpoint = directory / "checkpoints/model_245_iter.pth"
    assert checkpoint.is_file() and checkpoint.with_suffix(".json").is_file(), arm
    losses = [float(row["validation_loss"]) for row in rows]
    auc = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in
              zip(zip((0, 251_904), losses), zip((251_904, 501_760), losses[1:]))) / 501_760
    return losses, float(early["validation_loss"]), auc


def main() -> None:
    data = {arm: load(arm) for arm in ARMS}
    print("| Arm | 0 loss | 100K loss | 250K loss | 500K loss | 0–500K AUC |")
    print("|:--|--:|--:|--:|--:|--:|")
    for arm, (losses, early, auc) in data.items():
        print(f"| {arm} | {losses[0]:.5f} | {early:.5f} | "
              f"{losses[1]:.5f} | {losses[2]:.5f} | {auc:.5f} |")
    high, unfiltered, low = (data[arm] for arm in ARMS)
    print("\n| Contrast (positive favors high) | 250K loss | 500K loss | 0–500K AUC |")
    print("|:--|--:|--:|--:|")
    for name, other in (("Unfiltered − high", unfiltered), ("Low − high", low)):
        print(f"| {name} | {other[0][1]-high[0][1]:+.5f} | "
              f"{other[0][2]-high[0][2]:+.5f} | {other[2]-high[2]:+.5f} |")
    gate = (unfiltered[0][2]-high[0][2] >= .03
            and low[0][2] > high[0][2]
            and unfiltered[0][1] > high[0][1]
            and unfiltered[2] > high[2])
    print(f"\nPredeclared short-screen gate: {'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
