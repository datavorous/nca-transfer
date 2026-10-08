"""Read-only full-reference freeze of the vertical lag-6 sampling target.

Run: uv run --all-packages python progress/nca_search/freeze_vertical_lag6.py
Reads saved trajectories and replication chunks; writes only a screening receipt.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from screen_vertical_lag_target import SOURCE, ROOT, features


LIMIT = 25_000
STRIDE = 20
OUT = ROOT / "progress/nca_search/vertical_lag6_freeze.json"
REPLICATION = ROOT / "runs/nca-data-search/patch-replication"


def hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def source_rows() -> tuple[list[dict], list[Path]]:
    rows = []
    paths = []
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        paths.append(path)
        for row in json.loads(path.read_text())["accepted"]:
            if row["band"] == "40_50":
                rows.append(row)
                if len(rows) == LIMIT:
                    return rows, paths
    raise RuntimeError(f"Only {len(rows)} reference 40–50 worlds")


def reference() -> tuple[list[dict[str, float]], list[Path]]:
    receipts, receipt_paths = source_rows()
    by_dataset = defaultdict(list)
    for i, row in enumerate(receipts):
        by_dataset[row["dataset"]].append((i, row))
    values: list[dict[str, float] | None] = [None] * LIMIT
    datasets = []
    for name, members in sorted(by_dataset.items()):
        path = SOURCE / name / "trajectories-000000.npy"
        datasets.append(path)
        worlds = np.load(path, mmap_mode="r")
        for index, row in members:
            values[index] = features(worlds[row["row"]], row["gzip_ratio"])
    assert all(row is not None for row in values)
    return values, receipt_paths + datasets


def replication_sample() -> dict[str, list[dict[str, float]]]:
    sampled: dict[str, list[dict[str, float]]] = defaultdict(list)
    for block in "abc":
        root = REPLICATION / f"block-{block}"
        counters = np.zeros(3, dtype=np.int64)
        for path in sorted((root / "chunks").glob("chunk-*.npz")):
            with np.load(path) as chunk:
                for i in range(len(chunk["states"])):
                    if int(chunk["split"][i]) != 0:
                        continue
                    arm = int(chunk["arm"][i])
                    if counters[arm] % STRIDE == 0:
                        key = f"block-{block}/{('high', 'unfiltered', 'low')[arm]}"
                        sampled[key].append(features(chunk["states"][i], chunk["gzip_ratio"][i]))
                    counters[arm] += 1
        assert np.array_equal(counters, [19_975] * 3), (block, counters)
    return dict(sampled)


def stats(rows: list[dict[str, float]], threshold: float) -> dict:
    keys = ("gzip", "change", "vertical_excess", "vertical_visible_excess",
            "horizontal_global_excess", "lag76_visible_excess")
    values = {key: np.array([row[key] for row in rows]) for key in keys}
    bins = np.minimum(4, ((values["gzip"] - .4) / .02).astype(int))
    high = values["vertical_excess"] >= threshold
    def group(mask: np.ndarray) -> dict:
        return {key: {"mean": float(values[key][mask].mean()),
                      "median": float(np.median(values[key][mask]))} for key in keys}
    return {
        "n": len(rows), "high_n": int(high.sum()), "low_n": int((~high).sum()),
        "overall": group(np.ones(len(rows), dtype=bool)),
        "high": group(high), "low": group(~high),
        "bins": [{"band": f"{40 + 2*b}-{42 + 2*b}", "n": int(np.sum(bins == b)),
                  "high_n": int(np.sum((bins == b) & high)),
                  "low_n": int(np.sum((bins == b) & ~high)),
                  "high_fraction": float(np.mean(high[bins == b]))}
                 for b in range(5)],
    }


def main() -> None:
    rows, paths = reference()
    scores = np.array([row["vertical_excess"] for row in rows])
    threshold = float(np.median(scores))
    result = {
        "schema": "vertical-lag6-full-reference-freeze-v1",
        "feature": "frame-frequency-adjusted vertical adjacent 2x2 patch exact-copy excess",
        "threshold": threshold,
        "reference_worlds": LIMIT,
        "reference_source_files": len(paths),
        "reference_source_bundle_sha256": hash_files(paths),
        "feature_code_sha256": hashlib.sha256(
            (ROOT / "progress/nca_search/screen_vertical_lag_target.py").read_bytes()).hexdigest(),
        "reference": stats(rows, threshold),
        "replication_samples": {
            name: stats(cohort, threshold)
            for name, cohort in replication_sample().items()},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"threshold={threshold!r}; source SHA-256={result['reference_source_bundle_sha256']}")
    for bin_ in result["reference"]["bins"]:
        print(f"{bin_['band']}%: n={bin_['n']}, high={bin_['high_n']}, low={bin_['low_n']}")
    for name, cohort in sorted(result["replication_samples"].items()):
        print(f"{name}: n={cohort['n']}, high={cohort['high_n']}, "
              f"vertical={cohort['overall']['vertical_excess']['mean']:+.6f}, "
              f"horizontal={cohort['overall']['horizontal_global_excess']['mean']:+.6f}, "
              f"lag76={cohort['overall']['lag76_visible_excess']['mean']:+.6f}")


if __name__ == "__main__":
    main()
