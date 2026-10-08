"""Freeze 45-cell matched-copy quotas from the saved 25,000 reference worlds.

Run before sampling: uv run --all-packages python progress/nca_search/freeze_matched_copy.py
The output is write-once: an existing freeze must match byte-for-byte.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from screen_patch_copy_matching import world_features
from sample_patch_replication import THRESHOLD, gzip_bin, quotas


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "runs/nca-100k-regimes"
OUT = REPO / "progress/nca_search/matched_copy_freeze.json"
LAG_CUTS = (0.2727777778, 0.3616666667)
MI_CUTS = (0.1099513937, 0.1826968063)
LIMIT = 25_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows() -> tuple[list[dict], list[Path]]:
    rows: list[dict] = []
    receipts: list[Path] = []
    for path in sorted((SOURCE / "receipts").glob("batch-*.json")):
        receipts.append(path)
        for row in json.loads(path.read_text())["accepted"]:
            if row["band"] == "40_50":
                rows.append(row)
                if len(rows) == LIMIT:
                    return rows, receipts
    raise RuntimeError(f"Only {len(rows)} saved reference worlds")


def source_bundle_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO)).encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    assert weights.shape == (3, 3) and np.all(weights > 0)
    exact = weights.ravel() / weights.sum() * total
    floor = np.floor(exact).astype(np.int64)
    extras = total - int(floor.sum())
    # Stable mergesort retains lexicographic (lag2, MI) order for exact ties.
    floor[np.argsort(-(exact - floor), kind="stable")[:extras]] += 1
    assert floor.sum() == total
    return floor.reshape(3, 3)


def build() -> dict:
    rows, receipts = source_rows()
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["dataset"]].append((index, row))
    features: list[tuple[float, float, float, float] | None] = [None] * LIMIT
    data_paths: list[Path] = []
    for name, members in sorted(grouped.items()):
        path = SOURCE / name / "trajectories-000000.npy"
        data_paths.append(path)
        worlds = np.load(path, mmap_mode="r")
        for index, row in members:
            features[index] = world_features(worlds[row["row"]], row["gzip_ratio"])
    assert all(row is not None for row in features)
    x = np.asarray(features, dtype=np.float64)
    cuts = np.quantile(x[:, 2:4], [1 / 3, 2 / 3], axis=0)
    assert np.allclose(cuts[:, 0], LAG_CUTS, atol=1e-10, rtol=0), cuts
    assert np.allclose(cuts[:, 1], MI_CUTS, atol=1e-10, rtol=0), cuts
    counts = np.zeros((5, 3, 3, 2), dtype=np.int64)
    for gzip, copy, lag, mi in x:
        band = gzip_bin(gzip)
        assert band is not None
        a = int(copy < THRESHOLD)  # high 0, low 1
        l = int(np.searchsorted(LAG_CUTS, lag, side="right"))
        m = int(np.searchsorted(MI_CUTS, mi, side="right"))
        counts[band, l, m, a] += 1
    weights = counts.min(axis=-1)
    assert np.all(weights > 0), "All 45 cells must have common support"
    old_quotas = quotas()
    assert old_quotas.tolist() == [[2913, 3544, 4034, 4559, 4925], [45, 42, 49, 57, 63]]
    target = np.stack([[largest_remainder(weights[b], int(old_quotas[s, b]))
                        for b in range(5)] for s in range(2)])
    assert target.shape == (2, 5, 3, 3)
    assert np.array_equal(target.sum(axis=(2, 3)), old_quotas)
    result = {
        "schema": "nca-matched-copy-freeze-v1",
        "block_stream_seeds": {"g": 707, "h": 808, "i": 909},
        "arm_hash_person": "nca-mcopy-arm",
        "split_hash_person": "nca-split",
        "max_attempts_per_block": 2_000_000,
        "reference_worlds": LIMIT,
        "reference_source_bundle_sha256": source_bundle_hash(receipts + data_paths),
        "reference_packed_sha256": sha(REPO / "runs/1m-nca-transfer-v2/data/nca/40_50/packed.pt"),
        "feature_code_sha256": sha(REPO / "progress/nca_search/screen_patch_copy_matching.py"),
        "copy_code_sha256": sha(REPO / "progress/nca_search/sample_patch_replication.py"),
        "threshold": THRESHOLD,
        "lag2_cuts": list(LAG_CUTS),
        "transition_mi_cuts_bits": list(MI_CUTS),
        "reference_counts_by_gzip_lag2_mi_arm": counts.tolist(),
        "minimum_counts_by_gzip_lag2_mi": weights.tolist(),
        "gzip_quotas_by_split": old_quotas.tolist(),
        "quotas_by_split_gzip_lag2_mi": target.tolist(),
        "apportionment": "minimum high/low reference count; normalized within gzip; largest remainder; lexicographic lag2,MI ties",
    }
    return result


if __name__ == "__main__":
    content = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    if OUT.exists():
        assert OUT.read_text() == content, "Existing quota freeze differs; do not overwrite"
    else:
        OUT.write_text(content)
    print(f"freeze={OUT}; sha256={sha(OUT)}")
