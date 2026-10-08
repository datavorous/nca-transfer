"""Small contract checks; no NCA trajectory generation or model training.

Run: uv run --all-packages python progress/nca_search/test_vertical_lag6.py
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from nca import GameConfig

from sample_vertical_lag6 import (ARMS, BLOCK_SEEDS, REPO, THRESHOLD,
                                  arm_for, gzip_bin, pack_arm, quotas, sha, split_for,
                                  vertical_excess)
from screen_vertical_lag_target import features


def main() -> None:
    freeze_path = REPO / "progress/nca_search/vertical_lag6_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    assert freeze["threshold"] == THRESHOLD
    assert sha(REPO / "progress/nca_search/screen_vertical_lag_target.py") == freeze["feature_code_sha256"]
    assert BLOCK_SEEDS == {"d": 404, "e": 505, "f": 606}
    assert ARMS == ("high", "unfiltered", "low")
    assert quotas().tolist() == [[2913, 3544, 4034, 4559, 4925],
                                 [45, 42, 49, 57, 63]]
    assert [gzip_bin(x) for x in (.4, .419, .439, .459, .479, .499, .5)] == [0, 0, 1, 2, 3, 4, None]

    rng = np.random.default_rng(6081)
    worlds = [np.zeros((27, 12, 12, 1), dtype=np.uint8),
              rng.integers(0, 10, size=(27, 12, 12, 1), dtype=np.uint8)]
    # Vary rows while retaining exact vertical patch copies in each frame.
    repeated_row = rng.integers(0, 10, size=(27, 2, 12, 1), dtype=np.uint8)
    worlds.append(np.tile(repeated_row, (1, 6, 1, 1)))
    for states in worlds:
        score = vertical_excess(states)
        assert np.isclose(score, features(states, .45)["vertical_excess"], rtol=0, atol=1e-15)
    assert vertical_excess(worlds[0]) == 0.0
    assert vertical_excess(worlds[2]) > vertical_excess(worlds[1])
    for seed in ((0, 0), (1, 2), (123456789, 987654321)):
        key = np.array(seed, dtype=np.uint32)
        assert arm_for(key) in (0, 1, 2)
        assert split_for(key) in (0, 1, 2)
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / "block-d"
        directory = root / "high"
        directory.mkdir(parents=True)
        packed = directory / "packed.pt"
        packed.write_bytes(b"completed artifact")
        (directory / "done.json").write_text(json.dumps({
            "schema": "nca-vertical-lag6-arm-v1", "arm": "high", "block": "block-d",
            "attempts": 128, "gzip_bin_counts": quotas().tolist(),
            "packed_sha256": sha(packed),
        }))
        pack_arm(root, 0, quotas(), 128, GameConfig())
        assert packed.read_bytes() == b"completed artifact"
    print("vertical lag-6 sampler fixture checks passed")


if __name__ == "__main__":
    main()
