"""Small static fixtures for the frozen matched-copy sampler; no generation."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

import jax
import numpy as np

from sample_matched_copy import (ARMS, BLOCK_SEEDS, BATCH, FREEZE, THRESHOLD,
                                 arm_for, cell_bins, frozen, next_keys, replay,
                                 save_chunk, split_for, world_features)
from freeze_matched_copy import largest_remainder


def main() -> None:
    freeze, quotas = frozen()
    assert FREEZE.is_file() and ARMS == ("high", "low")
    assert quotas.shape == (2, 5, 3, 3)
    assert np.array_equal(quotas.sum(axis=(2, 3)),
                          [[2913, 3544, 4034, 4559, 4925], [45, 42, 49, 57, 63]])
    assert np.array_equal(largest_remainder(np.ones((3, 3)), 5),
                          [[1, 1, 1], [1, 1, 0], [0, 0, 0]])
    for seed in BLOCK_SEEDS.values():
        rules = np.asarray(jax.random.split(jax.random.PRNGKey(seed), 64))
        assert set(map(arm_for, rules)) == {0, 1}
        assert set(map(split_for, rules)) <= {0, 1, 2}
    # Read an existing immutable reference world to check feature binning.
    from freeze_matched_copy import source_rows, SOURCE
    row = source_rows()[0][0]
    states = np.load(SOURCE / row["dataset"] / "trajectories-000000.npy", mmap_mode="r")[row["row"]]
    _, copy, lag, mi = world_features(states, row["gzip_ratio"])
    assert np.isfinite([copy, lag, mi]).all()
    l, m = cell_bins(lag, mi, freeze)
    assert 0 <= l < 3 and 0 <= m < 3
    with TemporaryDirectory() as temporary:
        chunks = Path(temporary)
        cursor, rules = next_keys(jax.random.PRNGKey(707))
        save_chunk(chunks, 0, BATCH, cursor, [], 0)
        replayed_cursor, attempt, counts = replay(chunks, quotas, 707)
        assert attempt == BATCH and np.array_equal(np.asarray(cursor), np.asarray(replayed_cursor))
        assert not counts.any() and rules.shape == (BATCH, 2)
    print("matched-copy fixtures passed")


if __name__ == "__main__":
    main()
