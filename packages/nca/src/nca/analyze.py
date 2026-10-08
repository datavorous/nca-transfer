"""Measurements on saved or in-memory raw NCA trajectories."""

import gzip

import numpy as np


def gzip_ratio(states: np.ndarray, num_colors: int, patch_size: int = 2) -> float:
    """Upstream-style gzip ratio of 32-bit patch IDs, with no training delimiters.

    Patch IDs are made only for this measurement. The trajectory remains a raw grid.
    """
    states = np.asarray(states)
    frames, height, width, groups = states.shape
    if height % patch_size or width % patch_size:
        raise ValueError("grid dimensions must be divisible by patch_size")
    patches = states.reshape(
        frames, height // patch_size, patch_size, width // patch_size, patch_size, groups
    ).transpose(0, 1, 3, 2, 4, 5).reshape(-1, patch_size * patch_size, groups)
    powers = num_colors ** np.arange(patch_size * patch_size, dtype=np.int64)
    tokens = (patches * powers[None, :, None]).sum(axis=1).astype(np.int32)
    raw = tokens.tobytes()
    return len(gzip.compress(raw, compresslevel=9, mtime=0)) / len(raw)


def _tail_period(states: np.ndarray) -> tuple[int | None, int | None]:
    frames = len(states)
    for start in range(frames // 2 + 1):
        for period in range(1, (frames - start) // 3 + 1):
            if np.array_equal(states[start:-period], states[start + period:]):
                return start, period
    return None, None


def measure(states: np.ndarray, num_colors: int, patch_size: int = 2) -> dict:
    """Return simple trajectory statistics without changing the trajectory."""
    states = np.asarray(states)
    if states.ndim != 4 or not len(states):
        raise ValueError("states must have shape [frames, height, width, groups]")
    counts = np.bincount(states.reshape(-1).astype(np.int64), minlength=num_colors)
    distribution = counts / counts.sum()
    nonzero = distribution[distribution > 0]
    changes = np.mean(states[1:] != states[:-1], axis=(1, 2, 3))
    period_start, period = _tail_period(states)
    return {
        "gzip_ratio_paper_style": gzip_ratio(states, num_colors, patch_size),
        "state_distribution": distribution.tolist(),
        "entropy_bits": float(-np.sum(nonzero * np.log2(nonzero))),
        "change_fraction_by_step": changes.tolist(),
        "mean_change_fraction": float(changes.mean()) if len(changes) else 0.0,
        "mean_cell_state_variance": float(np.var(states.astype(np.float32), axis=0).mean()),
        "tail_period_start": period_start,
        "tail_period": period,
        "quickly_constant": period == 1,
        "quickly_periodic": period is not None and period > 1,
    }
