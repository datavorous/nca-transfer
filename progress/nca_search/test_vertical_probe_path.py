"""Fixture for the vertical NCA diagnostic's scored, visible target mask.

Run: uv run --all-packages python progress/nca_search/test_vertical_probe_path.py
No trajectory generation or model inference.
"""

import numpy as np
import torch

from diagnose_vertical_lag6_nca import target_masks
from examples.transfer_1m import CONTEXT, NCAWindows
from sampleeff_nca_pretraining.packing import PackedDataset


def main() -> None:
    sequence = torch.zeros((1, 1026), dtype=torch.int32)
    frames = sequence.reshape(1, 27, 38)
    frames[:, :, 0] = 10_000
    frames[:, :, -1] = 10_001
    packed = PackedDataset(sequence, sequence.clone(),
                           {"split_indices": {"validation": [0]}})
    windows = NCAWindows(packed, "validation", 1)
    x, y = windows.batch(np.arange(4))
    patch, visible = target_masks(torch.tensor([0, 256, 512, 768]), x, y)
    expected = torch.zeros_like(visible)
    for frame in range(27):
        for row in range(1, 6):
            for column in range(6):
                target = 38 * frame + 1 + 6 * row + column
                predecessor = target - 6
                if not 38 <= target <= 1024:
                    continue  # First frame has no scored patch targets.
                window = (target - 1) // CONTEXT
                if predecessor >= window * CONTEXT:
                    expected[window, target - 1 - window * CONTEXT] = True
    assert int((y != -100).sum()) == 987
    assert int(patch.sum()) == 26 * 36
    assert torch.equal(visible, expected)
    assert int(visible.sum()) == 765
    print("vertical probe mask fixture passed: 765 visible lag-6 scored targets/world")


if __name__ == "__main__":
    main()
