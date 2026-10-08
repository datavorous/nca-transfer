"""Plot one trajectory from a saved NCA dataset."""

from pathlib import Path

import numpy as np

from .analyze import measure
from .dataset import get_trajectory, read_manifest


def plot(directory: str | Path, index: int, output: str | Path, frame_count: int = 6) -> Path:
    import matplotlib.pyplot as plt

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    record = get_trajectory(directory, index)
    config = read_manifest(directory)["config"]
    states = record["states"]
    statistics = measure(states, config["num_colors"], config["patch_size"])
    positions = np.unique(np.linspace(0, len(states) - 1, min(frame_count, len(states)), dtype=int))
    figure = plt.figure(figsize=(2.5 * len(positions), 5))
    layout = figure.add_gridspec(2, len(positions), height_ratios=[2, 1])
    palette = "tab20" if config["num_colors"] <= 20 else "viridis"
    for column, frame in enumerate(positions):
        axis = figure.add_subplot(layout[0, column])
        axis.imshow(states[frame, :, :, 0], cmap=palette, vmin=0, vmax=config["num_colors"] - 1)
        axis.set_title(f"t={config['burn_in'] + frame * config['stride']}")
        axis.axis("off")
    activity_axis = figure.add_subplot(layout[1, :])
    activity_axis.plot(np.arange(1, len(states)), statistics["change_fraction_by_step"])
    activity_axis.set_xlabel("Saved frame transition")
    activity_axis.set_ylabel("Fraction of cells changing")
    activity_axis.set_ylim(0, 1)
    figure.suptitle(
        f"World {index} · rule {record['rule_id']} · gzip {statistics['gzip_ratio_paper_style']:.3f} "
        f"· entropy {statistics['entropy_bits']:.2f} bits"
    )
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return output
