"""Analyze and plot one stored trajectory; generation is not repeated."""

from pathlib import Path

from nca.analyze import measure
from nca.dataset import TrajectoryDataset
from nca.visualize import plot


raw = TrajectoryDataset.load("runs/tiny/raw")
config = raw.metadata["config"]
print("trajectories:", len(raw))
print(measure(raw[0]["states"], config["num_colors"], config["patch_size"]))
plot(raw.directory, 0, Path("runs/tiny/world.png"))
