"""Tiny public-API demonstration: `uv run --package sampleeff-nca python main.py`."""

from pathlib import Path

from nca import GameConfig, NCAGenerator, TrajectoryDataset


config = GameConfig(grid_size=4, num_colors=3, rule_count=3, frames=4, burn_in=1)
output = Path("runs/demo/raw")
if not output.exists():
    NCAGenerator(config).save(output)
worlds = TrajectoryDataset.load(output)
print(f"{len(worlds)} worlds saved at {output}; shape={worlds[0]['states'].shape}")
