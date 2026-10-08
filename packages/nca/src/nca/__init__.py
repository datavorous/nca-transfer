"""Standalone zero-player neural cellular automata tools."""

from .config import GameConfig
from .dataset import TrajectoryDataset
from .generate import NCAGenerator, Trajectory, TrajectoryBatch

__all__ = ["GameConfig", "NCAGenerator", "Trajectory", "TrajectoryBatch", "TrajectoryDataset"]
