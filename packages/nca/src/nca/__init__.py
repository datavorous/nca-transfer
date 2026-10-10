"""Standalone zero-player neural cellular automata tools."""

from .config import GameConfig
from .dataset import TrajectoryDataset
from .generate import NCAGenerator, Trajectory, TrajectoryBatch
from .recurrence import HorizontalRecurrenceGenerator, reproduce_saved
from .rule_design import NeuralRuleGenerator, RuleDesign, RuleSearch, optimize_rule

__all__ = ["GameConfig", "NCAGenerator", "HorizontalRecurrenceGenerator",
           "NeuralRuleGenerator", "RuleDesign", "RuleSearch", "optimize_rule",
           "Trajectory", "TrajectoryBatch", "TrajectoryDataset", "reproduce_saved"]
