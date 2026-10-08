"""Configuration for a zero-player NCA game and its trajectory sample."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    grid_size: int = 12
    num_colors: int = 10
    groups: int = 1
    identity_bias: float = 0.0
    temperature: float = 1e-3
    seed: int = 0
    rule_count: int = 100
    trajectories_per_rule: int = 1
    conv_channels: int = 4
    hidden_channels: int = 16
    frames: int = 27
    stride: int = 1
    burn_in: int = 10
    batch_size: int = 64
    filter_mode: str = "none"  # none, gzip, diff
    min_complexity: float = 0.0
    max_complexity: float = 1.0
    probe_frames: int = 10
    patch_size: int = 2
    max_rule_attempts: int = 100_000

    def __post_init__(self) -> None:
        for name in (
            "grid_size", "num_colors", "groups", "rule_count", "trajectories_per_rule",
            "frames", "stride", "batch_size", "probe_frames", "patch_size", "max_rule_attempts",
            "conv_channels", "hidden_channels",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_colors < 2 or self.num_colors > 256:
            raise ValueError("num_colors must be between 2 and 256")
        if self.burn_in < 0:
            raise ValueError("burn_in cannot be negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.filter_mode not in {"none", "gzip", "diff"}:
            raise ValueError("filter_mode must be none, gzip, or diff")
        if not 0 <= self.min_complexity <= self.max_complexity:
            raise ValueError("complexity bounds must satisfy 0 <= min <= max")
        if self.filter_mode != "none" and self.min_complexity == self.max_complexity:
            raise ValueError("complexity filter needs min_complexity < max_complexity")
        if self.filter_mode == "diff" and self.probe_frames < 2:
            raise ValueError("diff filtering needs at least two probe frames")
        if self.filter_mode == "gzip" and self.grid_size % self.patch_size:
            raise ValueError("gzip filtering needs grid_size divisible by patch_size")
        if self.filter_mode == "gzip" and self.num_colors ** (self.patch_size ** 2) > 2**31 - 2:
            raise ValueError("patch tokens must fit into signed 32-bit integers for gzip filtering")
