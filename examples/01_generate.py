"""Save four small raw NCA worlds; run with the sampleeff-nca package."""

from pathlib import Path

from nca import GameConfig, NCAGenerator


config = GameConfig(grid_size=4, num_colors=3, rule_count=4,
                    trajectories_per_rule=1, frames=4, burn_in=1,
                    batch_size=2, seed=7)
output = Path("runs/tiny/raw")
NCAGenerator(config).save(output)
print(output)
