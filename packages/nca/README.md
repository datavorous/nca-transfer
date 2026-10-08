# NCA generation

`nca` is the standalone zero-player game package. `GameConfig` defines the grid, colors, stochastic rule architecture, rollout length, rule and trajectory counts, seed, and optional complexity filter. `NCAGenerator` samples rules, generates raw trajectories, and saves them without importing model training code.

```python
from nca import GameConfig, NCAGenerator, TrajectoryDataset

config = GameConfig(grid_size=4, num_colors=3, rule_count=3, frames=4)
NCAGenerator(config).save("runs/worlds/raw")
worlds = TrajectoryDataset.load("runs/worlds/raw")
print(worlds[0]["states"].shape)
```

For explicit streaming, use `nca.generate.sample_rules`, `nca.generate.generate`, and `nca.io.save_batches`. `nca.analyze.measure` reports trajectory statistics; `nca.visualize.plot` plots a saved world. `TrajectoryDataset` exposes raw NumPy arrays and the saved manifest; `nca.generate.reproduce` replays a world from its keys.

The [raw version 1 data contract](../../docs/DATA_CONTRACT.md) fixes filenames, tensor layout, dtypes, metadata, and replay expectations. [Upstream provenance](../../docs/UPSTREAM.md) records the source and known RNG differences.
