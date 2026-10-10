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

## Horizontal recurrence as a generator knob

`HorizontalRecurrenceGenerator` keeps the neural cell rule and adds a local stochastic update: after each neural step, an aligned 2×2 patch copies its updated left neighbor with the chosen probability. This directly changes the zero-player game's dynamics. It does not select worlds by a measured recurrence score, and the probability does not guarantee an exact score for an individual world. `0` follows the original neural update exactly.

```python
from nca import GameConfig, HorizontalRecurrenceGenerator, reproduce_saved

config = GameConfig(rule_count=100, frames=27, probe_frames=27,
                    filter_mode="gzip", min_complexity=0.4, max_complexity=0.5)
generator = HorizontalRecurrenceGenerator(config, copy_probability=0.03)
generator.save("runs/worlds/patch-copy-003")
initial, states = reproduce_saved("runs/worlds/patch-copy-003", index=0)
```

Increase `copy_probability` to request more horizontal patch repetition. A small feature-only check found mean horizontal excess within the 40–50% gzip band of roughly 0.005 at probability `0`, 0.034 at `0.03`, and 0.053 at `0.05` across two rule streams. This is not evidence of improved language transfer. With gzip filtering, rule acceptance uses the designed game's probe trajectory; set `probe_frames` to the full frame count if the full-world gzip band matters. The saved `design.json` records the knob for replay. Raw v1 arrays and tokenization remain unchanged. See [the probe and limitations](../../progress/generator_design/DESIGN.md).

## Designing the original neural rules

`optimize_rule` changes the weights of the original 3×3 convolutional NCA rule. It measures **hard sampled full trajectories** against requested horizontal recurrence and gzip targets, then keeps the best weight perturbations. The search uses separate training and held-out initial-state keys. It does not change the cell update, add a patch-copy gate, or train a language model.

```python
from nca import (GameConfig, NeuralRuleGenerator, RuleDesign, RuleSearch,
                 optimize_rule, reproduce_saved)

config = GameConfig(rule_count=3, trajectories_per_rule=8)
design = RuleDesign(horizontal_excess=0.04, gzip_min=0.4, gzip_max=0.5)
search = RuleSearch(generations=100, population=16)
rules = [optimize_rule(config, seed, design, search) for seed in (17, 23, 47)]
NeuralRuleGenerator(config, rules).save("runs/worlds/neural-designed-004")
initial, states = reproduce_saved("runs/worlds/neural-designed-004", index=0)
```

The target is a **distribution-level request**, not a per-world guarantee. In a small check with three optimized rules and eight unseen rollouts per rule, requested horizontal excess `0.015`, `0.04`, and `0.07` produced means `0.017`, `0.035`, and `0.071`; 24/24, 22/24, and 21/24 worlds stayed in gzip [0.40, 0.50). Other features changed too. Read the held-out worlds in each rule's receipt before using a designed corpus. [Probe details](../../progress/generator_design/NEURAL_RULE_OPTIMIZATION.md).

[Implementation notes](../../progress/generator_design/IMPLEMENTATION_NOTES.md) describe the changes, search and replay paths, checks, and limits.
