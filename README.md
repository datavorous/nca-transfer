# SampleEff

This uv workspace separates synthetic neural cellular automata (NCA) trajectory generation from transformer pre-pre-training and downstream language training. It adapts [Training Language Models via Neural Cellular Automata](https://arxiv.org/abs/2603.10055); the source revision and known differences are in [upstream provenance](docs/UPSTREAM.md).

| Package | Python import | Responsibility |
| --- | --- | --- |
| [NCA](packages/nca) | `nca` | Configure zero-player games; generate, save, analyze, and visualize raw trajectories |
| [NCA pre-pre-training](packages/pre-pre-training) | `sampleeff_nca_pretraining` | Pack saved trajectories and train a transformer |
| [Shared](packages/shared) | `sampleeff_shared` | Transformer and checkpoint utilities |
| [Downstream pretraining](packages/downstream-pretraining) | `sampleeff_downstream` | Other datasets, continued language training, evaluation |

The artifact flow is **raw v1 trajectories → packed tokens → model checkpoint → downstream training**. NCA generation can be used on its own.

## Install and try it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this repository root:

```sh
uv sync --package sampleeff-nca-pre-pretraining
uv run --locked --package sampleeff-nca python main.py
```

The Python API is the primary interface:

```python
from nca import GameConfig, NCAGenerator, TrajectoryDataset

config = GameConfig(grid_size=4, num_colors=3, rule_count=3, frames=4)
NCAGenerator(config).save("runs/example/raw")
worlds = TrajectoryDataset.load("runs/example/raw")
print(len(worlds), worlds[0]["states"].shape)
```

The raw writer refuses to overwrite a dataset. Choose a new output path for another generation run.

## Research workflow

Follow the short scripts in [examples](examples/README.md), or run the complete tiny machinery check with a new output directory:

```sh
uv run --locked --package sampleeff-nca-pre-pretraining python examples/reproduce_tiny.py runs/my-tiny-reproduction
```

[Architecture](docs/ARCHITECTURE.md) · [Data contract](docs/DATA_CONTRACT.md) · [Upstream provenance](docs/UPSTREAM.md) · [Reproduction notes](docs/REPRODUCTION.md) · [Figure visual grammar](docs/VISUAL_GRAMMAR.md)

| Status | Result |
| --- | --- |
| NCA generation | Working |
| Raw v1 save/load and replay | Working |
| Original patch packing equivalence | Working for the tested fixture |
| Tiny transformer smoke reproduction | Working; same-world loss decreases |
| Downstream sample-efficiency reproduction | One-seed exploratory run completed; five-seed replication launched |

The tiny run checks pipeline mechanics and checkpoint reload. Its validation examples are also training examples, so it is not a held-out research result.

Experiment orchestration is in [examples/transfer_1m.py](examples/transfer_1m.py) and [examples/replicate_1m.py](examples/replicate_1m.py). Generated datasets, checkpoints, and results live under `runs/` and are excluded from Git.
