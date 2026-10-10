# NCA Transfer

Generate neural cellular automata (NCA) trajectories, pretrain a small Transformer on them, and measure how that changes later language learning. This repository contains the standalone NCA generator, training code, experiment reports, and our paper, [Sampling Better Neural Cellular Automata for Pretraining](paper/Sampling%20Better%20Neural%20Cellular%20Automata%20for%20Pretraining/main.pdf).

The project builds on [Training Language Models via Neural Cellular Automata](https://arxiv.org/abs/2603.10055). Our experiments study whether selecting NCA worlds by horizontal recurrence improves transfer within a fixed gzip range.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run the small NCA example from the repository root:

```sh
uv run --locked --package sampleeff-nca python main.py
```

The [NCA package](packages/nca/README.md) can generate trajectories on its own. It also has optional controls for horizontal recurrence and neural-rule optimization. The training packages live in [`packages/`](packages/), and runnable examples are in [`examples/`](examples/README.md).

## Results and source

- [Paper PDF](paper/Sampling%20Better%20Neural%20Cellular%20Automata%20for%20Pretraining/main.pdf) and [LaTeX source](paper/Sampling%20Better%20Neural%20Cellular%20Automata%20for%20Pretraining/main.tex)
- [Complete experiment report](progress/FINAL_NCA_PRETRAINING_REPORT.md)
- [Architecture](docs/ARCHITECTURE.md), [data contract](docs/DATA_CONTRACT.md), and [reproduction notes](docs/REPRODUCTION.md)

Generated datasets and model checkpoints stay in `runs/` and are excluded from Git.
