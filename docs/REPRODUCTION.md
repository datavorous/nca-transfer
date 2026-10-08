# Tiny reproduction check

From the repository root, with uv installed:

```sh
uv sync --package sampleeff-nca-pre-pretraining
uv run --locked --package sampleeff-nca python -m unittest discover -s packages/nca/tests -v
uv run --locked --package sampleeff-nca-pre-pretraining python -m unittest discover -s packages/pre-pre-training/tests -v
uv run --locked --package sampleeff-nca-pre-pretraining python examples/reproduce_tiny.py runs/my-tiny-reproduction
```

Choose a new output directory for each reproduction; the raw writer refuses to overwrite an existing dataset. The example saves raw v1 shards, a packed `.pt`, a plot, a model `.pth` and JSON sidecar, plus CSV/JSON results. It verifies that checkpoint reload produces identical model output.

The October 2026 smoke run generated **3** tiny 4×4 worlds, trained a **21,875-parameter** one-layer model for **30 CPU steps**, and reduced same-world validation loss from **4.7610 to 1.7685**. All **6** boundary tests passed; the MPS test checked a forward pass, not MPS training. The validation set was deliberately the training set to check that optimization works. This is **not** a held-out result, a replication of the paper's sample-efficiency claim, or a comparison to downstream language training.

The fixture-backed tests protect fixed-seed upstream rollout (`stride=1`), gzip scoring, tokenization, raw save/load and replay, checkpoint weights, resume behavior, CPU training, and an MPS forward pass where available. See [UPSTREAM.md](UPSTREAM.md) for the known numerical limits.
