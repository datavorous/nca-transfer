# Tiny research workflow

Run the stages separately from the repository root. Each file exposes its
settings at the top and consumes the previous stage's saved artifact.

```sh
uv run --locked --package sampleeff-nca python examples/01_generate.py
uv run --locked --package sampleeff-nca python examples/02_load_analyze.py
uv run --locked --package sampleeff-nca-pre-pretraining python examples/03_pack.py
uv run --locked --package sampleeff-nca-pre-pretraining python examples/04_pretrain.py
uv run --locked --package sampleeff-nca-pre-pretraining python examples/05_reload_resume.py
```

These use `runs/tiny/`. Choose a new output directory or remove that small
experimental run before repeating generation; the v1 writer refuses to
overwrite an existing raw dataset.

For one complete smoke check, use a new output directory:

```sh
uv run --locked --package sampleeff-nca-pre-pretraining python examples/reproduce_tiny.py runs/my-tiny-reproduction
```

`reproduce_tiny.py` saves every intermediate artifact and verifies that its
same-world validation loss decreases, then checks identical model output after
checkpoint reload. The same-world check tests optimization and serialization;
it is not evidence of generalization to unseen NCA rules.
