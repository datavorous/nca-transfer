# Upstream provenance

- Paper: [Training Language Models via Neural Cellular Automata](https://arxiv.org/abs/2603.10055)
- Source: [danihyunlee/nca-pre-pretraining](https://github.com/danihyunlee/nca-pre-pretraining)
- Adapted revision: `bdd1c71e04eb2e5a2c91481faf278e95846b4431`
- Source license: MIT; see the repository `LICENSE`.

The active NCA package adapts the discrete Flax network, periodic 3×3 neighborhood, stochastic update, initial-state draw, rollout ordering, and gzip complexity scoring from upstream `utils/nca.py`. The packing code preserves upstream `utils/tokenizers.py` patch order, integer encoding, grid markers, and target masking. The shared model and checkpoint utilities adapt upstream `utils/models.py` and `utils/util.py`.

Upstream generated trajectories in memory; this project introduced the documented raw v1 `.npy` and `manifest.json` format. Training now consumes saved and packed trajectories rather than generating worlds inside the optimizer loop. A tiny JSON fixture in `packages/pre-pre-training/tests/fixtures/upstream_tiny.json` was generated from the published implementation at the revision above; tests compare fixed-seed rollout, gzip ratio, and tokens against it.

Known numerical differences: rule candidate keys are sampled sequentially here rather than in upstream batches, and `stride>1` uses a different number of split update keys. Fixed-seed rollout equivalence is established for `stride=1`. See [DATA_CONTRACT.md](DATA_CONTRACT.md) for exact formats and replay expectations.
