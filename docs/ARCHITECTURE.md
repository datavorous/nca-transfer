# Architecture

The uv workspace root is virtual. Implementation lives in four packages:

| Package | Owns | Depends on |
| --- | --- | --- |
| `packages/nca` (`nca`) | NCA rules, rollout, raw v1 storage, analysis, visualization | JAX, Flax, NumPy, Matplotlib |
| `packages/pre-pre-training` (`sampleeff_nca_pretraining`) | Patch packing, packed datasets, NCA model configuration, training and checkpoint API | `nca`, `sampleeff_shared` |
| `packages/shared` (`sampleeff_shared`) | Transformer architecture, causal masks, scheduler and checkpoint helpers | PyTorch, Transformers |
| `packages/downstream-pretraining` (`sampleeff_downstream`) | Other datasets, continued language training, and evaluation | `sampleeff_shared` |

The intended flow is:

```text
GameConfig → raw trajectory → v1 .npy dataset → metrics/plot
                                           ↓
                                     packed .pt tokens
                                           ↓
                              model + training config → .pth checkpoint
                                           ↓
                              downstream model weights and other datasets
```

Each arrow is a Python API boundary. NCA does not import model code. Packing reads raw trajectories but does not generate them. Training accepts prepared token pairs and does not know how NCA rules were sampled. Downstream code depends on shared model and checkpoint utilities, without importing NCA.

The small scripts in `examples/` show each boundary separately. `main.py` demonstrates only generation and loading. The published upstream implementation has been removed; a deterministic tiny fixture in `packages/pre-pre-training/tests/fixtures/` protects its relevant behavior.
