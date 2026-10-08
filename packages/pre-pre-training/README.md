# NCA pre-pre-training

`sampleeff_nca_pretraining` reads saved `nca` trajectories, applies the original non-overlapping patch tokenization, and stores a separate packed PyTorch dataset. It configures and trains the transformer using `sampleeff_shared` model and checkpoint utilities. Generation remains in `nca`.

Use `pack_dataset` and `PackedDataset` from `sampleeff_nca_pretraining.packing`; `ModelConfig` and `create_model` from `.model`; and `TrainingConfig`, `prepare`, `train`, `save_checkpoint`, and `load_checkpoint` from `.training`.

The [stage examples](../../examples/README.md) show generation through reload and resume. The [data contract](../../docs/DATA_CONTRACT.md) describes raw and packed artifacts. A tiny upstream fixture protects fixed-seed rollout, gzip scoring, and tokenization equivalence.
