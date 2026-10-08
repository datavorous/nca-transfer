# Downstream pretraining

`sampleeff_downstream` owns language and other downstream dataset handling, continued training, and evaluation. Its modules include `datasets`, `language`, `openwebtext`, `preprocess`, and `eval`. It uses `sampleeff_shared` model and checkpoint utilities and has no dependency on NCA generation or packing.

This package preserves the downstream code path for future experiments. The paper's downstream sample-efficiency result has not been reproduced in this workspace. See the [architecture](../../docs/ARCHITECTURE.md) and [reproduction notes](../../docs/REPRODUCTION.md).
