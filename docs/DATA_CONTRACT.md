# Data contract

## Original in-memory boundary

The published generator returned a JAX categorical array with axes `[simulation, time, height, width, group]`, normally `int32` under JAX's default X64 setting. It did not serialize raw trajectories. The isolated package introduced the disk format below as **schema version 1**. That format remains unchanged.

## Raw version 1 files

| File | Dtype and shape |
| --- | --- |
| `rules.npy` | `uint32[rule_count, 2]` JAX PRNG keys |
| `trajectories-NNNNNN.npy` | `uint8[batch, frames, height, width, groups]` cell IDs |
| `rule-ids-NNNNNN.npy` | `int32[batch]` indices into `rules.npy` |
| `initial-seeds-NNNNNN.npy` | `uint32[batch, 2]` initial-condition PRNG keys |
| `initial-states-NNNNNN.npy` | `uint8[batch, height, width, groups]` pre-rollout grids |

`manifest.json` has exactly these top-level keys: `schema_version`, `config`, `implementation`, `shape`, `frame_times`, `rules`, `shards`. Each shard record has `states`, `rule_ids`, `initial_seeds`, `initial_states`, `count`. The config includes generator seed, rule count, rollout timing, network size, and filter settings. `implementation` records upstream revision and runtime/backend details. The writer does not overwrite an existing directory. A rule seed is `rules[rule_id]`; the matching initial seed and full config are enough for `nca.generate.reproduce` to replay a saved world.

Cell IDs remain raw categorical values in time and row-major grid order. No model tokens are stored in these files. NumPy `.npy` serialization, filenames, axes, dtypes, and manifest structure are frozen for version 1.

## Packed training artifact

`sampleeff_nca_pretraining.packing` converts a saved trajectory to JAX `int32` only in memory, then applies the upstream non-overlapping patch order and base-`num_colors` encoding. With a 2×2 patch, start and end grid tokens are `num_colors**4` and `num_colors**4 + 1`. It returns flattened sequences and marker-masked tokenizer targets. The training view casts tokens to PyTorch `long`, masks the first `min_grid` grids and shifts labels by one, matching the original token training path.

Optional packed `.pt` files contain `sequences`, `targets`, and `metadata`. Their metadata includes the source manifest, selected trajectory indices, source directory and manifest SHA-256, and explicit packing settings. They do not modify raw v1 files.

## Checkpoints and reproducibility limits

New `.pth` checkpoints retain the published dictionary keys (`model`, `optimizer`, `scheduler`, `epoch`, `count`, `best_val_loss`, `best_val_loss_el`, `metrics`, `total_iterations`). A neighboring JSON sidecar records model/training configurations, model seed, and source dataset metadata. An upstream checkpoint without a sidecar requires explicit config objects when loading.

Fixed-seed rollout equivalence with the published source is tested for `stride=1` when the same rule and initial keys are supplied. At `stride>1`, the existing isolated generator splits `burn_in + (frames-1)*stride + 1` step keys while upstream splits `burn_in + frames*stride`, changing the random update stream. Version 1 replay keeps the isolated behavior so previously saved worlds remain reproducible. The isolated rule sampler also draws candidate rule keys sequentially rather than in upstream batches. These are numerical limits, not file-format changes.
