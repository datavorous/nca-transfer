# Frozen Top-10 out-of-corpus confirmation

Frozen before any new corpus or language outcome. The question is whether the selected Top-10 horizontal-recurrence policy generalizes to **new rule populations**, beyond the block-A corpus used to choose the tier. No threshold or checkpoint selection is allowed after outcomes are read.

## Corpus units and intervention

Generate three independent rule streams with JAX seeds **8101, 8202, 8303**. Within each stream, assign each attempted rule to `designed` or `ordinary` with a domain-separated BLAKE2b hash (`person=b"nca-confirm"`) **before** rollout or feature measurement. Keep the original `nca-split` train/validation hash. A selected rule appears in only one arm. Audit selected rule IDs against the original 100K-bank receipts, every earlier NCA-search chunk containing rule IDs, and the other new blocks before training. A collision fails the block. Never reuse selected worlds from the exploratory sweep.

Both arms use the original NCA generator, 12×12 ten-color worlds, 27 frames after burn-in 10, 2×2 patch tokens, gzip [0.40, 0.50), and exactly the original five 0.02-wide train/validation quotas: train **[2913, 3544, 4034, 4559, 4925]**, validation **[45, 42, 49, 57, 63]**. Each arm has 19,975 train and 256 validation worlds, 1,026 int32 tokens per world and **19,715,325 scored training labels**. `Ordinary` accepts the first eligible worlds in stream order. `Designed` additionally requires horizontal patch-copy excess greater than or equal to the frozen Top-10 cutoff in its gzip sub-bin:

| Gzip sub-bin | .40–.42 | .42–.44 | .44–.46 | .46–.48 | .48–.50 |
|:--|--:|--:|--:|--:|--:|
| Minimum excess | .034107767225 | .025746229122 | .017753705187 | .012177637072 | .008144630498 |

The score is the rate of equal adjacent horizontal 2×2 patch IDs within patch rows, minus `sum_id n_id(n_id-1)/(972×971)` across the world. The five cutoffs are copied from the pre-outcome [Top-10 design](../boosted_nca/DESIGN.md); there is no new ranking or quantile estimation. Sampling stops when all quotas fill, with **8,000,000 attempted rules per block** as a fixed failure cap. Do not loosen the cap, bins or cutoffs based on results. Check packing against `pack_one`, score/gzip quotas, duplicate IDs and data hashes before model training.

## Frozen model benchmark

For each block and model seed **0–4**, train both arms through the full original **19,715,325-label NCA stage**, transfer the Transformer body into the saved seed-matched random language initialization (reinitialize NCA embedding and tied head), then train all parameters on the unchanged first **10,000,000 TinyStories tokens**. Use the exact original `examples/transfer_1m.py` model, optimizer, order seed 17, training seed 23, MPS device, TinyStories binaries and validation stream. Preserve evaluation milestones 0, 250K, 500K, 1M, 2M, 3M, 5M, 7.5M, 10M. Saved seed-matched scratch curves are the deterministic baseline; verify their file hashes and schedule, and do not retrain scratch for every corpus block. Run all 30 NCA/transfer chains regardless of interim outcomes. Do not interpret or tune between blocks.

## Analysis fixed in advance

Primary comparison: at 10M language tokens, report `Ordinary loss − Designed loss` for each block × seed, then average within each block and report the mean and sample SD across the **three block means**. Corpus block is the independent generation unit; five model seeds are within-block initialization variation. Report each block's seed win count and the number of positive block means. Secondary: normalized 0–10M loss AUC, final PPL, each seed's scratch-target first downward crossing (linear interpolation only), and paired `Designed speedup − Ordinary speedup` with `speedup = 10M / crossing tokens`. Report whether all conditions reach the target; do not extrapolate. Also report designed-vs-scratch and ordinary-vs-scratch separately. No new thresholds, cohorts, endpoints, targets or model branches are chosen from these results. The original selected-corpus ~2× mean remains exploratory; this independent-corpus analysis is the confirmation.
