# Frozen vertical lag-6 sampling probe

**Status: frozen hypothesis; block D sampling and its corpus/feature audits are complete, with no English transfer probe yet.** The horizontal-copy model-seed follow-up remains active. Do not change its corpora, checkpoints, or runner. The read-only [freeze receipt](vertical_lag6_freeze.json) is reproducible with `uv run --all-packages python progress/nca_search/freeze_vertical_lag6.py`.

## Hypothesis and one sampling change

The fixed causal transformer may benefit from trajectories where a 2×2 patch often repeats directly beneath itself in the next patch row. Those tokens are six positions apart in the original raster packing, and **94.44%** of eligible pairs have a scored target with its predecessor visible in the same 256-token window. **Prediction:** selecting higher frame-frequency-adjusted vertical exact-copy excess will improve the fixed 1M TinyStories probe relative to independently sampled unfiltered and lower-excess NCA trajectories at the same gzip distribution and NCA label budget.

For each of the 27 frames, count equality across its five adjacent patch-row pairs and six columns. Subtract that frame's exact fixed-count chance, `sum_id n_id(n_id-1)/(36*35)`, from each of its 30 equality indicators; average all 810 differences. This is the `vertical_excess` function in `screen_vertical_lag_target.py`. It measures all adjacent vertical pairs, **not** only the visible subset; visible-pair excess is a separate manipulation diagnostic. Freeze the cutoff to the median of the first **25,000** accepted `40_50` worlds in sorted reference receipt order:

```text
vertical_excess >= -0.0005878894767783627 : high
vertical_excess <  -0.0005878894767783627 : low
```

The unfiltered arm ignores the vertical score. This cutoff is fixed before any vertical-arm generation or downstream result. No generator rule, NCA update, tokenizer, tensor, model, optimizer, or TinyStories setting changes. The earlier read-only screen reported 94.07% visibility because it excluded a predecessor at the first input token of a 256-token window. The actual `NCAWindows` input includes that token: an independent fixture finds **765/810** visible and scored vertical pairs per world. This corrects only the visibility diagnostic; the frozen cutoff scores all 810 pairs and is unchanged.

## Full-reference freeze and feasibility

The source is the first 25,000 accepted `40_50` receipts and their saved `trajectories-000000.npy` arrays under `runs/nca-100k-regimes/`. The freeze receipt hashes the sorted, path-prefixed receipt/data bundle as SHA-256 **`f6994c2ceb925c3372420b0ade2d266a23d97905805a8081b63abc6f621fb9f7`**. The reference packed tensor at `runs/1m-nca-transfer-v2/data/nca/40_50/packed.pt` has SHA-256 **`1ba5e8ce68c70faf19183170ac107959d5348e8db220898baa20bb9668d7ffbf`**. The scored-feature code `screen_vertical_lag_target.py` has SHA-256 **`e8fee372155bc438a493c15c4a2d9c8f019dc9158eb24fe5ee25c07a11405903`**. The standalone freeze script checks that all 25,000 source worlds exist and writes only its JSON receipt.

| Reference gzip bin | Worlds | High | Low | High fraction |
| --- | ---: | ---: | ---: | ---: |
| 40–42% | 3,640 | 1,805 | 1,835 | .496 |
| 42–44% | 4,435 | 2,142 | 2,293 | .483 |
| 44–46% | 5,013 | 2,542 | 2,471 | .507 |
| 46–48% | 5,766 | 2,830 | 2,936 | .491 |
| 48–50% | 6,146 | 3,184 | 2,962 | .518 |

The source has 12,503 high and 12,497 low worlds; the slight difference arises from ties at the median. With each bin near 50/50, both sides appear feasible for the original quotas. This is a saved-source feasibility check, not a guarantee for a fresh stream.

| Reference mean | High | Low | High − low |
| --- | ---: | ---: | ---: |
| Gzip ratio | .455743 | .454761 | +.000981 |
| Cell change | .703091 | .703426 | −.000334 |
| Vertical excess | +.009920 | −.006335 | +.016255 |
| **Visible** vertical excess | +.010044 | −.006392 | +.016436 |
| Horizontal copy excess | +.003516 | +.001940 | +.001575 |
| **Visible lag-76** excess | +.010400 | +.003415 | +.006985 |

The lag-76 shift is material: a vertical threshold also selects some temporal recurrence. Therefore a later English gain would support a **vertical-score selection policy** under this data distribution, not prove the vertical attention mechanism. Exact within-bin quotas control broad gzip-bin counts but may leave a within-bin gzip shift. Audit full gzip distributions and these nuisance properties before interpreting any probe.

The existing horizontal-copy replication arms provide a distinctness check. Every twentieth training world in each completed arm was scored (999 per arm/block). Its high-versus-low **horizontal** arm differences in mean vertical excess are +.001156 in A, +.001271 in B, and +.000349 in C, far smaller than the +.016255 high-versus-low vertical split in the reference. Across the nine arm samples, 46.4–53.0% exceed this frozen vertical cutoff. This is descriptive and does not establish independence of the two features, but it supports testing vertical structure as a different selection target. Full block/arm values are in the JSON receipt.

## Prospective controlled sampler

1. Use new JAX rule streams with seeds **404, 505, 606** for blocks D/E/F, respectively. For each attempted rule seed, assign `high`, `unfiltered`, or `low` by `blake2b(rule_seed.tobytes(), digest_size=8, person=b"nca-v6-arm")`, interpreted as a `[0,1)` fraction and split into equal thirds. This assignment and the existing `person=b"nca-split"` train/validation/test assignment happen before rollout or feature measurement. Reject and log any seed shared with the reference, earlier search corpora, another arm, or another block. Do not recycle rejected seeds.
2. Generate with the original NCA `GameConfig`, rule family, rollout, and trajectory seed derivation. Retain only gzip `[.40,.50)` and the assigned arm's frozen vertical criterion; unfiltered ignores it. Fill **exact** train bin quotas `[2913,3544,4034,4559,4925]` and validation bin quotas `[45,42,49,57,63]` in bins `[.40,.42), …, [.48,.50)`. Keep 19,975 training and 256 validation worlds per arm, 27 frames, original 2×2 packing and masks, and **19,715,325** usable training labels. Accept in attempt order, with a cap of 2,000,000 attempted rules per block. A missed quota means an incomplete block; do not relax threshold or quota after inspecting an outcome.
3. Write each new block to a fresh `runs/nca-data-search/vertical-lag6/block-{d,e,f}/` root with sampling config, source/feature/code hashes, selected rule IDs, attempts, score, gzip bin, split, tensor hash, and audit. The selected worlds live in search-internal `.npz` chunks and a `.pt` training artifact; this sampler does not produce a new raw v1 manifest or shards. Leave existing raw v1 files untouched, and preserve the packed tensor layout/dtype/tokenization/serialization. [Search packed metadata](PACKED_METADATA_COMPATIBILITY.md) documents the provenance-key difference. Audit `pack_one` agreement, markers/masks, split and rule disjointness, exact quotas, and 987 usable labels per training world before training. Do not mutate the active horizontal-copy replication files.

## Probe, decision gate, interpretation

First sample and audit block D's three arms. Report the actual full gzip CDF by arm and the following held-out diagnostics before reading English outcomes: vertical excess, *visible* vertical excess, horizontal copy excess, visible lag-76 excess, cell change, patch entropy/frequencies, and rule overlap. Predeclare manipulation success as `mean(high vertical excess) > mean(unfiltered) > mean(low)` and the same order for visible vertical excess. If it fails, report an invalid manipulation rather than transfer evidence. Report nuisance shifts regardless of outcome; the one-change sampling policy does not silently match away temporal recurrence.

Run the original NCA stage and fixed **1,001,472 TinyStories token** checkpoint for seed 0 in all three block-D arms, using the same original base initialization, 19,715,325 NCA labels, model, optimizer, TinyStories order seed 17, training seed 23, validation set, and 10M learning-rate schedule. Compare paired `val_loss(unfiltered) − val_loss(high)` as primary and `val_loss(low) − val_loss(high)` as secondary; also report perplexity and normalized 0–1M loss AUC at identical milestones. Probe NCA validation loss specifically on scored patch targets with a causally visible lag-6 predecessor, alongside all scored targets; because arms have different held-out distributions, these losses diagnose learnability rather than directly compare transfer quality.

**Triage gate:** advance to predeclared model seeds 1–4 and fresh blocks E/F only if block-D high beats **both** unfiltered and low at 1M and improves the 0–1M AUC versus unfiltered, with a successful manipulation audit. Otherwise log the result and stop this costly branch; do not retune the cutoff. For replication, the corpus block is the independent unit. Report within-block seed means and all three block contrasts; even three of three positive blocks give only two-sided exact sign `p=.25`. Continue to 10M only if the 1M pattern survives fresh blocks and model seeds. A lower NCA visible-target loss with no English gain falsifies the proposed transfer step. A downstream gain without that loss improvement calls for another mechanism; a gain with it remains associational because lag-76 and other rule properties also shift.

The active horizontal-copy replication determines whether this branch is worth the substantial compute. This protocol freezes the **next candidate** before reading any vertical-arm English result; it is not a claim that vertical lag-6 is superior to the current high-copy policy.

The prepared probe is `uv run --all-packages python progress/nca_search/probe_vertical_lag6.py --block d --arm high --seed 0` (repeat for `unfiltered` and `low`). It writes only under the selected `vertical-lag6/block-d/<arm>/seeds/seed_0/` path and resumes completed stages. The separate [NCA diagnostic](diagnose_vertical_lag6_nca.py) reloads the saved checkpoint for inference and reports loss over all scored targets, patch targets, and the exact 765 visible vertical targets per validation world; it does not change training. After completed arms, [the summarizer](summarize_vertical_lag6.py) writes `vertical_lag6_1m.csv` from completed receipts, including 0–1M AUC and the NCA diagnostic when present. The [mask fixture](test_vertical_probe_path.py) also checks a synthetic packed world; an existing saved reference validation world gave the same 987 scored, 936 patch, and 765 visible vertical targets. Block D's corpus and feature audits passed; no vertical model has run.
