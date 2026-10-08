# Cell-transition information in the causal patch stream

## Read-only screen

`uv run --all-packages python progress/nca_search/screen_transition_info_target.py --first 25000 --stride 1`

Source: all 25,000 accepted saved 40–50 gzip worlds, in receipt order, from `runs/nca-100k-regimes/`. The script reads trajectories and receipts only. It does not generate a corpus or run a model.

The original packed stream has 27 frames × (`START`, 36 raster-order 2×2 patches, `END`), or 1,026 positions. A prior-frame patch at the same site is exactly 38 positions back. For each world, the script fits a 10×10 cell transition table on frame transitions 0→1 through 12→13, with 0.5 pseudocount per transition. It scores target patches in frames 14–26 only where both target and lag-38 source lie in the same nonoverlapping 256-token training window and the target is scored. This leaves **414 visible patch targets per world**. The four cell probabilities are multiplied to give a normalized factorized patch predictor. Its held-out loss is compared with a factorized color-unigram predictor fitted on the same early transitions. A separate, normalized exact-patch-copy mixture is fitted on early frames and scored on the same visible targets. Gains are baseline loss minus predictor loss, in bits per patch. This is a diagnostic predictor; it is not the transformer or its training loss.

| Property across worlds | Mean | Median | 10th–90th percentile |
| --- | ---: | ---: | ---: |
| Full-trajectory cell-transition MI, bits/cell | 0.1701 | 0.1419 | 0.0675–0.3117 |
| Held-out unigram patch loss, bits/patch | 8.2160 | 8.2660 | 7.3152–8.9979 |
| Held-out factorized transition patch loss | 7.6315 | 7.6853 | 6.7142–8.4818 |
| **Held-out transition gain** | **0.5845** | **0.4584** | **0.1526–1.1972** |
| Held-out exact-copy mixture gain | 0.0168 | −0.0011 | −0.0155–0.0364 |

The transition predictor beats the unigram predictor in 99.4% of worlds. This is evidence that the cell signal survives patch tokenization at model-visible positions. Exact patch copying yields little typical held-out gain, so the transition gain is not simply whole-patch identity copying. It still may reflect learnable local color substitutions that a transformer could discover from its context; this screen does not establish that it does.

| World-level Spearman association | ρ |
| --- | ---: |
| Cell MI × held-out transition gain | +0.975 |
| **Early-frame cell MI** × held-out transition gain | **+0.914** |
| Held-out transition gain × gzip | −0.044 |
| Held-out transition gain × mean cell change | +0.124 |
| Held-out transition gain × horizontal patch-copy excess | +0.061 |
| Cell MI × gzip | −0.019 |
| Cell MI × mean cell change | +0.127 |
| Cell MI × horizontal patch-copy excess | +0.067 |

The early-frame MI relationship avoids using scored late transitions to compute that particular predictor of late gain. Within the five 0.02-wide gzip bins, rank correlation is +0.975 for full MI × patch gain and +0.055 for patch gain × horizontal excess. These are within-corpus feature associations, **not** correlations with downstream English results.

The full-reference cell-MI median is **0.141948363126 bits**. Above and below it are 12,500 worlds each; the high side contains 49–51% of worlds in each gzip sub-bin. High-MI worlds have mean held-out transition gain **0.9032** bits/patch versus **0.2658** below the median. The high side also has higher mean change (**0.71174** versus **0.69478**) and horizontal-copy excess (**+0.00433** versus **+0.00112**), while mean gzip differs slightly (**0.45492** versus **0.45558**). Those shifts matter despite small raw rank associations.

## Proposed next selector

Freeze `cell_transition_mi >= 0.141948363126` as the high arm and `< 0.141948363126` as the low arm. Compute the same full-trajectory 10-color transition table on each generated 27-frame world. Use the existing five gzip sub-bin quotas, train/validation world counts, tokenizer, packed layout, NCA label budget, model seeds and TinyStories schedule. Within each gzip sub-bin, add predeclared mean-change strata if quota feasibility permits; report exact gzip, change, horizontal-copy excess, lag-38/76 excess, color marginals, and fitted held-out patch gain for both arms. Prefer fresh disjoint rule streams and compare high, low and an unfiltered arm as independent corpus blocks. Only then use the paired downstream 1M probe and, for a credible direction, full 10M curves.

This is an independently motivated **candidate**, not an observed transfer winner. The factorized predictor assumes independent cells conditional on their previous colors and receives its rule-specific transition estimates from the same world; a single transformer trained across many worlds has no such table given to it. The MI threshold uses all 27 frames, so the high/low held-out gain comparison is partly selection on the evaluated trajectory. The separate early-frame MI association reduces that concern but is still descriptive. A useful causal test needs selected corpora, audit of secondary shifts, and independent rule-corpus replication.
