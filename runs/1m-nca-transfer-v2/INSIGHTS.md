# Interim NCA → TinyStories research insights

**Snapshot:** 2026-10-08 09:15:09 IST. **Status:** 3/5 seeds have all six downstream conditions complete (18/30 downstream stages); 16/25 NCA stages have completed. Seeds 3–4 have no completed downstream run in this snapshot. The live training queue was not changed.

**Source and methods.** Every number comes from `seeds/seed_*/{downstream,nca}/*/metrics.json`, restricted to stages with `done.json`. The frozen design is in `config/experiment.json`. All downstream stages shown consumed exactly 10,000,000 TinyStories targets on the same validation set. “±” is sample standard deviation across seeds 0–2 (n=3). Perplexity summaries average each seed’s logged `exp(validation_loss)`; they are not `exp(mean loss)`. Lower is better. Gains are paired within seed. This is an interim descriptive analysis, not the five-seed result.

## Interim figures (completed seeds 0–2)

- [English learning curves: full and late-training views](plots/english_learning_curves_interim_n3.png) ([SVG](plots/english_learning_curves_interim_n3.svg))
- [Paired gains versus scratch and matched shuffled NCA](plots/english_paired_gains_interim_n3.png) ([SVG](plots/english_paired_gains_interim_n3.svg))
- [Tokens needed to reach scratch's 10M-token loss](plots/english_tokens_to_scratch_interim_n3.png) ([SVG](plots/english_tokens_to_scratch_interim_n3.svg))

## Final downstream metrics at 10M tokens

| Condition | Validation loss, mean ± SD | Perplexity, mean ± SD | Seeds with lower loss than scratch |
| --- | ---: | ---: | ---: |
| Scratch | 4.3175 ± 0.0036 | 75.00 ± 0.27 | — |
| Shuffled NCA | 4.2054 ± 0.0232 | 67.06 ± 1.56 | 3/3 |
| 20–30 | 4.2214 ± 0.0225 | 68.14 ± 1.53 | 3/3 |
| 30–40 | 4.1891 ± 0.0140 | 65.96 ± 0.92 | 3/3 |
| 40–50 | 4.1787 ± 0.0234 | 65.30 ± 1.53 | 3/3 |
| 50+ | 4.2258 ± 0.0322 | 68.45 ± 2.18 | 3/3 |

## 1. Synthetic pretraining effect: structured NCA versus scratch

Positive gain means lower validation loss after NCA pretraining. The shuffled control is included as a generic-pretraining reference; it is not structured NCA.

| Condition | Seed 0 gain | Seed 1 gain | Seed 2 gain | Mean ± SD gain | Wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shuffled NCA | 0.0825 | 0.1300 | 0.1239 | 0.1121 ± 0.0259 | 3/3 |
| 20–30 | 0.1186 | 0.0809 | 0.0890 | 0.0962 ± 0.0198 | 3/3 |
| 30–40 | 0.1399 | 0.1155 | 0.1300 | 0.1285 ± 0.0123 | 3/3 |
| 40–50 | 0.1406 | 0.1129 | 0.1628 | 0.1388 ± 0.0250 | 3/3 |
| 50+ | 0.0669 | 0.0757 | 0.1325 | 0.0917 ± 0.0356 | 3/3 |

## 2. Structure-specific effect: structured NCA versus shuffled NCA

Here gain = shuffled-NCA loss − structured-NCA loss. **Only 40–50 is a matched-content comparison:** the shuffled trajectories were derived from the 40–50 packed corpus with patch-token order permuted. Comparisons with the other gzip bands also change source-corpus composition.

The [shuffled-control gzip analysis](analysis/SHUFFLED_GZIP.md) finds that approximately 97% of shuffled trajectories remain in the source 40–50 gzip band for every seed.

| Structured band | Seed 0 gain | Seed 1 gain | Seed 2 gain | Mean ± SD gain | Wins vs shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20–30 | 0.0361 | -0.0491 | -0.0349 | -0.0160 ± 0.0457 | 1/3 |
| 30–40 | 0.0575 | -0.0146 | 0.0061 | 0.0163 ± 0.0371 | 2/3 |
| **40–50 (matched)** | 0.0582 | -0.0171 | 0.0389 | 0.0267 ± 0.0391 | 2/3 |
| 50+ | -0.0155 | -0.0543 | 0.0086 | -0.0204 ± 0.0317 | 1/3 |

## Full downstream learning curves

Mean validation loss ± sample SD at each planned checkpoint. Actual consumed-token positions were identical for every condition: 251,904, 501,760, 1,001,472, 2,000,896, 3,000,320, 5,001,088, 7,501,696, 10,000,000.

| Planned tokens | Scratch | Shuffled NCA | 20–30 | 30–40 | 40–50 | 50+ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25M | 9.924 ± 0.084 | 9.589 ± 0.049 | 9.514 ± 0.081 | 9.526 ± 0.166 | 9.545 ± 0.032 | 9.804 ± 0.056 |
| 0.5M | 8.414 ± 0.057 | 7.400 ± 0.053 | 7.329 ± 0.006 | 7.309 ± 0.126 | 7.310 ± 0.040 | 7.943 ± 0.186 |
| 1M | 6.533 ± 0.022 | 6.036 ± 0.020 | 6.052 ± 0.052 | 6.020 ± 0.016 | 5.998 ± 0.016 | 6.180 ± 0.125 |
| 2M | 5.583 ± 0.016 | 5.345 ± 0.036 | 5.437 ± 0.087 | 5.325 ± 0.054 | 5.292 ± 0.031 | 5.314 ± 0.050 |
| 3M | 5.070 ± 0.011 | 4.852 ± 0.040 | 4.922 ± 0.068 | 4.820 ± 0.032 | 4.809 ± 0.044 | 4.835 ± 0.043 |
| 5M | 4.536 ± 0.004 | 4.388 ± 0.026 | 4.427 ± 0.038 | 4.370 ± 0.017 | 4.359 ± 0.030 | 4.403 ± 0.032 |
| 7.5M | 4.346 ± 0.004 | 4.231 ± 0.023 | 4.248 ± 0.025 | 4.214 ± 0.015 | 4.203 ± 0.025 | 4.251 ± 0.033 |
| 10M | 4.318 ± 0.004 | 4.205 ± 0.023 | 4.221 ± 0.023 | 4.189 ± 0.014 | 4.179 ± 0.023 | 4.226 ± 0.032 |

## Sample efficiency and area under the learning curve

Threshold: the **mean scratch loss at 10M**, 4.3175. “Tokens to threshold” interpolates the three-seed **mean curve** between logged checkpoints. The paired column instead uses each seed’s own scratch@10M loss. AUC uses the trapezoidal rule from 0.5M to 10M actual TinyStories tokens, with a linearly interpolated 0.5M endpoint; its unit is loss × million tokens (lower is better).

| Condition | Mean-curve tokens to threshold | Savings vs 10M | Paired tokens: seeds 0 / 1 / 2 | AUC, mean ± SD |
| --- | ---: | ---: | --- | ---: |
| Scratch | 10.000M | 0.0% | 10.000M / 10.000M / 10.000M | 46.664 ± 0.023 |
| Shuffled NCA | 6.123M | 38.8% | 6.613M / 5.887M / 5.851M | 44.710 ± 0.259 |
| 20–30 | 6.531M | 34.7% | 6.097M / 6.852M / 6.584M | 45.052 ± 0.392 |
| 30–40 | 5.847M | 41.5% | 5.681M / 6.116M / 5.734M | 44.507 ± 0.168 |
| 40–50 | 5.669M | 43.3% | 5.598M / 6.159M / 5.219M | 44.374 ± 0.268 |
| 50+ | 6.407M | 35.9% | 6.770M / 6.717M / 5.741M | 45.009 ± 0.417 |

## Gzip-band ranking and the 40–50 hypothesis

Bands are the generation-time gzip-ratio intervals 20–30%, 30–40%, 40–50%, and ≥50%. The table ranks final downstream validation loss; AUC is a separate summary.

| Mean rank | Band | Final loss, mean ± SD | Seed 0 rank | Seed 1 rank | Seed 2 rank | AUC, mean ± SD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 40–50 | 4.1787 ± 0.0234 | 1 | 2 | 1 | 44.374 ± 0.268 |
| 2 | 30–40 | 4.1891 ± 0.0140 | 2 | 1 | 3 | 44.507 ± 0.168 |
| 3 | 20–30 | 4.2214 ± 0.0225 | 3 | 3 | 4 | 45.052 ± 0.392 |
| 4 | 50+ | 4.2258 ± 0.0322 | 4 | 4 | 2 | 45.009 ± 0.417 |

## Completed NCA-side validation inventory

These are final held-out-rule NCA losses for structured bands. The shuffled value uses shuffled validation data and must not be interpreted as rule generalization. Losses across different bands also reflect different prediction difficulty, so they do not directly rank transfer quality.

| Seed | Shuffled NCA | 20–30 | 30–40 | 40–50 | 50+ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6.0849 | 3.9230 | 4.9275 | 6.0356 | 8.1987 |
| 1 | 6.0521 | 3.9422 | 4.9325 | 6.0708 | 8.1027 |
| 2 | 6.1064 | 3.9409 | 4.9415 | 6.1075 | 8.2344 |
| 3 | 6.2464 | — | — | — | — |

## Paired uncertainty and limits

- 40–50 versus scratch: paired gain 0.1388 ± 0.0250 loss, 3/3 wins. 40–50 versus matched shuffled: 0.0267 ± 0.0391, 2/3 wins. 40–50 versus 30–40: 0.0103 ± 0.0195, 2/3 wins.
- With only three paired seeds, even a 3/3 directional win has a two-sided exact sign-test p-value of 0.25. No comparison here establishes conventional statistical significance. Four gzip-band comparisons are correlated and should not be counted as twelve independent replications.
- The current report omits unfinished downstream seeds 3–4 rather than treating them as missing at random. The final five-seed analysis may change ranks, effect sizes, and uncertainty.

## Defensible conclusions from this snapshot

1. **Structured pretraining beats scratch in all 12 available paired band–seed comparisons.** The four bandwise mean final-loss gains span 0.0917–0.1388, but only three independent seeds are complete.
2. **Generic pretraining also helps.** Shuffled NCA beats scratch in all three seeds, with a mean final-loss gain of 0.1121. Structured-versus-scratch gains alone therefore cannot identify an NCA-structure effect.
3. **The matched structure-specific effect is uncertain.** 40–50 beats its shuffled counterpart in seeds 0 and 2 but loses in seed 1; the paired mean gain is 0.0267 ± 0.0391 loss.
4. **40–50 has the best three-seed mean final loss and lowest AUC among the structured bands.** It ranks first in seeds 0 and 2, second in seed 1. Its final mean edge over 30–40 is only 0.0103 loss, so “Goldilocks optimum” is a hypothesis, not a settled result.
5. **Sample-efficiency gains are substantial relative to scratch.** Mean curves reach scratch@10M in 5.669M–6.531M tokens for structured bands (34.7–43.3% nominal savings); shuffled NCA needs 6.123M (38.8% savings). The incremental 40–50-versus-shuffled advantage is much smaller.
6. **Benefits are largest early but do not vanish by 10M.** At ~0.5M tokens, 40–50 lowers mean validation loss versus scratch by 1.105; at 10M the gain remains 0.139. The matched 40–50-versus-shuffled gap is 0.090 at ~0.5M and 0.027 at 10M.
7. **Seed dependence is material.** Shuffled NCA is best overall in seed 1, while 40–50 is best overall in seeds 0 and 2. The 20–30 band loses to shuffled NCA in seeds 1 and 2.
8. **NCA validation difficulty is not a direct transfer score.** Held-out-rule NCA loss generally rises from 20–30 through 50+, but the lowest NCA loss does not give the best TinyStories final loss.

**Data-quality note:** `config/experiment.json` retains `one_seed_exploratory: true` from v1 although it also lists seeds 0–4 and six v2 conditions. This is a metadata inconsistency; the completion markers and per-seed metrics define the actual snapshot. No experiment configuration or training artifact was edited for this report.
