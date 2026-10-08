# Vertical lag-6 target: saved-data screen

**Status:** Descriptive 1,000-world screen and intervention design. Its provisional threshold has since been replaced by the [frozen full-reference protocol](VERTICAL_LAG6_FROZEN_PROTOCOL.md). No new vertical-arm trajectories were generated and no vertical probe model was trained. A world's feature value has no associated English loss, so the correlations here cannot establish transfer benefit.

Run `uv run --all-packages python progress/nca_search/screen_vertical_lag_target.py` from the repository root. The script deterministically reads every fifth world among the first 5,000 accepted 40–50% gzip reference receipts (1,000 worlds), and every twentieth **training** world from each completed high/low horizontal-copy corpus (999 per arm). These are three samples from existing, nonindependent rule streams; their statistics should not be treated as confidence intervals for new corpora.

## Target and visibility

The original packed frame has `START`, 36 raster-order 2×2 patches, and `END`. Two patches at the same column in consecutive patch rows are six packed token positions apart. Each world has 27 frames × 5 adjacent row pairs × 6 columns = 810 eligible lag-6 patch pairs. Define **vertical excess** as the fraction of identical pairs minus the average exact fixed-count chance within each frame, `Σ_id n_id(n_id−1)/(36×35)`. This removes the frame's patch-frequency contribution under random rearrangement, but leaves other spatial and rule structure. The first frame is masked in NCA training; four nonoverlapping 256-token windows also hide some source positions. **94.44%** of eligible vertical pairs are both scored and causally visible in the actual model windows. The original read-only screen used a stricter `s+1` boundary and reported 94.07%; `NCAWindows.batch` supplies input token `s`, so three additional predecessors are visible. The frozen all-pair selector is unchanged.

The completed horizontal-copy selection used a distinct global-world fixed-count chance, `Σ_id n_id(n_id−1)/(972×971)`, for 810 horizontal neighbor pairs. The script reports that exact completed-selector metric alongside vertical excess. It also reports a *visible* lag-76 same-site, two-frame patch excess, using the two frames' patch-frequency product as its chance baseline. This checks whether a vertical selector may also move a plausible temporal target.

| Cohort sample | n | Vertical excess mean | Vertical excess p10 / median / p90 | Horizontal excess mean | Visible lag-76 excess mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unfiltered 40–50 reference | 1,000 | +.001925 | −.009765 / −.000294 / +.015168 | +.002918 | +.006204 |
| High horizontal copy | 999 | +.002089 | −.009759 / −.000235 / +.015626 | +.011258 | +.009997 |
| Low horizontal copy | 999 | +.002055 | −.009230 / −.000764 / +.014145 | −.006240 | +.003416 |

The high and low horizontal-copy arms shift the *vertical mean* by only +.000034 in these samples, despite a +.017498 difference in horizontal excess. Their vertical medians and tails are also close. Visible lag-76 moves more strongly across those arms, so it remains a diagnostic for any new selector. These sampled differences are descriptive; they do not establish that vertical structure was held constant throughout the full corpora.

| Reference-world association with vertical excess | Spearman ρ | Spearman ρ within five gzip sub-bins |
| --- | ---: | ---: |
| Gzip ratio | +.028 | +.047 |
| Cell-change rate | −.047 | −.071 |
| Horizontal excess, completed selector | +.086 | +.071 |
| Visible lag-76 excess | +.265 | +.249 |
| Vertical excess using global-world chance | +.978 | +.977 |
| Visible vertical excess | +.993 | +.991 |

The correlation between horizontal excess and visible lag-76 excess is +.271 in the same reference sample. Lag-6 therefore offers a spatial target distinguishable from horizontal-copy selection in this screen, but it is moderately associated with a temporal target. Small rank associations also cannot rule out changes in patch marginals, rule identity, or nonlinear relationships.

The provisional 1,000-world vertical median was **−.000293945**. Fractions above it across gzip bins 40–42, 42–44, 44–46, 46–48, and 48–50% were **.520, .489, .543, .489, .474** respectively (bin sample sizes 152, 186, 199, 235, 228). The later full 25,000-world scan froze the actual cutoff at **−.0005878894767783627**; use the linked protocol for any new corpus.

## One-property intervention to test

1. Freeze the full-reference median of the **frame-adjusted vertical excess** above. Generate one high arm (`>= median`) and one low arm (`< median`) from a fresh, disjoint rule stream. Keep the original NCA generator, 27-frame trajectories, 2×2 tokenization, packed schema, training/validation world counts, usable NCA labels, five gzip-bin quotas, model initialization seeds, optimizer, and TinyStories schedule fixed. Determine the rule stream and train/validation split before inspecting the vertical score.
2. Check feasibility and balance before training: each arm's within-bin gzip distribution, cell-change rate, horizontal-copy excess, patch entropy/frequencies, visible lag-6 excess, and visible lag-76 excess. Predeclare tolerances for gzip/change/horizontal matching; if feasible, balance these nuisance features *within* gzip bins without changing the vertical high/low assignment rule. Report rule overlap and failed quotas. If matching cannot be achieved, retain the descriptive comparison and do not call it isolated.
3. First use an NCA patch-target probe to check whether the model actually learns the additional lag-6 dependency, including loss on targets with a visible vertical predecessor. Then run paired model seeds through the original 1M English triage and, for any promising result, the full 10M curve. Compare high vs low **and** matched unfiltered reference at identical downstream token milestones, reporting paired loss, perplexity, AUC, and seed variation.

**Prediction:** If locally learnable vertical recurrence helps this transformer objective, the high arm should have lower NCA loss on visible lag-6 targets and then lower downstream loss at fixed English token counts. A benefit on the NCA objective without an English benefit would reject the transfer part of this hypothesis. A downstream benefit without better visible-target learning would require another mechanism. The existing high-horizontal result does not validate this vertical prediction; the high/low horizontal arms have nearly equal sampled vertical excess.

Because previous selectors were screened and all model seeds for a given arm share one synthetic corpus, exploratory p-values from that corpus are not independent-corpus evidence. Replication should use multiple fresh rule-corpus blocks as outlined in [REPLICATION_DESIGN.md](REPLICATION_DESIGN.md).
