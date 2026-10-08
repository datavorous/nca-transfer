# Attention-lag screen on saved NCA trajectories

This is a **descriptive, read-only hypothesis screen**, not a language-transfer result. The [reproduction script](screen_attention_lags.py) samples every fifth accepted world among the first 5,000 saved `40_50` receipts (1,000 worlds), reads raw trajectories, and uses the original 2×2 patch IDs, raster order, and 38-token frames (`START`, 36 patches, `END`). Run:

```sh
uv run --all-packages python progress/nca_search/screen_attention_lags.py
```

For each lag, `copy` is the share of patch-token pairs with identical IDs. `excess` subtracts the expected copy rate after independently rearranging patches **within each frame**, preserving that frame's patch-ID counts. The finite-count expectation is used for two different positions in the same frame; the product of frame frequencies is used across frames. This is an analytic frequency baseline, not a simulation or an estimate of model learning. `Visible` means the earlier token lies in the same one of the four nonoverlapping 256-token NCA training windows as a scored patch target; first-frame targets are masked. All ρ values are world-level Spearman correlations, not correlations with English loss.

| Packed-token lag | Geometric relationship | Copy median | Excess median [p10, p90] | Visible pairs | ρ excess with gzip | ρ excess with cell change |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Horizontal neighbor in a patch row | .0099 | −.0004 [−.0101, +.0149] | 96.0% | +.040 | −.037 |
| 6 | Vertical neighbor in a frame | .0099 | −.0003 [−.0098, +.0152] | 94.4% | +.028 | −.047 |
| 37 | One site offset near the previous frame | .0099 | −.0004 [−.0067, +.0099] | 88.8% | −.005 | −.082 |
| 38 | Same site, previous frame | .0075 | −.0023 [−.0133, +.0315] | 88.5% | +.121 | −.717 |
| 39 | Opposite site offset near the previous frame | .0099 | −.0004 [−.0072, +.0097] | 88.1% | +.038 | −.046 |
| 76 | Same site, two frames earlier | .0111 | −.0005 [−.0145, +.0346] | 76.0% | +.099 | −.058 |
| 114 | Same site, three frames earlier | .0104 | +.0001 [−.0097, +.0190] | 62.5% | +.066 | −.037 |
| 152 | Same site, four frames earlier | .0121 | +.0007 [−.0086, +.0206] | 47.8% | −.012 | −.082 |
| 266 | Same site, seven frames earlier | .0097 | −.0005 [−.0081, +.0096] | 0.0% | +.047 | −.120 |

The lag-6 visibility cell is corrected to **765/810 = 94.44%** using the actual `NCAWindows.batch` input, which starts at window offset `s`. The original screen used a stricter `s+1` predecessor boundary and printed 94.1%; the other visibility cells retain that original screen convention. Copy/excess statistics and the frozen lag-6 selection cutoff are unaffected.

The **lag-38 same-site signal is strongly coupled to cell change** (ρ −.717), so selecting on it largely revisits the previous change-rate experiment. Lag 76 has a wide distribution but weak association with change (ρ −.058); it correlates +.795 with two-frame **cell** recurrence, as expected. Lag 114 correlates +.670 with three-frame cell recurrence. Lag 76 and lag 38 excess correlate only +.355, and lag 114 and lag 38 only +.130. These are distinct temporal motifs in this sample, although their common rule causes remain unknown. The `lag76 − lag38` excess median is −.0009 [−.0250, +.0244] and its ρ with change is +.672, so that *difference* would be a confounded selector unless change is matched.

The median excess at lags 37 and 39 stays close to zero. At lag 38 the median is negative, but the upper tail is positive; aligned copying is therefore present only in a subset. `lag38 − mean(lag37, lag39)` has median −.0018 [−.0143, +.0314] and ρ −.668 with change. Horizontal lag-1 excess is almost the same selector as the existing horizontal-copy measure (ρ +.976). Vertical lag-6 excess has only +.086 correlation with horizontal-copy excess and thus offers a separate spatial test.

The 256-token window increasingly hides long lags. Only 76% of eligible lag-76 pairs and 62.5% of lag-114 pairs are visible to the model; lag 266 is never visible in this NCA objective. The median **visible** excess is −.0003 at lag 76, −.0002 at lag 114, and +.0007 at lag 152. Selectors should use visible pairs or at least check them after sampling. Higher excess on invisible pairs cannot directly help this model through that exact dependency.

## Testable one-property selectors

Use the same 40–50 gzip band, train/validation counts, patch tokenizer, 19,715,325 usable NCA labels, generator, model, and TinyStories schedule as the reference. Freeze a threshold using saved reference worlds, match the five existing gzip sub-bin quotas, then check residual change rate and patch marginals. A seed-0 1M probe is only triage; paired seeds and full 10M curves are needed for a sample-efficiency conclusion.

| Priority | Selector | Why test it | Main diagnostic |
| ---: | --- | --- | --- |
| 1 | **High visible lag-76 excess** | Tests two-frame same-site patch recurrence at a causal lag with low raw correlation to gzip/change. | Check cell lag-2 recurrence, lag-38 copy, change and each gzip sub-bin; 24% of eligible pairs are hidden. |
| 2 | **High visible lag-114 excess** | Tests a longer three-frame recurrence, largely distinct from lag-38 copy. | Check cell lag-3 recurrence and whether 62.5% visibility leaves enough examples. |
| 3 | **High vertical lag-6 excess** | Offers local spatial copying distinct from the horizontal patch-copy candidate tested in iteration 04. | Check horizontal excess, patch frequencies and anisotropy; the same spatial rule may drive both. |
| 4 | **High lag-76 excess conditional on lag-38 excess** | Could isolate two-step memory from simple previous-frame persistence. | First establish cohort feasibility within gzip and change strata; matching two temporal properties may distort other features. |
| 5 | **Moderate visible lag-76 excess** | An interior recurrence level might supply predictability without nearly periodic copying. | Predefine the interval before probing; two-sided selection can change entropy and rule mix. |
| 6 | **High visible lag-152 excess** | Tests whether a fourth-frame motif remains useful when fewer than half the pairs fit the window. | Lower priority because 52.2% of pairs are invisible; compare visible and full-world scores. |
| 7 | **High lag-38 aligned advantage over lags 37/39** | Directly tests site-specific temporal alignment rather than generic cross-frame token repetition. | Strong change correlation (−.668); match change tightly or it mostly repeats the existing screen. |

These are **hypotheses**, not evidence of benefit. The frame-marginal baseline removes simple patch-frequency effects but does not condition on spatial position, rule identity, or temporal nonstationarity. Some excess values are negative because spatial arrangements can avoid copies. Copy is only one kind of predictability; transformations and conditional rules may be learnable without identical tokens. World-level measurements are correlated because they arise from one NCA rule, and screening multiple selectors risks finding a chance winner. Retain a strict one-change-at-a-time probe log and use held-out seeds for confirmation.
