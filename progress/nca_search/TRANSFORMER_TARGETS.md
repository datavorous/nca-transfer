# Transformer-facing NCA data targets

This is a **saved-data hypothesis screen**, not a transfer result. No trajectories were generated and no model was trained. The [reproduction script](screen_transformer_targets.py) reads every fifth accepted world among the first 15,000 `40_50` receipts, yielding 3,000 deterministic worlds. It uses the original 2×2 patch powers `(1, 10, 100, 1000)`, raster order, 38-token frames including markers, and four nonoverlapping 256-token training windows. Run:

```sh
uv run --all-packages python progress/nca_search/screen_transformer_targets.py
```

Each Spearman coefficient below is across **worlds**. There is no English outcome for an individual world, so none is a correlation with language transfer. All properties can be selected within the existing 40–50 gzip band; that alone does not hold gzip, change rate, token frequencies, or rule identity constant.

## Measurements on the actual patch stream

Percentiles are across the 3,000 worlds. `ρ gzip` and `ρ change` use the logged gzip ratio and mean cell-change rate. `ρ change | bin` ranks and centers variables within five 0.02-wide gzip sub-bins before pooling; it is a descriptive diagnostic, not causal adjustment.

| Property | Exact definition | Median [p10, p90] | ρ gzip | ρ change | ρ change \| bin |
| --- | --- | ---: | ---: | ---: | ---: |
| Patch entropy | Plug-in Shannon entropy of 972 patch IDs per world, bits | 7.317 [6.568, 7.883] | +.982 | +.286 | +.168 |
| Effective patch vocabulary | `2 ** patch_entropy`; diversity equivalent, not observed distinct count | 159.5 [94.8, 236.0] | +.982 | +.286 | +.168 |
| Rare-patch mass | Share of patch positions whose ID occurs at most twice in that world | .267 [.168, .363] | +.820 | +.131 | -.142 |
| Context repeat | Share of scored patch targets seen earlier in their *same* 256-token training window | .468 [.365, .589] | -.979 | -.261 | -.056 |
| Horizontal patch copy | Fraction of raster-adjacent 2×2 patches with the same ID, excluding row edges | .0099 [.0012, .0346] | -.398 | -.187 | -.090 |
| Vertical patch copy | Fraction of vertically adjacent 2×2 patches with the same ID | .0099 [.0012, .0346] | -.402 | -.180 | -.094 |
| Temporal same-site copy | Fraction of patches identical at the same grid coordinate in consecutive frames | .0064 [.0000, .0471] | -.216 | -.903 | -.899 |
| Chance patch copy | Exact fixed-count probability `sum_id n_id(n_id−1)/(972×971)` for two distinct random patch positions | .0111 [.0056, .0231] | -.843 | -.324 | -.223 |
| Horizontal copy excess | Horizontal patch copy minus chance patch copy | -.0001 [-.0098, .0170] | +.023 | +.014 | +.011 |
| Temporal copy excess | Temporal same-site copy minus chance patch copy | -.0030 [-.0138, .0311] | +.170 | -.738 | -.838 |
| Previous-frame vocabulary share | Per-frame fraction of patch positions with an ID anywhere in the preceding frame; mean over 26 transitions | .254 [.157, .389] | -.853 | -.424 | -.390 |
| Late patch novelty | Share of positions in the final 9 frames with IDs absent from the first 18 frames | .204 [.114, .293] | +.757 | +.105 | -.137 |
| Horizontal bigram repeat | Share of within-row ordered adjacent-patch pairs already seen earlier in the world | .059 [.016, .164] | -.863 | -.247 | -.068 |
| Late − early patch entropy | Entropy of final 9 frames minus entropy of first 9 frames, bits | -.039 [-.499, .209] | +.146 | +.003 | -.026 |

The within-world entropy estimates have finite-sample bias because each world has only 972 patch positions. `chance patch copy` uses the exact fixed-count probability for two distinct random positions, not the with-replacement `sum_id (n_id/972)^2` baseline. The latter overstates chance by `(1−sum_id (n_id/972)^2)/971`, about .0010 here; correcting it raises the median horizontal-copy excess from −.0012 to −.0001 and temporal-copy excess from −.0040 to −.0030, with virtually unchanged rank associations. It remains a frequency baseline, not a fully conditioned null for adjacent positions; excess can be negative. The context-repeat calculation follows the NCA training windows and excludes the masked first-frame patch targets. It measures availability of an exact prior token, **not** what the transformer learned or its conditional entropy. A patch at the same site in the preceding frame is 38 token positions earlier and lies inside the same 256-token window except near a window boundary; the temporal-copy measure itself does not apply that boundary check. Repeated bigrams are a simple in-world reuse count, not a held-out predictive score.

The full 25,000-world reference scan (`uv run --all-packages python progress/nca_search/screen_transformer_targets.py --first 25000 --stride 1`) gives horizontal-copy excess median **-0.000374227071**, with Spearman ρ **+.029** to gzip and **+.035** to cell change. That frozen median is the threshold for iteration 04.

## Ranked, testable hypotheses

Rank reflects distinctness from the current gzip/change screens and a plausible link to the causal patch-token objective, **not** expected improvement. Use the existing 19,975/256 world counts, original generator and rule family, exact tokenizer and label budget, fixed model/TinyStories setup, and five gzip sub-bin quotas. Record actual gzip and change distributions after selection. For a clean property test, match mean-change strata as well where feasible, or report the residual difference. Predefine thresholds from these saved data before reading a downstream probe result.

| Rank | Candidate target and reason | One-property intervention | Main confound / diagnostic |
| ---: | --- | --- | --- |
| 1 | **Higher horizontal patch-copy excess.** Adjacent raster tokens may provide a learnable local dependency beyond patch frequency. It varies almost independently of gzip (ρ +.023), change (+.014), and earlier cell-level spatial excess (+.024). | Select above versus below the exact fixed-count-baseline median, with gzip and change-stratum quotas; compare seed-0 1M probes. | Copy excess can reflect repeated motifs or marginal shifts. Check patch entropy, rare mass, vertical copy, and within-frame predictive loss. |
| 2 | **Less late entropy decline.** Sustained patch diversity may keep later training windows informative; with equal 9-frame blocks, median late−early entropy is −.039 bits and its correlation with change after gzip bins is −.026. | Threshold on late−early patch entropy, then match gzip/change and world counts. | Frame order and rule dynamics can change patch marginals without changing useful predictability; compare against a time-shuffled frame-order baseline. |
| 3 | **Higher cell transition mutual information.** Existing screen found this only weakly linked to gzip/change, and it may create learnable state transitions; this script shows only weak relation to patch-level copy excess (ρ +.052). | Select high/low cell-transition MI within matched gzip/change strata; evaluate patch-level NCA val learning and English probe. | One-cell MI may not translate to patch-token prediction; compare temporal patch copy and lag-2 recurrence. |
| 4 | **Higher repeated horizontal bigram share at fixed gzip.** Recurrent local token transitions may help a causal transformer learn compositional structure. Median .059. | Choose high/low bigram-repeat cohorts after tight gzip and patch-entropy matching. | Strong gzip (−.863) and context-repeat (+.843) associations; a naive threshold is largely a lower-complexity intervention. |
| 5 | **Moderate late patch novelty.** New patch IDs in later windows could balance reuse and coverage; median .204. | Predeclare an interior novelty interval versus both tails, preserving gzip-bin counts. | Strong gzip association (+.757); interior selection also changes entropy and rare mass. This is two-sided and needs matched controls. |
| 6 | **Higher exact context reuse at fixed patch entropy.** A 256-token model can exploit recurring patch IDs only if they reappear in its visible window. Median scored-target repeat .468. | Stratify jointly by gzip and patch entropy, then select high/low residual context repeat. | Nearly a gzip/entropy proxy (ρ gzip −.979; ρ patch entropy −.981). Feasibility may be poor; reject the intervention if cohorts cannot be matched. |
| 7 | **Interior rare-patch mass.** Too many singleton/doubleton tokens may make the 10,000-ID NCA vocabulary sparse; too few may over-repeat. Median .267. | Compare predeclared middle interval to tails with tight gzip and patch-entropy matching. | Correlates +.820 with gzip. Within-world counts depend on finite sample size and patch marginal distribution. |
| 8 | **Higher same-site temporal patch reuse at fixed change.** A 38-token lag is within attention range and may reward temporal memory. | Select on temporal copy excess only after matching mean cell-change and gzip strata. | Temporal same-site copy is almost a restatement of cell-change (ρ −.903); excess remains strongly tied to change (ρ −.739). Low priority unless residual variation is substantial. |

The [high-change probe](EXPERIMENT_LOG.md) gave only a +.00302 mean paired 1M endpoint gain across five model seeds and improved early-curve AUC in one of five. The low-change and high-spatial seed-0 probes were worse than the reference at 1M. None identifies a world-level mechanism or validates any target above. The next independent probe is horizontal patch-copy excess, followed by cell-transition MI. For every credible winner, use paired seeds and the complete 10M downstream curve before making a sample-efficiency claim.
