# Saved-data hypothesis screen

This is a **hypothesis list, not a transfer result**. It reads existing raw trajectories and logged losses only. No NCA generation or model training was run for this screen. The [reproduction script](screen_existing.py) samples every fifth accepted world among the first 25,000 `40_50` receipts (5,000 worlds, in receipt order), loads the matching saved trajectories, and prints the measurements below:

```sh
uv run --all-packages python progress/nca_search/screen_existing.py
```

The original selection used gzip of 32-bit patch IDs, so many properties are expected to correlate with gzip. The correlations below are **world-level Spearman associations with gzip or mean cell-change rate**, not correlations with English validation loss. There is no per-world English outcome; all 40–50 worlds were pooled for pre-pretraining. Percentiles are across the 5,000 sampled worlds. Exact trajectory source: `runs/nca-100k-regimes/40_50/`; receipt gzip values: `runs/nca-100k-regimes/receipts/`.

## World-level measurements within 40–50 gzip

| Property | Definition | Median [10th, 90th] | ρ with gzip | ρ with change |
| --- | --- | ---: | ---: | ---: |
| Cell-change rate | Fraction of cells differing from preceding frame, averaged over 26 steps | .708 [.559, .848] | +.262 | 1.000 |
| Change slope | Mean change in last 13 steps minus first 13 | .000 [-.038, .034] | -.003 | +.153 |
| Change volatility | SD of 26 per-step change fractions | .035 [.026, .051] | -.067 | -.491 |
| Lag-2 recurrence | Fraction of cells equal to their state two frames earlier | .314 [.183, .454] | -.304 | -.192 |
| Lag-3 recurrence | Same at a three-frame lag | .303 [.200, .411] | -.372 | -.318 |
| Color entropy | Shannon entropy of all cell colors in a world, bits | 2.071 [1.830, 2.244] | +.822 | +.297 |
| Color-entropy drift | Last-frame minus first-frame color entropy, bits | -.014 [-.215, .166] | +.045 | +.024 |
| Frame-entropy volatility | SD of 27 framewise color entropies, bits | .090 [.062, .129] | -.103 | -.136 |
| Dominant-color share | Largest color count divided by all cell counts | .424 [.338, .538] | -.416 | -.355 |
| Spatial-edge fraction | Unequal cells across horizontal/vertical periodic neighbors | .703 [.613, .783] | +.459 | +.196 |
| Spatial excess agreement | Periodic neighbor agreement minus exact fixed-count random-neighbor expectation | -.0017 [-.0619, .0563] | -.020 | +.046 |
| Spatial anisotropy | Absolute horizontal–vertical difference in edge fractions | .054 [.009, .182] | -.173 | +.022 |
| Transition conditional entropy | H(next cell color \| current cell color), bits | 1.895 [1.667, 2.087] | +.884 | +.241 |
| Transition mutual information | H(next cell color) − conditional entropy, bits | .143 [.068, .316] | -.040 | +.129 |
| Unique-patch fraction | Unique 2×2 patch IDs divided by 972 patch positions | .311 [.221, .392] | +.910 | +.189 |

The sampled world's median cell-change rate is .7075; the **full 25,000-world** median used for iterations 01–02 is .7086004. Every sampled world has all 144 cells active at least once; `active_cells` is saturated and is therefore a poor selector. Early and late change rates correlate with overall change at .988 and .991, so separate thresholds on them mostly retest the existing change-rate hypothesis. Transition entropy, color entropy, and patch diversity largely track gzip within this band; changing them requires more careful matching.

For spatial excess agreement, each frame's fixed-color-count random-neighbor baseline is `sum_c n_c(n_c−1)/(144×143)`; subtract its 27-frame mean from observed periodic horizontal/vertical neighbor agreement. This finite-grid correction matters near zero: a with-replacement `sum_c p_c²` baseline shifts the sampled median down by about .005. The corrected sampled median is **-.0017**.

A full reference scan (`uv run --all-packages python progress/nca_search/screen_existing.py --stride 1`) covers all 25,000 saved 40–50 worlds. Spatial excess agreement has median **-0.001144367464**, Spearman ρ with gzip **+.007**, and ρ with mean change **+.047**. This full-corpus median is the frozen threshold for iteration 03.

## Feature overlap and gzip-bin confounds

The second coefficient ranks each property **within** each of the five .02-wide gzip sub-bins, centers those ranks, then pools all 5,000 worlds. It describes feature overlap after coarse gzip matching; it is not a causal adjustment or a transfer-outcome correlation. Selected pairwise Spearman coefficients:

| Feature pair | Raw ρ | Within-bin ρ |
| --- | ---: | ---: |
| Spatial excess agreement × transition mutual information | -.040 | -.042 |
| Spatial excess agreement × lag-2 recurrence | -.042 | -.054 |
| Spatial excess agreement × change slope | +.033 | +.034 |
| Spatial excess agreement × change volatility | -.008 | -.009 |
| Spatial excess agreement × mean change | +.046 | +.060 |
| Transition mutual information × lag-2 recurrence | +.218 | +.235 |
| Transition mutual information × mean change | +.129 | +.111 |
| Lag-2 recurrence × change volatility | +.291 | +.278 |
| Lag-2 recurrence × mean change | -.192 | -.137 |
| Change volatility × mean change | -.491 | -.495 |
| Mean change × gzip | +.262 | +.064 |

Spatial excess agreement is almost uncorrelated with these temporal properties in this sample. A spatial-coherence probe would therefore be meaningfully different from the current mean-change screen, although individual rules may differ for many unmeasured reasons. Higher lag-2 recurrence tends to come with higher change volatility; a recurrence selection should measure both. Lower volatility strongly selects higher mean change even inside gzip bins, so it needs additional mean-change matching. Coarse gzip-bin matching reduces some associations (mean change versus gzip: +.262 to +.064) but does not eliminate within-bin gzip variation; compare exact selected gzip distributions too.

## Thirteen candidate interventions

All directions below are proposals. For each, keep the original rule family, train/validation split, world counts, 19,715,325 usable NCA labels, model, optimizer, tokenizer, TinyStories stream, and 40–50 gzip band. Match the five 0.02-wide gzip sub-bin quotas used in [iteration 01](EXPERIMENT_LOG.md); change **one** selection threshold at a time. Use the same seed-0 1M-token probe first, then paired seeds and full 10M only for credible winners. Check actual within-sub-bin gzip means, rule overlap, and other correlated features before assigning a mechanism.

| Priority | Proposed selector | Testable rationale | Main confound / diagnostic |
| --- | --- | --- | --- |
| 1 | Higher spatial excess agreement | Local spatial coherence above color-frequency chance may create recurring learnable patch patterns. | Could track other edge or rule properties; compare patch frequencies and temporal features after selection. |
| 2 | Higher transition mutual information | A predictable relationship between consecutive cell states may supply learnable temporal structure while retaining substantial activity. | A one-cell statistic ignores spatial context; compare its level and NCA validation loss after selection. |
| 3 | Higher lag-2 recurrence at fixed change rate | Two-step structure might reward memory beyond copying the immediately previous frame. | Could merely select alternating or short-period worlds; inspect lag-1/lag-3 recurrence and tail periods. |
| 4 | Near-zero or positive change slope | Sustained or increasing late activity may avoid trajectories that exhaust their dynamics. | Mean change can drift; match mean change as well as gzip for attribution. |
| 5 | Lower frame-entropy volatility | Stable color diversity over time may give steadier learning targets. | Could still select different average entropy; compare world color entropy. |
| 6 | Low absolute color-entropy drift | Avoid worlds collapsing or exploding in color diversity over 27 frames. | Opposite signed drifts may have different effects; inspect both tails separately. |
| 7 | Higher spatial anisotropy | Directional patterns might create distinct local predictive rules. | Periodic grid and random rule orientation; verify this is not a rare artifact. |
| 8 | Lower change volatility at matched mean change | Consistent transition difficulty may be easier to learn than intermittent bursts. | Correlates -.491 with mean change; a simple threshold changes mean activity. |
| 9 | Higher spatial-edge fraction | More neighboring contrasts could expose local rule interactions. | Correlates +.459 with gzip; coarse sub-bin matching may not remove it. |
| 10 | Lower lag-3 recurrence at matched lag-2 | Fewer trivial multi-step repeats might raise useful dynamics. | Recurrence also depends on dominant-color share and activity. |
| 11 | Interior color-entropy band | Both nearly single-color and highly diffuse worlds may be less useful than moderate occupancy. | Strong gzip correlation (+.822); compare matched gzip and color marginals. |
| 12 | Interior conditional-transition entropy band | A mix of predictability and novelty may beat either trivial or near-random transitions. | Very strong gzip correlation (+.884); first check whether feasible within each gzip sub-bin. |
| 13 | Interior unique-patch fraction | Moderate patch vocabulary reuse may improve learnability of the patch tokenizer. | Almost a gzip proxy (+.910); low identification value unless tightly matched. |

Priority reflects **how distinct the property is from gzip and the existing mean-change screen**, plus a plausible mechanism. It is not a prediction of effect size. For priorities 1–3, sample high and low cohorts around a saved-corpus median; for 4–13, predefine thresholds from the saved distribution before seeing transfer results. A percentile threshold might fail quota feasibility in some gzip sub-bins; record attempts and acceptance rates.

## Logged transfer evidence is much coarser

The completed five-seed run compares four **gzip bands**, not the properties above. Its logged mean downstream validation losses are:

| Band | NCA val loss | TinyStories @1M | TinyStories @10M |
| --- | ---: | ---: | ---: |
| 20–30 | 3.9393 | 6.0512 | 4.2057 |
| 30–40 | 4.9413 | 6.0182 | 4.1951 |
| 40–50 | 6.1158 | 6.0133 | 4.1849 |
| 50+ | 8.1980 | 6.1909 | 4.2328 |

Sources: [five-seed summary](../../runs/1m-nca-transfer-v2/summary/README.md) and [post-hoc report](../part1/PAPER_REGIME_POSTHOC.md). Across these four condition means, rank correlation between NCA validation loss and either TinyStories column is only **+0.20**: NCA validation gets harder monotonically, while language transfer follows a shallow non-monotonic ordering. Four selected bands cannot identify which world-level property caused that ordering. The 40–50 lead over 30–40 at 10M is only .0102 loss units, with seed-dependent rankings; use it as a reference corpus, not proof of an exact optimum.

Iteration 01's selected **low-change** 40–50 corpus was worse than the reference by +.03457 seed-0 validation loss at ~1M TinyStories tokens, despite nearly matched mean gzip. Iteration 02's high-change corpus gave a mean paired 1M gain of only +.00302 loss across five model seeds and improved early-curve AUC in just one. Neither result identifies a reliable change-rate benefit or a full 10M outcome; details are in [the experiment log](EXPERIMENT_LOG.md).

The most informative next independent screens are **spatial excess agreement** and **transition mutual information**, followed by lag-2 recurrence with mean activity checked or matched. These features offer measurable variation inside 40–50 without mostly restating gzip; the rest are useful candidate explanations but are more entangled with existing selection variables.
