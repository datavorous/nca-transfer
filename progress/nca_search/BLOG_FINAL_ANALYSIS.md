# NCA recurrence and early language learning

**Final horizontal-replication analysis, 2026-10-08.** All 45 high/unfiltered/low × A/B/C × seed 0–4 probes completed at 1,001,472 TinyStories tokens. The corpus block, not each model seed, is the independent data-generation unit. No 10M horizontal continuation or other follow-up training is included.

The later, separately authorized one-seed dose sweep and winning-model 10M continuation are reported in [SWEEP_RESULTS.md](../boosted_nca/SWEEP_RESULTS.md); they are not part of this 45-run replication.

## Fixed comparison

The reference is the existing 40–50% gzip NCA corpus. Each arm has 19,975 training worlds, 256 validation worlds, and 19,715,325 scored synthetic labels. The text model has 1,081,177 trainable parameters. The tokenizer, NCA rule family, model initialization within seed, optimizer, and TinyStories stream are held fixed. The horizontal selector counts exact copies between neighboring 2×2 patches in each frame's patch rows (**810** adjacent pairs per 27-frame world) and subtracts the fixed-count chance rate `Σ nᵢ(nᵢ−1)/(972×971)` from all 972 patch tokens. High accepts scores **≥ −0.000374227071** and low accepts **≤** that reference-derived cutoff; unfiltered ignores it. In each fresh rule block A/B/C, a rule is assigned to an arm by hash before its feature is measured. [Original configuration](../../runs/1m-nca-transfer-v2/config/experiment.json); [sampler definition](sample_patch_replication.py); [frozen replication](REPLICATION_DESIGN.md).

## What the original experiment established

At 10M TinyStories tokens over five seeds, scratch reaches **4.322 ± 0.020** validation loss and **75.33 ± 1.55** perplexity; structured 40–50 NCA reaches **4.185 ± 0.024** and **65.70 ± 1.58**. The paired loss gain is **+0.137 ± 0.039**. The NCA condition reaches each seed's scratch@10M loss after **5.726 ± 0.685M** downstream tokens on average, **42.7% ± 6.9% fewer**, or mean paired ratio **1.77 ± 0.21x**. These are downstream-token ratios; they exclude synthetic pre-pretraining compute. Shuffled NCA also beats scratch (mean final loss **4.203 ± 0.025**), so structured-versus-scratch alone does not isolate trajectory order. [Five-seed analysis](../part1/PAPER_REGIME_POSTHOC.md).

| Condition | 10M validation loss, mean ± seed SD |
|:--|--:|
| Scratch | 4.322 ± 0.020 |
| 20–30% gzip NCA | 4.206 ± 0.028 |
| 30–40% gzip NCA | 4.195 ± 0.024 |
| 40–50% gzip NCA | **4.185 ± 0.024** |
| 50%+ gzip NCA | 4.233 ± 0.026 |

Every structured band beats paired scratch at 10M in all five seeds. The 40–50 band has the lowest mean but is the best final-loss band in only **2/5** seeds. Its mean edge over 30–40 is **0.0103 loss**, smaller than their paired-difference SD **0.0377**. It beats 50+ by **0.048 ± 0.017** in **5/5** seeds. Thus 40–50 is a useful fixed reference, not a demonstrated sharp Goldilocks optimum. [Band analysis](../part1/PAPER_REGIME_POSTHOC.md).

## Exploratory search before fresh blocks

These early 1M probes kept the gzip regime, NCA budget, model, and downstream setup fixed. Candidate corpora often shared a rule stream with the reference, so these are selection leads, not independent corpus replications. Positive gain means lower loss than the original unfiltered 40–50 reference.

| Selection | Seeds | Mean paired 1M loss gain | Reading |
|:--|--:|--:|:--|
| Lower temporal cell change | 1 | −0.03457 | Lost |
| Higher temporal cell change | 5 | +0.00302 | Mixed; mean AUC worse |
| Higher spatial excess agreement | 1 | −0.03915 | Lost |
| Higher transition MI | 1 | −0.05833 | Lost at every logged milestone |
| Higher horizontal patch-copy excess | 5 | **+0.05917** | Won 5/5 endpoints, 3/5 AUCs |
| Lower horizontal patch-copy excess | 5 | −0.04275 | Opposite-direction control; lost 5/5 |

The selected high-copy discovery corpus retained a smaller exploratory 10M paired gain, **+0.01864 ± 0.02700**, winning **4/5** seeds. Its advantage was larger near 1–3M than at 10M. This one-corpus continuation does not establish persistent final quality across fresh rule populations. [Experiment log](EXPERIMENT_LOG.md); [1M table](experiments.csv); [directional control](patch_copy_direction.csv); [exploratory 10M table](full_10m.csv).

The base NCA-versus-scratch result and the incremental high-copy-versus-unfiltered result answer different questions. The later ratio, if supported by the fresh curves, must be reported as a change **within the same synthetic pre-pretraining budget**, not as the full speedup attributable to NCA.

## Fresh horizontal high / unfiltered / low replication

All three new rule-population blocks were generated independently, with disjoint rule IDs, pre-feature arm assignment, matched gzip sub-bin quotas and the same NCA label budget. Every row below pairs the three arms on corpus block and model initialization. **Positive U−H means high-copy has lower loss than contemporaneous unfiltered NCA.** AUC is normalized trapezoidal validation loss from zero to 1,001,472 TinyStories tokens; lower is better. The source is each arm's `done.json` and `metrics.json`, indexed in [replication_1m.csv](replication_1m.csv) and verified by the read-only [closeout renderer](render_horizontal_closeout.py).

| Block | Seed | High loss | Unfiltered loss | Low loss | U−H loss | L−U loss | L−H loss | H AUC | U AUC | L AUC |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A | 0 | 5.98590 | 6.07016 | 6.10281 | +0.08426 | +0.03266 | +0.11692 | 8.10619 | 8.20196 | 8.21991 |
| A | 1 | 5.97552 | 6.05090 | 6.09334 | +0.07538 | +0.04244 | +0.11782 | 8.08011 | 8.12383 | 8.25020 |
| A | 2 | 6.00096 | 5.96640 | 6.03513 | −0.03456 | +0.06873 | +0.03417 | 8.09676 | 8.05342 | 8.11078 |
| A | 3 | 5.96517 | 6.01268 | 6.04248 | +0.04751 | +0.02980 | +0.07731 | 8.09070 | 8.15449 | 8.19111 |
| A | 4 | 5.99368 | 6.10538 | 6.05746 | +0.11170 | −0.04792 | +0.06378 | 8.11812 | 8.20091 | 8.18024 |
| B | 0 | 5.94989 | 6.02404 | 6.06766 | +0.07415 | +0.04362 | +0.11777 | 8.02196 | 8.19283 | 8.25285 |
| B | 1 | 5.97654 | 5.97781 | 6.08604 | +0.00126 | +0.10823 | +0.10950 | 8.02745 | 8.05640 | 8.18790 |
| B | 2 | 5.91060 | 5.95660 | 5.99940 | +0.04600 | +0.04280 | +0.08880 | 8.02618 | 8.02237 | 8.05781 |
| B | 3 | 6.01518 | 6.03886 | 6.00793 | +0.02369 | −0.03093 | −0.00725 | 8.18715 | 8.14965 | 8.11845 |
| B | 4 | 6.00610 | 6.08816 | 6.06527 | +0.08205 | −0.02289 | +0.05916 | 8.15047 | 8.21472 | 8.16677 |
| C | 0 | 5.94417 | 5.98520 | 6.02842 | +0.04103 | +0.04323 | +0.08426 | 8.13681 | 8.14761 | 8.12013 |
| C | 1 | 5.97216 | 6.01290 | 6.09062 | +0.04074 | +0.07773 | +0.11847 | 8.01336 | 8.08130 | 8.21765 |
| C | 2 | 5.97738 | 5.99371 | 6.07107 | +0.01633 | +0.07736 | +0.09368 | 8.11044 | 8.00309 | 8.11199 |
| C | 3 | 5.89281 | 6.03310 | 6.08981 | +0.14029 | +0.05671 | +0.19700 | 8.07078 | 8.16784 | 8.16527 |
| C | 4 | 6.01579 | 6.05805 | 6.07675 | +0.04226 | +0.01870 | +0.06096 | 8.10243 | 8.19178 | 8.21090 |

The following means average **five model seeds inside each block**. Parenthesized sample SD is model-initialization variation within that block, and is kept separate from the across-block SD below.

| Block | High loss | Unfiltered loss | Low loss | U−H loss mean (seed SD) | L−U loss mean (seed SD) | L−H loss mean (seed SD) | U−H AUC mean (seed SD) | H beats U | U beats L |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A | 5.98424 | 6.04110 | 6.06625 | +0.05686 (0.05601) | +0.02514 (0.04363) | +0.08200 (0.03586) | +0.04855 (0.05501) | 4/5 | 4/5 |
| B | 5.97166 | 6.01709 | 6.04526 | +0.04543 (0.03387) | +0.02817 (0.05693) | +0.07360 (0.05053) | +0.04455 (0.08010) | 5/5 | 3/5 |
| C | 5.96046 | 6.01659 | 6.07134 | +0.05613 (0.04828) | +0.05474 (0.02488) | +0.11087 (0.05237) | +0.03156 (0.08467) | 5/5 | 5/5 |

| Across the three block means | Mean ± sample block SD | Positive blocks |
|:--|--:|--:|
| High 1M loss | 5.97212 ± 0.01190 | — |
| Unfiltered 1M loss | 6.02493 ± 0.01401 | — |
| Low 1M loss | 6.06095 ± 0.01382 | — |
| U−H 1M loss gain | **+0.05281 ± 0.00640** | 3/3 |
| L−U 1M loss penalty | +0.03602 ± 0.01629 | 3/3 |
| L−H 1M loss gap | +0.08882 ± 0.01955 | 3/3 |
| U−H 0–1M AUC gain | +0.04155 ± 0.00888 | 3/3 |
| U−H paired perplexity reduction | +21.43 ± 2.73 | 3/3 |

High beats unfiltered at the 1M endpoint in **14/15** paired model runs; low is worse than unfiltered in **12/15**. The one high-copy loss is A/seed 2 (−0.03456); B/seed 1 is nearly tied (+0.00126), and C/seed 3 has the largest gain (+0.14029). High beats unfiltered on AUC in **11/15** paired runs, so the curve result is less uniform than the endpoint. The block means are stable in sign but model initialization materially changes effect size. Three positive independent block means give a two-sided exact sign-test **p = 0.25**; these data support a robust directional pattern, not precise population inference.

At the shared language checkpoints, the across-block mean U−H loss gaps are **+0.03078** at zero TinyStories tokens, **+0.01982** at 251,904, **+0.05222** at 501,760, and **+0.05281** at 1,001,472. Thus part of the advantage is present immediately after synthetic pretraining, and the 1M endpoint gain is not just a zero-shot offset. These four points cannot establish a persistent final advantage at 10M.

## Tokens to a common validation loss

The target grid **10, 9, 8, 7, 6.25 loss** was fixed before the full block-C five-seed result. Each value below is `unfiltered TinyStories tokens / high-copy TinyStories tokens`, using the first downward linear crossing of the four saved 0–1M checkpoints. Values above 1 favor high. Every high/unfiltered pair crossed all five targets, so there is no missing-target extrapolation. [Exact interpolated tokens and all 75 paired ratios](horizontal_tokens_to_target.json); [read-only method](tokens_to_target_1m.py).

| Block | Seed | Loss 10 | Loss 9 | Loss 8 | Loss 7 | Loss 6.25 |
|:--|--:|--:|--:|--:|--:|--:|
| A | 0 | 1.079 | 1.034 | 1.023 | 1.045 | 1.033 |
| A | 1 | 0.974 | 0.988 | 1.009 | 1.048 | 1.032 |
| A | 2 | 0.966 | 0.989 | 0.991 | 0.979 | 0.986 |
| A | 3 | 1.050 | 1.029 | 1.021 | 1.039 | 1.021 |
| A | 4 | 1.015 | 1.021 | 1.028 | 1.067 | 1.044 |
| B | 0 | 1.113 | 1.059 | 1.055 | 1.126 | 1.045 |
| B | 1 | 1.032 | 1.016 | 1.010 | 1.016 | 1.003 |
| B | 2 | 0.962 | 0.990 | 1.000 | 1.014 | 1.016 |
| B | 3 | 0.948 | 0.977 | 0.987 | 0.990 | 1.005 |
| B | 4 | 1.013 | 1.008 | 1.015 | 1.042 | 1.031 |
| C | 0 | 1.053 | 1.014 | 0.995 | 0.986 | 1.008 |
| C | 1 | 1.033 | 1.023 | 1.024 | 1.059 | 1.023 |
| C | 2 | 0.915 | 0.949 | 0.957 | 0.920 | 0.990 |
| C | 3 | 1.029 | 1.028 | 1.030 | 1.073 | 1.053 |
| C | 4 | 1.057 | 1.029 | 1.028 | 1.053 | 1.021 |

| Target loss | A mean ± seed SD | B mean ± seed SD | C mean ± seed SD | Mean ± sample block SD |
|--:|--:|--:|--:|--:|
| 10 | 1.017 ± 0.048 | 1.014 ± 0.065 | 1.017 ± 0.059 | 1.016 ± 0.002 |
| 9 | 1.012 ± 0.022 | 1.010 ± 0.031 | 1.009 ± 0.034 | 1.010 ± 0.002 |
| 8 | 1.014 ± 0.015 | 1.013 ± 0.026 | 1.007 ± 0.031 | 1.011 ± 0.004 |
| 7 | **1.036 ± 0.033** | **1.038 ± 0.053** | **1.018 ± 0.064** | **1.031 ± 0.011** |
| 6.25 | 1.023 ± 0.022 | 1.020 ± 0.018 | 1.019 ± 0.023 | 1.021 ± 0.002 |

At the common loss-7 target, unfiltered needs **1.031×** the downstream tokens high-copy needs on average across block means, or high-copy uses about **3.0% fewer** TinyStories tokens. This is an approximate interpolation over sparse checkpoints, not a wall-time or total-compute speedup; both arms already consumed the full NCA stage. The endpoint loss difference (+0.05281) is the cleanest primary effect size. Perplexity reduction (+21.43 at 1M) is mathematically consistent but less stable as a headline because it depends on the high absolute loss scale. The token ratio is a useful secondary sample-efficiency reading across a predeclared grid, not a cherry-picked best crossing.

## Interpretation and remaining confounds

The fresh-arm feature audits show that high horizontal-copy selection also shifts **lag-2 cell recurrence** and **cell-transition mutual information** in the same direction across A/B/C. These features and horizontal copy each have 6/6 directional agreement with the seed-0 arm ordering; those six within-block contrasts are dependent, descriptive observations. Exact gzip-bin quotas limit coarse complexity differences but do not balance all token statistics or unmeasured rule properties. The failed one-seed high-MI selector does not identify which co-moving property caused high-copy transfer. [Predictor audit](TRANSFER_PREDICTOR_AUDIT.md).

**Strongest defensible claim:** within this fixed NCA generator, tokenizer, model and 40–50 gzip regime, selecting for a Transformer-visible recurrence property improved the 1M TinyStories endpoint relative to matched unfiltered NCA in all three fresh rule-population blocks. Selecting against that property generally degraded performance. This is evidence for a useful **data-selection policy**, not proof that literal horizontal patch copying is the uniquely causal mechanism. Lag-2 cell recurrence, transition MI and potentially unmeasured rule properties remain correlated with the selector. A fresh-block 10M continuation was not run, so persistent final quality is unresolved.

## Conclusions for the research blog

1. The base five-seed result supports synthetic pre-pretraining: ordinary 40–50 NCA improves scratch 10M loss by **0.137 ± 0.039** and reaches each seed's scratch@10M loss after **5.726 ± 0.685M** downstream tokens. That token ratio excludes NCA pretraining compute, so it is not total-compute speedup.
2. Shuffled NCA also beats scratch. The structured-versus-scratch result alone cannot isolate trajectory order; the incremental effect of structure is smaller.
3. The 40–50 gzip band has the lowest mean final loss among the tested bands, but wins as the best band in only **2/5** seeds. The data do not show a sharp Goldilocks optimum.
4. Low temporal cell change, high spatial agreement and high transition MI failed their one-seed 1M probes; high temporal change was mixed over five seeds. Simulator-level intuitions did not reliably predict transfer.
5. High horizontal recurrence is the strongest search lead: the fresh high-versus-unfiltered **1M loss gain is +0.05281 ± 0.00640 across three corpus-block means**, all positive; the 15 model seeds are within-block variations, not 15 corpus replicates.
6. Low-copy selection moved in the opposite direction on average in every fresh block (**+0.03602 ± 0.01629** loss penalty versus unfiltered). It is a useful directional control, although three individual seed pairs reverse.
7. The high-copy advantage is present at zero TinyStories tokens, remains at the 1M endpoint, and improves normalized 0–1M AUC in every block. A common loss-7 crossing uses about **3% fewer downstream tokens** under high selection, based on sparse-checkpoint interpolation.
8. The effect varies by initialization: one of 15 high-versus-unfiltered endpoints reverses and AUC favors high in 11/15. With only three independent corpus blocks, even 3/3 positive block signs have two-sided exact sign **p = 0.25**. The study supports a directional policy signal, not a uniquely identified mechanism or proven 10M final advantage.

## Later generator design — no run authorized

The separate [boosted-NCA dose-sweep protocol](../boosted_nca/SWEEP_PROTOCOL.md) tests stricter top-25/top-10 recurrence cohorts against the exact block-A top-50 anchor with one model seed and a fixed 1M tokens-to-target ranking. Its [canonical 10M configuration template](../boosted_nca/canonical_10m.template.json) preserves the original NCA budget, model, optimizer, TinyStories stream and milestones. The selected tier remains unset here; the single 10M stage resumes only the eventual winning 1M model after the sweep decision.
