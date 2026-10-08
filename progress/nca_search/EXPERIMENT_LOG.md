# NCA data search

**Current scope (2026-10-08).** The horizontal A/B/C high/unfiltered/low × seeds 0–4 matrix completed all 45 runs; the runner exited. Its [final blog analysis](BLOG_FINAL_ANALYSIS.md) contains every paired endpoint/AUC, block means, seed sensitivity and predeclared tokens-to-target grid. The subsequently authorized bounded [horizontal-recurrence dose sweep](../boosted_nca/SWEEP_PROTOCOL.md), selected seed-0 10M continuation, and frozen top-10 corpus validation on model seeds 1–4 are complete. See [sweep results](../boosted_nca/SWEEP_RESULTS.md) and the [complete five-seed validation](../boosted_nca/SEED_VALIDATION.md). Across held-out seeds 1–4, downstream-token speedup to each seed's scratch@10M loss was **1.998 ± 0.034x** for top-10 versus **1.761 ± 0.238x** for ordinary NCA (sample SD); top-10 was faster in 4/4 pairs. This tests model-initialization sensitivity on one selected corpus, not fresh-corpus generalization. No vertical, MI, lag-76, matched-copy or mechanism branch was added. The phase stops here; earlier “Next” paragraphs and expansion gates below are historical decisions.

**Dose-sweep hypothesis and controlled change.** Increasing horizontal patch-copy excess beyond the validated global-median top-50 anchor may reduce TinyStories tokens needed to reach fixed validation losses; an extreme tail may instead saturate or reverse. Keep block A's original high arm and seed-0 model as the exact anchor. Select top-25/top-10 using frozen within-gzip-bin reference quantiles from that same high-arm rule stream, with original gzip quotas, world count, tokenizer, **19,715,325 NCA labels**, model initialization, optimizer and TinyStories schedule. [SWEEP_PROTOCOL.md](../boosted_nca/SWEEP_PROTOCOL.md) froze loss targets **7.0/6.5/6.25**, interpolation/ranking, top-5 gate and final-stage order before any top-25/top-10 English outcome. The [isolated sampler](../boosted_nca/sample_sweep.py) reused **10,207 top-25** and **4,053 top-10** qualifying worlds from the verified block-A high corpus, then continued its saved JAX rule cursor to fill both quotas. Both finished at **2,916,352 attempted rules**, passed the [complete corpus audit](../../runs/nca-data-search/boosted-nca/sweep-a/audit.json), and trained with one seed through 1M TinyStories tokens. Top-5's prespecified 2%-per-target monotonic gate failed, so no top-5 corpus or model was created. Top-10 won the fixed ranking and its saved seed-0 checkpoint was resumed to 10M. Its loss was **4.15957** versus **4.17338** for historical ordinary NCA seed 0 and **4.31401** for scratch seed 0. At the fixed scratch@10M loss, interpolated downstream tokens were **5.041M** for top-10 versus **5.598M** for ordinary NCA, corresponding to **1.984x** versus **1.786x** scratch-relative token ratios. This is a single selected corpus/model continuation, not fresh-block evidence of a general 10M advantage; synthetic pretraining compute is excluded.

**Horizontal closeout result.** High-copy beats contemporaneous unfiltered NCA at 1,001,472 TinyStories tokens in **14/15 model-seed pairs** and **all 3 corpus-block means**. Across the three block means, U−H validation-loss gain is **+0.05281 ± 0.00640** sample SD; low-copy has a **+0.03602 ± 0.01629** loss penalty versus unfiltered, also positive in all three block means. U−H normalized 0–1M AUC gain is **+0.04155 ± 0.00888** across blocks. At predeclared loss 7, first-crossing interpolation gives mean unfiltered/high downstream-token ratio **1.031 ± 0.011** across block means. The effect changes with model seed: A/seed 2 reverses the endpoint, and AUC favors high in only **11/15** pairs. Three positive independent blocks give two-sided exact sign **p=.25**; no fresh-block 10M final advantage or unique copy mechanism is established. Lag-2 cell recurrence, transition MI, and other rule properties remain possible correlated explanations. [Verified 45-arm table](BLOG_FINAL_ANALYSIS.md); [exact crossing data](horizontal_tokens_to_target.json).

**Vertical short-probe interruption.** Under the preceding instruction to screen a distinct property, the block-D vertical high seed-0 run was started. Its 19,715,325-label NCA stage finished and wrote a checkpoint; the TinyStories stage wrote only the zero-token initial validation/checkpoint. The user's subsequent stop instruction arrived during the first downstream update, and the process was interrupted immediately. No 100K/250K/500K vertical outcome exists, and the unfiltered/low vertical models were never started. The saved partial artifacts remain for provenance; they are not evidence of transfer and will not be resumed in this phase.

**Tokens-to-target grid frozen for closeout.** Before block C had a complete five-seed matrix (35/45 arm runs recorded), set target validation losses to **10, 9, 8, 7, and 6.25** for the existing four-point 0–1M curves. [Read-only interpolation](tokens_to_target_1m.py) uses each arm's first downward linear crossing, reports `unfiltered tokens / high tokens` for paired block/seed curves, never extrapolates, and averages model seeds within each corpus block before computing across-block mean and sample SD. Any uncrossed target is `NA` with its denominator shown. This grid was chosen from the shared observed curve range, not from the largest observed ratio; the final report will use all five targets and describe the sparse-checkpoint interpolation limit.

Reference: completed `runs/1m-nca-transfer-v2/`, structured `nca_40_50`, model seeds 0–4 where reported. This log records post-hoc controlled probes. The ~1M-parameter transformer, 19,715,325 usable NCA training labels, tokenizer, optimizer, and TinyStories stream remain fixed. A 1M downstream-token checkpoint is a cheap screen; it does not establish final 10M-token quality.

## Iteration 01 — temporal cell-change rate

**Hypothesis.** Among trajectories in the 40–50 gzip regime, lower frame-to-frame cell-change rates may produce more learnable local dynamics and better transfer than the unfiltered regime.

**Single sampling change.** Require the full 27-frame trajectory's mean cell-change fraction to be at or below 0.7086004273504274, the median of the existing 25,000-world 40–50 corpus. Keep the original NCA rule family, rule stream, frame sampling, gzip interval, and train/validation rule split. Match the reference train and validation world counts in five 0.02-wide gzip sub-bins, so coarse gzip composition and NCA token budget stay fixed.

**Reference probe.** Seed 0, `nca_40_50`, validation loss at the logged ~1M-token checkpoint (1,001,472 TinyStories tokens): 5.985511479899287. The candidate uses the same seed-0 base initialization and the same 10M-token downstream learning-rate schedule, stopping after that checkpoint.

**Sampling result.** Complete after 267,904 attempted rules. Candidate: 19,975 training and 256 validation worlds; 19,715,325 usable NCA labels. The five gzip sub-bin counts match the reference exactly. Mean cell-change fraction falls from 0.70335 in the matched reference selection to 0.61888 in the candidate, while mean gzip ratio is 0.45520 versus 0.45486. The candidate reuses 9,995/19,975 reference training rules and 124/256 reference validation rules; the remainder comes later in the same rule stream. The packed tokenizer output was checked against the original `pack_one` implementation. [Sampling receipt](../../runs/nca-data-search/iteration-01-low-change/done.json).

**Probe result.** NCA training completed at 19,715,325 usable labels. Final NCA validation loss: 6.03701 versus 6.03565 for the reference. At identical downstream checkpoint steps, token counts, and learning rates:

| TinyStories tokens | Reference validation loss | Low-change validation loss | Candidate − reference |
| ---: | ---: | ---: | ---: |
| 0 | 11.45375 | 11.37634 | -0.07742 |
| 251,904 | 9.57821 | 9.67999 | +0.10178 |
| 501,760 | 7.33914 | 7.42385 | +0.08470 |
| 1,001,472 | 5.98551 | 6.02008 | **+0.03457** |

At 1M, the lower-change sample is worse by 0.03457 validation-loss units (~3.5% higher perplexity). Trapezoidal validation-loss AUC over the four logged 0–1,001,472 token points, divided by that token span, is 8.13591 versus 8.07982 for the reference (higher is worse). The zero-token ordering reverses during language training. This one-seed screen **does not support** the lower-change hypothesis. It cannot separate a real property effect from corpus-sampling noise or predict 10M-token quality. [Candidate metrics](../../runs/nca-data-search/iteration-01-low-change/downstream/probe_1m/metrics.json); [completion marker](../../runs/nca-data-search/iteration-01-low-change/downstream/probe_1m/done.json).

**Next experiment.** Test the opposite, higher-change half of the 40–50 regime (change fraction ≥ the same reference median), with the same generator, five gzip sub-bin quotas, NCA token budget, model, seed-0 initialization, and 1M-token TinyStories probe. If high-change beats both the reference and low-change cohorts, temporal activity merits a paired multi-seed follow-up. If not, move to a different trajectory property rather than optimize this threshold.

## Iteration 02 — higher temporal cell-change rate

**Hypothesis.** The opposite side of the same measurable property may transfer better: higher frame-to-frame cell-change rates could provide richer transition signals within the fixed 40–50 gzip regime.

**Single sampling change.** Require mean cell-change fraction ≥ 0.7086004273504274. Use the same candidate rule stream, five reference gzip sub-bin quotas, split function, world counts, tokenizer, NCA token budget, seed-0 initialization, model, optimizer, TinyStories data and 1M-token probe schedule as iteration 01.

**Sampling result.** Complete after 266,048 attempted rules: 19,975 training and 256 validation worlds, 19,715,325 usable NCA labels, and the exact reference counts in all five gzip sub-bins. Mean cell-change fraction is 0.79453, versus 0.70335 in the matched reference and 0.61888 in iteration 01. Mean gzip ratio is 0.45548 versus 0.45520 in the reference. [Sampling receipt](../../runs/nca-data-search/iteration-02-high-change/done.json).

**Seed-0 probe result.** NCA training finished at the fixed label budget; candidate NCA validation loss is 6.16979 versus 6.03565 for the reference. The language curve at identical steps, token counts, and learning rates is:

| TinyStories tokens | Reference validation loss | High-change validation loss | Candidate − reference |
| ---: | ---: | ---: | ---: |
| 0 | 11.45375 | 11.40981 | -0.04394 |
| 251,904 | 9.57821 | 9.70600 | +0.12778 |
| 501,760 | 7.33914 | 7.43523 | +0.09608 |
| 1,001,472 | 5.98551 | 5.95815 | **-0.02736** |

The high-change cohort reaches a **better 1M endpoint** by 0.02736 validation-loss units (perplexity 386.90 versus 397.63), but its normalized trapezoidal 0–1M validation-loss AUC is **worse**: 8.13544 versus 8.07982. Both filtered cohorts lag the reference at 250k and 500k; high change only overtakes by 1M. This is a single-seed, partial-corpus-overlap screen and does **not** yet establish reliable or earlier sample-efficiency gain. [Candidate metrics](../../runs/nca-data-search/iteration-02-high-change/downstream/probe_1m/metrics.json); [completion marker](../../runs/nca-data-search/iteration-02-high-change/downstream/probe_1m/done.json).

**Paired replication at 1M.** The same packed high-change corpus was trained with original base initializations 0–4, the corresponding reference NCA runs, identical NCA budget, and matching TinyStories step, token, and learning-rate histories. Positive gain means lower candidate validation loss.

| Seed | Reference loss | High-change loss | Endpoint gain | Reference AUC | High-change AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.98551 | 5.95815 | +0.02736 | 8.07982 | 8.13544 |
| 1 | 6.01663 | 6.01436 | +0.00227 | 8.08148 | 8.16731 |
| 2 | 5.99324 | 6.02807 | -0.03483 | 8.01865 | 8.10551 |
| 3 | 6.02718 | 6.00585 | +0.02133 | 8.20033 | 8.13877 |
| 4 | 6.04374 | 6.04477 | -0.00103 | 8.10353 | 8.13219 |

Endpoint wins: **3/5**. Mean paired endpoint gain: **+0.00302 ± 0.02438** (mean ± sample SD), with an exploratory two-sided t interval of **[-0.02725, +0.03329]**. Mean candidate-minus-reference normalized 0–1M loss AUC: **+0.03908** (higher is worse); AUC wins: **1/5**. The endpoint signal is small, seed dependent, and does not show a consistent early-learning advantage. Do not promote high change to a full 10M run on this evidence. All outcomes use one selected corpus, so five model seeds do not measure uncertainty over candidate-corpus draws.

High-change NCA validation losses exceed the reference in every seed, but the validation corpora are filtered differently, so those absolute losses are **not a controlled measure of representation quality**. Downstream TinyStories validation is the shared comparison set.

**Next.** Test a distinct property: high spatial excess neighbor agreement relative to fixed framewise color counts. It is almost uncorrelated with gzip and temporal change in the saved 40–50 corpus. Use the same matching and 1M probe. [The saved-data hypothesis list](HYPOTHESES.md) also records transition mutual information and other candidates; these are hypotheses, not transfer findings.

## Iteration 03 — higher spatial excess neighbor agreement

**Hypothesis.** Spatially coherent local patterns beyond what color frequencies alone predict may supply repeated, learnable patch contexts to the NCA pre-pretraining transformer.

**Single sampling change.** Select worlds with periodic right/down neighbor agreement above the fixed-color-count random-neighbor expectation, at or above the median **-0.001144367464** measured on all 25,000 existing 40–50 worlds. Keep the original NCA rule stream, 40–50 gzip interval, five gzip sub-bin quotas, train/validation split and counts, NCA label budget, tokenization, architecture, optimizer, and downstream probe schedule. [Reference feature screen](screen_existing.py) (`--stride 1`).

**Sampling result.** Complete after 188,800 attempted rules: 19,975 train worlds, 256 validation worlds, 19,715,325 usable NCA labels, and exact counts in all five reference gzip sub-bins. Candidate mean spatial excess agreement is **+0.03324** versus **-0.00294** in the matched reference selection; mean gzip is 0.45522 versus 0.45520, and mean cell-change fraction is 0.70851 versus 0.70335. The candidate shares 10,103/20,231 rule seeds with that reference selection; the rest come later in the same rule stream. [Sampling receipt](../../runs/nca-data-search/iteration-03-high-spatial/done.json).

**Seed-0 probe result.** NCA training finished at the fixed 19,715,325-label budget. At exactly matched TinyStories steps, tokens, and learning rates:

| TinyStories tokens | Reference validation loss | High-spatial validation loss | Candidate − reference |
| ---: | ---: | ---: | ---: |
| 0 | 11.45375 | 11.48363 | +0.02987 |
| 251,904 | 9.57821 | 9.73786 | +0.15965 |
| 501,760 | 7.33914 | 7.47970 | +0.14055 |
| 1,001,472 | 5.98551 | 6.02467 | **+0.03915** |

Normalized trapezoidal 0–1M loss AUC is **8.18594**, versus **8.07982** for the reference (higher is worse). The candidate loses at every logged language checkpoint. This one-seed screen does **not support** the proposed high-spatial selector, though a single candidate corpus and initialization cannot rule out all spatial mechanisms. The NCA validation loss (6.00548 versus reference 6.03565) is measured on differently filtered validation sets and is not a controlled representation-quality comparison. [Candidate metrics](../../runs/nca-data-search/iteration-03-high-spatial/downstream/probe_1m/metrics.json); [completion marker](../../runs/nca-data-search/iteration-03-high-spatial/downstream/probe_1m/done.json).

**Next.** Test a more directly transformer-facing but largely independent property: higher horizontal adjacent-patch copy above fixed patch-frequency chance, as measured in [the patch-token screen](TRANSFORMER_TARGETS.md). Freeze its threshold from saved reference trajectories before generating new worlds. Continue the same gzip sub-bin matching and seed-0 1M probe.

## Iteration 04 — higher horizontal patch-copy excess

**Hypothesis.** Extra recurrence of adjacent patch tokens within a frame, beyond what the world's patch frequencies predict, may give the causal 256-token transformer a useful local dependency to learn.

**Single sampling change.** Accept worlds whose horizontal adjacent 2×2 patch-copy fraction minus the exact fixed-count random-pair probability is at or above **-0.000374227071**, the median from all 25,000 saved 40–50 worlds. The chance term is `sum_id n_id(n_id−1)/(972×971)`, and adjacent pairs exclude raster row boundaries. Match the original five gzip sub-bin counts, train/validation split, world count, NCA label budget, initialization, optimizer and downstream probe. The full reference threshold is reproducible with `uv run --all-packages python progress/nca_search/screen_transformer_targets.py --first 25000 --stride 1`.

**Sampling result.** Complete after 189,440 attempted rules: 19,975 train worlds, 256 validation worlds, 19,715,325 usable NCA labels, and exact reference counts in all five gzip sub-bins. Mean horizontal patch-copy excess rises from **+0.00274** in the matched reference selection to **+0.01163** in the candidate. Mean gzip is 0.45523 versus reference 0.45520; mean cell-change fraction is 0.70588 versus 0.70335. Rule-seed overlap with the reference selection is 10,135/20,231. The generator selector was checked against the read-only feature-screen implementation on a saved world. [Sampling receipt](../../runs/nca-data-search/iteration-04-high-patch-copy/done.json).

**Seed-0 probe result.** NCA training finished at the fixed 19,715,325-label budget. Matching TinyStories steps, tokens, and learning rates:

| TinyStories tokens | Reference validation loss | High-patch-copy validation loss | Candidate − reference |
| ---: | ---: | ---: | ---: |
| 0 | 11.45375 | 11.41568 | -0.03807 |
| 251,904 | 9.57821 | 9.65432 | +0.07611 |
| 501,760 | 7.33914 | 7.35131 | +0.01217 |
| 1,001,472 | 5.98551 | 5.92392 | **-0.06159** |

At 1M the candidate improves loss by 0.06159 (perplexity 373.87 versus 397.63), but normalized trapezoidal 0–1M loss AUC is 8.08329 versus reference 8.07982, effectively tied and slightly worse. This is an endpoint signal, not a consistent early-curve gain. [Candidate metrics](../../runs/nca-data-search/iteration-04-high-patch-copy/downstream/probe_1m/metrics.json); [completion marker](../../runs/nca-data-search/iteration-04-high-patch-copy/downstream/probe_1m/done.json).

**Paired 1M replication.** The same candidate corpus was pre-pretrained under original base initializations 0–4, then transferred with exactly matched downstream steps, token counts, and learning rates. Positive endpoint gain means lower candidate validation loss. AUC is the normalized trapezoidal validation-loss area from 0 to 1,001,472 TinyStories tokens.

| Seed | Reference 1M loss | Candidate 1M loss | Endpoint gain | Reference AUC | Candidate AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.98551 | 5.92392 | +0.06159 | 8.07982 | 8.08329 |
| 1 | 6.01663 | 5.96336 | +0.05327 | 8.08148 | 8.05389 |
| 2 | 5.99324 | 5.90230 | +0.09094 | 8.01865 | 8.02841 |
| 3 | 6.02718 | 6.01422 | +0.01297 | 8.20033 | 8.12330 |
| 4 | 6.04374 | 5.96667 | +0.07707 | 8.10353 | 8.06039 |

The candidate wins the **1M endpoint in 5/5 seeds**. Mean reference loss is 6.01326 and candidate loss 5.95409; mean paired gain **+0.05917 ± 0.02960** (mean ± sample SD). An exploratory two-sided t interval is **[+0.02241, +0.09592]**; the two-sided exact sign-test p value for five wins is **0.0625**. The normalized 0–1M AUC improves in 3/5 seeds; mean AUC gain is +0.02691, with exploratory t interval **[-0.01718, +0.07099]**. Mean milestone gains are +0.00775 at zero tokens, **-0.00161** at 251,904, +0.03094 at 501,760, and +0.05917 at 1,001,472. Thus the endpoint effect is more consistent than an immediate early-curve advantage.

This property was selected after several exploratory probes, so the interval and sign test are **descriptive, unadjusted** uncertainty summaries. Five model seeds on **one candidate corpus** do not test robustness to a new rule corpus. The [candidate audit](ITERATION_04_AUDIT.md) found close gzip matching and only small measured secondary feature shifts, but about half the rules differ; this is evidence for this selection policy in this corpus, not isolation of a causal patch-copy mechanism.

**Next.** Continue the same five downstream runs from their saved 1M optimizer states to the original 10M-token end under the unchanged schedule. This checks persistence across the full language curve. A later independent-corpus replication is needed before a general data-design claim.

**Completed 10M continuations.** All five runs resumed from their saved 1M optimizer states and match the reference at all nine logged steps, token counts, and learning rates. Positive gain means lower candidate loss; AUC is normalized trapezoidal validation-loss area over the complete 0–10M curve.

| Seed | Reference 10M loss | Candidate 10M loss | Final gain | Reference AUC | Candidate AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.17338 | 4.16865 | +0.00473 | 4.90801 | 4.87553 |
| 1 | 4.20435 | 4.17396 | +0.03040 | 4.94085 | 4.90204 |
| 2 | 4.15847 | 4.14839 | +0.01008 | 4.88361 | 4.85646 |
| 3 | 4.17271 | 4.18374 | -0.01102 | 4.91425 | 4.91618 |
| 4 | 4.21540 | 4.15638 | +0.05901 | 4.96293 | 4.87635 |

Final wins: **4/5**. Mean final loss is 4.16622 versus reference 4.18486, a paired gain **+0.01864 ± 0.02700** (mean ± sample SD); exploratory two-sided t interval **[-0.01488, +0.05216]**. Full-curve AUC improves in **4/5**; mean AUC gain **+0.03662**, exploratory interval **[-0.00311, +0.07634]**. The final and AUC intervals cross zero and are unadjusted for exploratory selection. Seed 3 reverses: it loses from 2M onward and ends 0.01102 worse.

| TinyStories tokens | Mean paired loss gain | Winning seeds |
| ---: | ---: | ---: |
| 0 | +0.00775 | 3/5 |
| 251,904 | -0.00161 | 2/5 |
| 501,760 | +0.03094 | 3/5 |
| 1,001,472 | +0.05917 | 5/5 |
| 2,000,896 | **+0.08371** | 4/5 |
| 3,000,320 | +0.05973 | 4/5 |
| 5,001,088 | +0.02955 | 4/5 |
| 7,501,696 | +0.01864 | 4/5 |
| 10,000,000 | +0.01864 | 4/5 |

The observed advantage is strongest around 1–3M TinyStories tokens and mostly narrows by 5–10M. This supports a **sample-efficiency hypothesis** for this selected corpus, with seed dependence and no independent-corpus confirmation. [Machine-readable full-curve summary](full_10m.csv).

## Iteration 05 — lower horizontal patch-copy excess

**Hypothesis.** If the iteration-04 selector's advantage is related to repeated adjacent patch tokens rather than chance corpus replacement, selecting the opposite half of the same 40–50 reference distribution should reduce or remove the 1M endpoint gain.

**Single sampling change.** Require horizontal patch-copy excess **≤ -0.000374227071**, the same frozen full-reference median used by iteration 04. Use the same rule stream, original generator, five gzip sub-bin quotas, split function, world counts, tokenizer, NCA budget, model and seed-0 downstream probe. This is a directional control, not an independent corpus replication.

**Sampling result.** Complete after 187,968 attempted rules: 19,975 train and 256 validation worlds, 19,715,325 usable NCA labels, and exact reference gzip sub-bin counts. Mean patch-copy excess is **-0.00617**, versus **+0.00274** in the matched reference selection and **+0.01163** in the high-copy cohort. Mean gzip is 0.45516 versus 0.45520 reference; mean cell-change fraction is 0.70231 versus 0.70335 reference. [Sampling receipt](../../runs/nca-data-search/iteration-05-low-patch-copy/done.json).

**Seed-0 probe result.** Completed at the same 19,715,325 NCA-label and 1,001,472 TinyStories-token checkpoints. At matched steps, tokens, and learning rates:

| TinyStories tokens | Reference loss | Low-copy loss | High-copy loss |
| ---: | ---: | ---: | ---: |
| 0 | 11.45375 | 11.50424 | 11.41568 |
| 251,904 | 9.57821 | 9.75439 | 9.65432 |
| 501,760 | 7.33914 | 7.58985 | 7.35131 |
| 1,001,472 | 5.98551 | **6.01197** | **5.92392** |

At 1M the low-copy cohort is 0.02645 worse than reference and 0.08805 worse than high-copy. Its normalized 0–1M AUC is **8.23073**, versus reference 8.07982 and high-copy 8.08329. This supports the expected directional ordering for **one model seed on three different corpora**, but it does not isolate copy as the causal feature. [Low-copy metrics](../../runs/nca-data-search/iteration-05-low-patch-copy/downstream/probe_1m/metrics.json).

**Next.** Repeat the low-copy 1M probe under original model seeds 1–4, then compare high, unfiltered, and low cohorts pairwise across all five seeds. Independent rule-corpus blocks remain the required next robustness test.

**Five-seed directional control.** The low-copy cohort finished the same NCA-label budget and 1M TinyStories probe for seeds 0–4. All histories have the reference steps, token counts, and learning rates.

| Seed | High-copy 1M loss | Unfiltered 1M loss | Low-copy 1M loss | Low − high |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.92392 | 5.98551 | 6.01197 | +0.08805 |
| 1 | 5.96336 | 6.01663 | 6.07536 | +0.11200 |
| 2 | 5.90230 | 5.99324 | 6.05246 | +0.15016 |
| 3 | 6.01422 | 6.02718 | 6.06478 | +0.05057 |
| 4 | 5.96667 | 6.04374 | 6.07548 | +0.10882 |

The **high < unfiltered < low** loss ordering holds in **5/5** model seeds. Mean losses are **5.95409, 6.01326, 6.05601** respectively. Mean low-minus-unfiltered loss is **+0.04275 ± 0.01533** (mean ± sample SD); mean low-minus-high is **+0.10192 ± 0.03642**. Exploratory two-sided paired t intervals are **[+0.02372, +0.06179]** and **[+0.05670, +0.14714]** respectively. Normalized 0–1M AUC means are **8.06985** high, **8.09676** unfiltered, **8.17802** low; high beats low on AUC in **5/5** seeds. These intervals are descriptive and unadjusted for the exploratory selection. All seeds still share the same three corpora and are not independent rule-corpus replicates. [Machine-readable comparison](patch_copy_direction.csv).

## Iteration 06 — higher cell-transition mutual information

**Hypothesis, frozen before sampling.** Within the original 40–50 gzip regime, NCA trajectories whose cell transitions contain more information about the next color may offer a learnable rule beyond literal patch copying. On the saved reference worlds, a world-fitted transition predictor improves held-out, transformer-visible patch prediction by 0.5845 bits per patch on average, but no downstream outcome has been measured for this selector. The model is not given the fitted table; transfer remains a hypothesis. [Read-only screen](TRANSITION_INFO_TARGET.md).

**Single sampling change.** Accept a world if full-trajectory first-order cell-transition mutual information is **≥ 0.141948363126 bits/cell**, the median from all 25,000 saved 40–50 reference worlds. Use the same 10-color transition table and entropy formula as the reference screen. Keep the existing rule stream, 27-frame rollout, five reference gzip sub-bin quotas, train/validation split and world counts, 2×2 tokenizer, packed tensor layout, NCA label budget, seed-0 initialization, optimizer, original full 10M TinyStories learning-rate schedule, and 1,001,472-token probe checkpoint. The frozen threshold has 49–51% of saved reference worlds above it in every sub-bin. This is a new corpus selection policy, not a causal isolation of mutual information: high-MI worlds in the reference also shift cell change and horizontal patch copy.

**Primary outcome.** Seed-0 TinyStories validation loss at 1,001,472 tokens minus the seed-0 unfiltered 40–50 reference loss **5.985511479899287**. Secondary outcome: normalized trapezoidal 0–1M validation-loss AUC at the four logged milestones. The result will be recorded whether it wins or loses; a single seed and candidate corpus are triage only. Sampling and probe commands are `uv run --all-packages python progress/nca_search/sample_low_change.py --transition-high` and `uv run --all-packages python progress/nca_search/probe_low_change.py --transition-high --seed 0`.

**Sampling and contract audit.** Complete after 191,936 attempted rules: 19,975 train and 256 validation worlds, 19,715,325 usable labels, and the exact reference counts in all five gzip sub-bins. Mean selected MI is **0.24563 bits/cell**; mean gzip is **0.455189** versus reference **0.455195**. The full empirical gzip CDF gap is **0.00400**. Rule overlap with the exact reference train/validation selection is **10,089/20,231**. A deterministic 1,017-world-per-corpus feature audit found MI **0.24222 versus 0.17201**, cell-change fraction **0.71091 versus 0.70424**, and horizontal patch-copy excess **0.00348 versus 0.00322**. Color entropy, rare-patch mass, and lag-2 recurrence also shift; they remain possible confounds. The candidate packed tensors have shape `(20,231, 1,026)`, int32 dtype, the same markers, target masks and split semantics, and sampled rows agree exactly with `pack_one`. [Sampling receipt](../../runs/nca-data-search/iteration-06-high-transition-mi/done.json); reproduce the audit with `uv run --all-packages python progress/nca_search/audit_iteration_04.py --candidate iteration-06-high-transition-mi`.

**Seed-0 1M probe result.** NCA training and the fixed-schedule TinyStories probe completed. Candidate and reference evaluation histories have identical step, token, and learning-rate positions:

| TinyStories tokens | Reference validation loss | High-MI validation loss | Candidate − reference |
| ---: | ---: | ---: | ---: |
| 0 | 11.45375 | 11.50610 | +0.05235 |
| 251,904 | 9.57821 | 9.74681 | +0.16860 |
| 501,760 | 7.33914 | 7.39640 | +0.05726 |
| 1,001,472 | 5.98551 | 6.04384 | **+0.05833** |

At 1M, high MI is worse by **0.05833** validation-loss units; perplexity is **421.51 versus 397.63**. Normalized trapezoidal validation-loss AUC over 0–1,001,472 tokens is **8.16462 versus 8.07982** (candidate worse by **0.08480**). The candidate loses at every logged checkpoint. This one-seed screen **does not support** the high-MI selection policy as a way to improve early English learning in this setup. A world-specific fitted transition table's held-out patch-prediction gain did not predict language transfer here. The shift in other selected world properties and the single model seed prevent a causal or general negative claim about transition information. [Candidate metrics](../../runs/nca-data-search/iteration-06-high-transition-mi/downstream/probe_1m/metrics.json); [completion marker](../../runs/nca-data-search/iteration-06-high-transition-mi/downstream/probe_1m/done.json).

**Next.** Prioritize independent rule-corpus replication of the iteration-04 high horizontal-patch-copy selection against unfiltered and low-copy arms under the frozen design in [REPLICATION_DESIGN.md](REPLICATION_DESIGN.md). Its five-seed positive 1M result and opposite low-copy control warrant this before screening more weakly motivated selectors. Begin with block A and model seed 0; a new corpus block, rather than another initialization on the same corpus, tests the main unresolved uncertainty. If this replication fails, return to distinct targets such as vertical lag-6 or visible lag-76 instead of tuning the high-copy threshold.

## Replication block A — frozen high-copy policy on fresh rules

**Hypothesis.** The 1M-token gain from selecting above-median horizontal patch-copy excess persists on an independent rule stream, against contemporaneous unfiltered and low-copy controls.

**Controlled change.** Use a new JAX rule stream seeded with **101**. Assign each attempted rule to `high`, `unfiltered`, or `low` using the pre-feature `blake2b` arm hash in [the frozen protocol](REPLICATION_DESIGN.md), then apply the original **−0.000374227071** patch-copy threshold only in the high/low arms. Each arm must independently meet the exact original five gzip-bin train/validation quotas and NCA label budget. The three arms will use the same original seed-0 base model, NCA optimizer and labels, and original TinyStories data, order, optimizer, schedule and 1M checkpoint.

**Primary outcome fixed before training.** Within-block seed-0 validation-loss gain at 1,001,472 TinyStories tokens: `unfiltered − high`. Secondary comparisons: `low − high`, `unfiltered − low`, and normalized 0–1M validation-loss AUC. A single fresh corpus block is a robustness check, not complete three-block confirmation. No training starts until the chunk/rule-index audit confirms disjoint rule IDs and exact data contract. [Sampler](sample_patch_replication.py); [pretraining audit](audit_patch_replication.py).

**Sampling and pretraining audit.** All three arms completed after 577,920 attempted rules, each with 19,975 training and 256 validation worlds, the exact five gzip-bin quotas, and 19,715,325 usable NCA labels. Mean gzip ratios are **0.455141 high**, **0.455141 unfiltered**, **0.455177 low**. Mean horizontal patch-copy excess separates as intended: **+0.011816**, **+0.002712**, **−0.006221**. The audit reconstructed the 60,693 selected rule IDs from the saved JAX stream, verified pre-feature arm/split hash assignments, threshold directions and quotas, and found **zero overlap** with earlier receipts/search corpora. All three packed files passed int32 shape `(20,231,1,026)`, marker, mask and split checks; the sampler checked `pack_one` on first/last worlds. [Audit receipt](../../runs/nca-data-search/patch-replication/block-a/audit.json). This is one fresh corpus block, still subject to chance corpus variation and any other properties shifted by selection.

**Seed-0 1M result.** All three arms completed the original NCA label budget and 1,001,472-token TinyStories probe. Logged steps, tokens, and learning rates match exactly across arms. Lower validation loss is better:

| TinyStories tokens | High copy | Unfiltered | Low copy |
| ---: | ---: | ---: | ---: |
| 0 | 11.39899 | 11.53942 | 11.41295 |
| 251,904 | 9.61558 | 9.72077 | 9.73664 |
| 501,760 | 7.40274 | 7.48486 | 7.54293 |
| 1,001,472 | **5.98590** | 6.07016 | 6.10281 |
| Normalized 0–1M loss AUC | **8.10619** | 8.20196 | 8.21991 |

The prespecified within-block endpoint contrast `unfiltered − high` is **+0.08426** validation-loss units, and `low − high` is **+0.11692**. High-copy also has lower 0–1M loss AUC than both controls; its AUC gain against unfiltered is **+0.09577**. At 1M, perplexities are **397.78 high**, **432.75 unfiltered**, **447.11 low**. The same directional 1M ordering as the discovery corpus therefore appears in this one fresh rule block. This is **not yet three-block confirmation** or a causal isolation of patch copying. The fresh unfiltered arm scores 6.07016, worse than the original reference seed-0 score 5.98551; the fresh high arm scores 5.98590, essentially tying that old reference. This illustrates meaningful corpus-draw variation and why the prespecified contrast is within each block. [Machine-readable results](replication_1m.csv) and [source metrics](../../runs/nca-data-search/patch-replication/block-a/high/seeds/seed_0/downstream/probe_1m/metrics.json).

**Secondary feature audit.** A deterministic sample of 1,017 worlds per arm, matched in count within each split and gzip sub-bin, gives high/unfiltered/low means of **+0.01156/+0.00239/−0.00620** for horizontal patch-copy excess. High versus unfiltered also shifts cell change by **+0.00904**, transition MI by **+0.00659 bits**, and lag-2 cell recurrence by **+0.01514**; patch entropy shifts only **+0.00511 bits** and same-token repeats within a 256-token context shift **−0.00159**. These are descriptive samples, not independently randomized feature manipulations. The observed language gain remains a property of the high-copy *selection policy*, with correlated dynamics as possible mechanisms. [Reproducible feature screen](screen_replication_arms.py); [saved audit](../../runs/nca-data-search/patch-replication/block-a/feature_audit.json).

**Next.** Complete the already frozen B/C rule blocks and their seed-0 three-arm comparisons before deciding whether the high-copy policy replicates across corpus draws. Only then apply the staged model-seed and full-curve gates in [REPLICATION_DESIGN.md](REPLICATION_DESIGN.md); do not select a favorable block after seeing outcomes.

**Blocks B/C sampling audit (outcomes pending).** The frozen streams **202** and **303** yielded complete three-arm corpora after 572,352 and 576,960 attempted rules respectively. Each of the six arms has 19,975 training worlds, 256 validation worlds, exact five-bin gzip quotas, and 19,715,325 usable NCA labels. Full rule-index audits found **zero prior or cross-block rule overlap**. All six packed tensors passed the original int32 shape, marker, mask and split checks. Mean patch-copy excess is high/unfiltered/low **+0.011769/+0.002753/−0.006172** in B and **+0.011712/+0.002567/−0.006189** in C; mean gzip ratios range from 0.455130 to 0.455266. [B audit](../../runs/nca-data-search/patch-replication/block-b/audit.json); [C audit](../../runs/nca-data-search/patch-replication/block-c/audit.json).

The deterministic 1,017-world-per-arm feature screens show high-minus-unfiltered patch-copy excess shifts of **+0.00916/+0.00934/+0.01037** in A/B/C. Cell-change shifts are **+0.00904/+0.00275/−0.00683** and patch-entropy shifts are **+0.00511/−0.01099/−0.02361**; their signs are not stable. Transition-MI shifts **+0.00659/+0.01162/+0.00656 bits** and lag-2 cell-recurrence shifts **+0.01514/+0.02367/+0.02022** are consistent secondary changes. Thus repeated downstream gains would weigh against a simple cell-change or patch-entropy explanation, but would still leave MI and two-step recurrence as possible mediators. These are descriptive feature audits, not causal mediation tests. [Feature screen](screen_replication_arms.py).

**Block B seed-0 1M result.** All three arms completed the same NCA label and TinyStories budgets; their logged steps, tokens and learning rates match. At 1,001,472 tokens, validation loss is **5.94989 high < 6.02404 unfiltered < 6.06766 low**. The prespecified `unfiltered − high` gain is **+0.07415** and `low − high` is **+0.11777**. Normalized 0–1M loss AUC is **8.02196 high**, **8.19283 unfiltered**, **8.25285 low**; high beats unfiltered by **+0.17088** AUC units and wins at every logged checkpoint. This is a second positive independent rule block under model seed 0, still not a completed three-block result. [Machine-readable results](replication_1m.csv); [source metrics](../../runs/nca-data-search/patch-replication/block-b/high/seeds/seed_0/downstream/probe_1m/metrics.json).

**Block C seed-0 1M result.** Under the same budgets and matched logged steps/tokens/learning rates, endpoint validation loss is **5.94417 high < 5.98520 unfiltered < 6.02842 low**. The within-block `unfiltered − high` gain is **+0.04103** and `low − high` is **+0.08426**. The early curve is less consistent than in A/B: high loses to unfiltered at zero tokens by 0.00062 and at 501,760 tokens by 0.05622, but wins at 251,904 by 0.08655 and at 1,001,472 by 0.04103. Normalized 0–1M AUC is **8.13681 high**, **8.14761 unfiltered**, **8.12013 low**. Thus high's AUC gain against unfiltered is only **+0.01080**, and low has better AUC than high despite its worse endpoint. [Machine-readable results](replication_1m.csv); [source metrics](../../runs/nca-data-search/patch-replication/block-c/high/seeds/seed_0/downstream/probe_1m/metrics.json).

**Three-block seed-0 decision.** The prespecified high-copy selector beats the contemporaneous unfiltered arm at 1M in **3/3 independent rule blocks** and beats low-copy in **3/3**. Block-level unfiltered-minus-high endpoint gains are **+0.08426, +0.07415, +0.04103**: mean **+0.06648 ± 0.02261** (mean ± sample SD), range **+0.04103 to +0.08426**. AUC gains are **+0.09577, +0.17088, +0.01080**: mean **+0.09248 ± 0.08009**. High beats unfiltered in 3/3 blocks at 251,904 and 1,001,472 tokens, but only 2/3 at zero and 501,760. A two-sided exact sign test for 3/3 wins gives **p = 0.25**; these three blocks and exploratory discovery do not justify a confirmatory significance claim. The result supports a promising **high-copy corpus-selection policy** for this fixed NCA/transformer/TinyStories setup, while MI and lag-2 recurrence remain possible correlated mechanisms. The frozen [replication design](REPLICATION_DESIGN.md) now calls for original model initializations 1–4 on all three arms in all blocks, followed by a 10M curve only if the model-seed pattern persists. [Machine-readable nine-run table](replication_1m.csv).

**Model-seed follow-up started.** The [resume-safe runner](run_patch_replication_model_seeds.sh) verified the three-block gate and is running original model initializations 1–4 sequentially for all three arms in all three fixed corpora. It appends only completed runs to [the machine-readable table](replication_1m.csv). These extra initializations measure sensitivity to model seed within each corpus block; they do not turn the 3 independent corpus blocks into 15 independent corpus observations.

**Interim block A, model seed 1.** The first complete extra initialization orders 1M validation loss **5.97552 high < 6.05090 unfiltered < 6.09334 low**. Within-block unfiltered-minus-high gain is **+0.07538**; 0–1M AUC is **8.08011 high versus 8.12383 unfiltered**, gain **+0.04373**. This is a second model initialization on the *same* block A corpus, not a fourth independent corpus. The sequential follow-up is still running and this interim result does not replace the prespecified all-block analysis.

**Interim block A, model seed 2.** This initialization reverses the high-versus-unfiltered result: 1M losses are **5.96640 unfiltered < 6.00096 high < 6.03513 low**. `unfiltered − high` is **−0.03456** at the endpoint and **−0.04335** for 0–1M loss AUC; `low − high` remains **+0.03417** at the endpoint. Across block A seeds 0–2, the unfiltered-minus-high endpoint gain averages **+0.04169** with **2/3** wins (AUC mean **+0.03205**, also 2/3 wins). This is model-seed sensitivity on one corpus; blocks B/C still have only seed 0, so no all-block model-seed conclusion is available. The resume-safe follow-up runner was restarted after an interrupted process and continues with seed 3, without changing the frozen protocol or completed run artifacts. [Completed-run table](replication_1m.csv).

**Interim block A, model seed 3.** The 1M ordering returns to **5.96517 high < 6.01268 unfiltered < 6.04248 low**. Unfiltered-minus-high is **+0.04751** at the endpoint and **+0.06379** for 0–1M AUC. Across block A seeds 0–3, high beats unfiltered in **3/4** endpoint and AUC comparisons; mean gains are **+0.04315** endpoint and **+0.03999** AUC. High beats low in **4/4**. These four initializations remain one corpus block, and B/C model-seed follow-up is incomplete. [Completed-run table](replication_1m.csv).

**Block A five-seed follow-up complete.** Seed 4 ends **5.99368 high < 6.05746 low < 6.10538 unfiltered**; its unfiltered-minus-high endpoint gain is **+0.11170**, while low beats unfiltered for this initialization. Across all five model seeds on the same block A corpus, high beats unfiltered at 1M in **4/5** and low in **5/5**. Mean unfiltered-minus-high endpoint gain is **+0.05686 ± 0.05601** (mean ± model-seed SD), and normalized 0–1M AUC gain is **+0.04855 ± 0.05501**, also 4/5 wins. Seed 2 remains a real high-versus-unfiltered counterexample. These are model-init sensitivities for one independently drawn corpus, not five independent corpus replicates; B/C seeds 1–4 are still running. [Completed-run table](replication_1m.csv).

**Interim block B, model seed 1.** The second model initialization in block B gives 1M losses **5.97654 high < 5.97781 unfiltered < 6.08604 low**. High's unfiltered-minus-high endpoint gain is only **+0.00127**, a near tie, while low-minus-high is **+0.10950**. Its normalized 0–1M AUC gain against unfiltered is **+0.02895**. Across completed B seeds 0–1, mean unfiltered-minus-high endpoint gain is **+0.03771** and AUC gain **+0.09991**, both 2/2 positive. This is interim model-seed sensitivity in one corpus block; B seeds 2–4 and C seeds 1–4 remain.

**Interim block B, model seed 2.** The 1M endpoint is **5.91060 high < 5.95660 unfiltered < 5.99940 low**. Unfiltered-minus-high gain is **+0.04600** at 1M, but the normalized 0–1M AUC gain is **−0.00381**: high is slightly worse on the integrated early curve. Across B seeds 0–2, mean endpoint gain is **+0.04047** with **3/3** endpoint wins; mean AUC gain is **+0.06534** with **2/3** wins. This supports an endpoint trend while showing the speedup is not uniform across seeds. B seeds 3–4 and C seeds 1–4 remain.

**Interim block B, model seed 3.** The 1M losses are **6.00793 low < 6.01518 high < 6.03886 unfiltered**. Unfiltered-minus-high is **+0.02369** at the endpoint, while low-minus-high is **−0.00725**. The normalized 0–1M AUC ordering is **8.11845 low < 8.14965 unfiltered < 8.18715 high**, giving unfiltered-minus-high **−0.03750**. Across B seeds 0–3, high beats unfiltered at 1M in **4/4**, mean gain **+0.03628**, but its AUC wins are **2/4**, mean gain **+0.03963**. This is another counterexample to a uniform early-learning gain; B seed 4 and C seeds 1–4 remain. [Completed-run table](replication_1m.csv).

**Block B five-seed follow-up complete.** Seed 4 ends **6.00610 high < 6.06527 low < 6.08816 unfiltered**, with unfiltered-minus-high endpoint gain **+0.08205** and AUC gain **+0.06425**. Across B seeds 0–4, high beats unfiltered at 1M in **5/5**, mean gain **+0.04543 ± 0.03387** (mean ± model-seed SD); its 0–1M AUC gain is **+0.04455 ± 0.08010**, with **3/5** wins. High beats low at the endpoint in **4/5**. These are five initializations of one block-B corpus, not five independent corpora. Block C seeds 1–4 remain before the three-block gate is decided. [Completed-run table](replication_1m.csv).

**Parallel read-only target screens.** [New transfer targets](NEW_TRANSFER_TARGETS.md) ranks visible vertical lag-6 patch copying first among currently untested selectors because it is mostly distinct from horizontal copying in the saved-world screen. A held-out horizontal boundary predictor is a possible noncopy target but has a large gzip shift under a crude split. [Visible noncopy successor reuse](SUCCESSOR_REUSE_TARGET.md) has residual variation after a frequency correction, yet still tracks gzip and bigram repetition, so it ranks lower. These are world-level diagnostics, not English-transfer outcomes. A [prospective controlled-copy protocol](CONTROLLED_COPY_PROTOCOL.md) specifies how to balance lag-2 recurrence and transition MI in a later mechanism test; no new corpus or probe has been run for that protocol.

**Next candidate frozen without a transfer outcome.** The [vertical lag-6 protocol](VERTICAL_LAG6_FROZEN_PROTOCOL.md) scanned all 25,000 saved 40–50 worlds and fixed the frame-adjusted vertical patch-copy median at **−0.0005878894767783627**. Both sides occur in every gzip sub-bin. Its high/low reference split changes mean vertical excess by **+0.01626** while horizontal excess changes **+0.00158**; visible lag-76 recurrence also changes **+0.00699**, so the proposed selector is distinct from horizontal copying but not mechanistically isolated. New rule streams, arm assignment, quotas, audits, and the fixed 1M probe gate are frozen in that protocol. No vertical trajectories or models have been run.

**Vertical implementation readiness.** The [sampler](sample_vertical_lag6.py), [audit](audit_vertical_lag6.py), and small [fixture check](test_vertical_lag6.py) now implement that frozen selection. The fixture, Python compilation, and both CLI help paths pass. Independent review fixed per-arm resume behavior and verifies logged overlap rejections. These are implementation checks only; the actual corpus audit requires completed sampled data, and the vertical 1M transfer probe remains gated on the active horizontal-replication decision.

**Vertical block D sampling started.** With the cutoff, rule stream 404, arm hash, quota table, and code hashes fixed in advance, the low-priority [block-D sampler](sample_vertical_lag6.py) is filling the three new arms alongside the existing horizontal model-seed follow-up. This is data sampling only; no vertical NCA or TinyStories model has started. The [sampling receipt](../../runs/nca-data-search/vertical-lag6/block-d/audit.json) and eventual [audit](audit_vertical_lag6.py) must establish completion and data validity before any 1M interpretation. Sampling in parallel changes no horizontal run files or protocol settings.

**Vertical block D sampling and contract audit complete.** All three arms filled the exact 19,975 training and 256 validation worlds and five gzip-bin quotas after **588,864** attempted rules; each supplies **19,715,325** usable NCA labels. The independent [audit receipt](../../runs/nca-data-search/vertical-lag6/block-d/audit.json) reconstructed **60,693** selected rule IDs, found zero overlap with earlier corpora, verified the arm/split stream and logged overlap rejections, and checked packed int32 layout, markers, masks, split indices, hashes and sampled `pack_one` agreement. Mean gzip is **0.455220 high / 0.455164 unfiltered / 0.455194 low**. Mean all-pair vertical excess separates **+0.009860 / +0.001789 / −0.006328**. No vertical model has run.

**Vertical block D feature audit complete before training.** The deterministic [feature audit](VERTICAL_LAG6_BLOCK_D_FEATURE_AUDIT.md) samples 1,017 worlds per arm in equal split/bin counts. Its actual model-visible vertical excess is **+0.010034 high / +0.001322 unfiltered / −0.006210 low**, satisfying the predeclared high > unfiltered > low manipulation check; all-pair excess orders the same. The full-corpus high-versus-unfiltered mean gzip difference is only **+0.000056**, with a maximum .005-grid CDF gap **0.00405**. Horizontal-copy excess shifts **+0.000703**, while visible lag-76 recurrence **+0.004432**, lag-2 cell recurrence **+0.018515**, and transition MI **+0.006735 bits** also move. A later language gain can support this vertical selection policy, but cannot isolate lag-6 attention from those temporal correlates. The 1M model probe has not run.

**Vertical probe path reviewed.** The [1M probe](probe_vertical_lag6.py) keeps the original model, base initialization, optimizer schedule, 19,715,325 NCA labels and 1,001,472 TinyStories checkpoint. Its audit gate now binds each packed arm by SHA-256. A separate [read-only diagnostic](diagnose_vertical_lag6_nca.py) scores NCA validation targets whose lag-6 predecessor is in the actual 256-token input; the mask fixture and one existing reference world both give **765/810 = 94.44%** visible and scored vertical pairs. Earlier read-only screens used a stricter window boundary and reported 762/810; the frozen selector score and cutoff are unchanged. No vertical model has run yet.

**Rule-level coupling screen.** A read-only [5,000-world screen](RULE_COUPLING_SCREEN.md) reconstructed saved random convolutional rules and measured local neighbor-versus-center sensitivity. Its rank association with horizontal patch-copy excess is **−0.013**, and with a held-out one-neighbor cell-prediction gain **+0.049**. This does not support selecting stronger raw neighbor coupling as the next expensive probe; the already frozen vertical token-level test has the clearer immediate rationale. These are source-world correlations, not language-transfer outcomes.

**Search stopping rule documented.** [STOPPING_RULE.md](STOPPING_RULE.md) fixes the remaining decision sequence: finish A/B/C seeds 0–4, use block-level endpoint and AUC gates for any 10M continuation, test matched copying to separate its measured temporal correlates, and advance vertical lag-6 beyond block D only through its frozen three-arm 1M gate. The rule was written after block A seeds 0–4 and block B seed 1 were known, so it is not a preregistration for those outcomes. After these finite branches, report a supported selection principle or an inconclusive result; do not keep adding post-hoc thresholds indefinitely.

**Next mechanism test, pending the model-seed gate.** A read-only [matching feasibility screen](PATCH_COPY_MECHANISM_FEASIBILITY.md) found that within gzip bins, matching on lag-2 recurrence and transition-MI terciles retains 10,781 high/low pairs from the 25,000 saved reference worlds (86.2% per arm) while preserving a +0.01681 copy-excess gap. This makes a future prospectively balanced high/low corpus plausible, though the full 19,975-per-arm quotas still require fresh generation and a frozen attempt cap. No such matched corpus or downstream probe has been run; the current model-seed follow-up takes precedence.

**Matched-copy preparation frozen before new data.** The [implementation](MATCHED_COPY_IMPLEMENTATION.md) fixes all 45 gzip × lag-2 × transition-MI cell quotas from the 25,000 saved reference worlds, a 2,000,000-attempt cap per new stream, and independent stream seeds 707/808/909. The frozen quota-file SHA-256 is `26e209aa1ba4f20fe14180e07d950a0cce717d4d7081b73ee94506c9b9507167`. The sampler and independent audit passed focused fixtures and compilation; no matched trajectory or transfer model has run. Exact sparse-cell filling and full-corpus nuisance balance remain unverified until new data exist.

**Horizontal 10M continuation prepared, not started.** The [read-only gate](gate_patch_replication_10m.py) requires all 45 A/B/C × arm × model-seed 1M receipts and the block-level endpoint/AUC criteria in [STOPPING_RULE.md](STOPPING_RULE.md). At preparation time it reported **30/45** complete and exited `NOT READY`. The [continuation](continue_patch_replication_10m.py) is designed to resume each saved 489-step model, optimizer and scheduler under the original 10M TinyStories order; compilation and shell syntax checks passed. This is a prepared path, not evidence of a 10M result.

**10M resume path reviewed before launch.** Independent code review found that the prepared continuation initially saved only at evaluation milestones. It now also saves every original 500-step checkpoint and accepts a resumed checkpoint whose last evaluation row is older than its optimizer step. The read-only gate still refused launch at **33/45** completed 1M runs. Syntax and whitespace checks passed; no 10M model has run, so numerical replay has not yet been observed.

**Vertical block D transfer readiness checked without training.** Calling only `setup_config()` for seed 0 in all three arms passed the real block-D audit, rule-index, packed-file SHA-256 and reference-checkpoint gates and wrote matching probe configs. No NCA or TinyStories model was run in this check. The first vertical 1M probe awaits an available MPS slot after the live horizontal follow-up.

**Next.** Finish the active A/B/C model-seed follow-up and apply the fixed block-level gate. Then run only the matched-copy and vertical branches admitted by [STOPPING_RULE.md](STOPPING_RULE.md), keeping negative and incomplete results visible.
