# Horizontal recurrence dose sweep and 10M continuation

Completed 2026-10-08. This bounded recipe search extended the replicated top-50% horizontal-recurrence selector to fixed top-25% and top-10% cohorts. The [protocol](SWEEP_PROTOCOL.md) fixed selection thresholds, loss targets, ranking, and the conditional top-5 gate before either new English outcome. The [decision receipt](../../runs/nca-data-search/boosted-nca/decision.json) froze top-10; the same seed-0 model and optimizer were then resumed from 1M to 10M TinyStories tokens. No top-5, vertical, MI, lag-76, or mechanism arm ran.

**Subsequent validation:** The frozen top-10 corpus was trained with model seeds 1–4 through 10M, without further selection. The complete paired result is in [SEED_VALIDATION.md](SEED_VALIDATION.md): held-out top-10 convergence **1.998 ± 0.034x**, ordinary NCA **1.761 ± 0.238x** (mean ± sample SD over four model seeds).

## Controlled setup

The existing block-A high arm is the top-50 anchor. The two stricter cohorts reuse eligible worlds from that arm, then continue its rule stream with the same high-arm assignment and train/validation split hashes. Each has 19,975 training and 256 validation worlds, the original five 40–50% gzip sub-bin quotas, the original int32 packed tensor layout, and exactly 19,715,325 scored NCA labels. The [audit receipt](../../runs/nca-data-search/boosted-nca/sweep-a/audit.json) verifies these properties, score thresholds, and packing. Sampling filled both cohorts after 2,916,352 attempted rules. The top-25/top-10 cohorts share 8,121 rule seeds; their rule populations are deliberately overlapping, not independent blocks.

Both new cohorts used the same seed-0 initialization, full NCA stage, TinyStories stream, optimizer, and 10M learning-rate schedule. Only the first 1,001,472 TinyStories tokens were used to choose a tier. Each crossing below is the first downward **linear interpolation** between logged checkpoints at 0, 251,904, 501,760, and 1,001,472 tokens. These are approximate crossing times, not dense-curve measurements.

## Frozen 1M tier decision

Smaller crossing-token counts and smaller loss/AUC are better. The selection score is the geometric mean of the three predeclared target crossings, with a 1% conservative tie rule. The top-5 gate required top-25 to beat top-50 by at least 2% **at every target** and top-10 to beat top-25 by at least 2% **at every target**.

| Tier | Loss 7 tokens | Loss 6.5 tokens | Loss 6.25 tokens | Geometric mean | 1M loss | 0–1M AUC |
|:--|--:|--:|--:|--:|--:|--:|
| Top-50 anchor | 643.8K | 820.2K | 908.3K | 782.8K | 5.98590 | 8.10619 |
| Top-25 | 640.6K | 811.8K | 897.4K | 775.7K | 5.94591 | 8.13864 |
| **Top-10** | **627.5K** | **798.0K** | **883.3K** | **761.9K** | **5.90355** | **8.09986** |

Top-10 used **2.7% fewer** downstream tokens than the top-50 anchor on the selection score. All three target crossings improved, but top-25 improved by less than 2% at every target, so the top-5 gate **failed**. Top-10 won the frozen ranking; no top-5 corpus was generated. Top-25's AUC was worse than the anchor despite its better crossing times and 1M endpoint, illustrating why the conclusion is tied to the stated ranking metric. [Exact probe histories](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_0/downstream/probe_1m/metrics.json); [read-only ranking](rank_sweep.py).

## Resumed top-10 result

The chosen [seed-0 1M checkpoint](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_0/downstream/probe_1m/done.json) supplied the model **and optimizer** for the one [10M continuation](../../runs/nca-data-search/boosted-nca/final-10m/done.json). Scratch and ordinary 40–50% gzip NCA are the saved seed-0 historical controls from the original run, with the same TinyStories stream and evaluation points.

| TinyStories tokens | Scratch loss | Ordinary NCA loss | Top-10 loss |
|--:|--:|--:|--:|
| 0 | 10.98895 | 11.45375 | 11.47622 |
| 251,904 | 10.01959 | 9.57821 | 9.68460 |
| 501,760 | 8.47863 | 7.33914 | 7.36855 |
| 1,001,472 | 6.55394 | 5.98551 | 5.90355 |
| 2,000,896 | 5.57218 | 5.30078 | 5.13200 |
| 3,000,320 | 5.05797 | 4.80274 | 4.68434 |
| 5,001,088 | 4.53251 | 4.35077 | 4.31611 |
| 7,501,696 | 4.34195 | 4.19677 | 4.18312 |
| 10,000,000 | 4.31401 | 4.17338 | **4.15957** |

| Condition | 10M loss | 10M perplexity | Normalized 0–10M AUC |
|:--|--:|--:|--:|
| Scratch | 4.31401 | 74.74 | 5.15871 |
| Ordinary NCA | 4.17338 | 64.93 | 4.90801 |
| **Top-10 NCA** | **4.15957** | **64.04** | **4.85835** |

At the **fixed target** of scratch seed-0's 10M loss (**4.31401**), ordinary NCA crosses at **5.598M** and top-10 at **5.041M** TinyStories tokens, by first downward linear interpolation of the logged curve. Thus top-10 uses about **557K fewer language tokens**, or **9.9% fewer** than ordinary NCA, to reach this target. Relative to scratch's 10M tokens, the corresponding ratios are **1.786x** for ordinary NCA and **1.984x** for top-10. These count downstream language tokens only; NCA pretraining compute and wall time are excluded. The 1.984x value is **seed-0 specific** and should not be compared as if it were a five-seed mean against the earlier ~1.65x plot.

The designed condition is worse than ordinary NCA at 250K and 500K, then better from 1M onward. Its final ordinary-NCA loss edge is **0.01381**, smaller than its 2M edge (**0.16878**). In this run the main visible benefit is faster learning through the middle of the curve, with a small 10M endpoint edge.

## What this does and does not establish

The prior [45-run horizontal replication](../nca_search/BLOG_FINAL_ANALYSIS.md) supports the top-50 high-copy **direction** across three fresh corpus blocks, with corpus block treated as the independent generation unit. This tighter-dose **selection** used one block and one model seed, with many shared rule seeds across tiers. The later [seed validation](SEED_VALIDATION.md) checks four additional model initializations on that **same selected corpus**; it does not replicate top-10 selection across fresh rule populations. The historical ordinary-NCA control is not a fresh contemporaneous matched corpus. Interpolated crossings depend on sparse evaluation milestones. Horizontal copy remains correlated with transition MI, lag-2 recurrence, and potentially unmeasured rule properties; no unique causal-mechanism claim follows.

Sources: [top-25 probe](../../runs/nca-data-search/boosted-nca/sweep-a/top_25/seeds/seed_0/downstream/probe_1m/metrics.json), [top-10 probe](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_0/downstream/probe_1m/metrics.json), [winning 10M history](../../runs/nca-data-search/boosted-nca/final-10m/metrics.json), [original seed-0 scratch](../../runs/1m-nca-transfer-v2/seeds/seed_0/downstream/scratch/metrics.json), [original seed-0 ordinary NCA](../../runs/1m-nca-transfer-v2/seeds/seed_0/downstream/nca_40_50/metrics.json), [ranking code](rank_sweep.py), [10M comparison code](summarize_final.py).
