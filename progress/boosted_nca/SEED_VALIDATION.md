# Frozen top-10 corpus: model-seed validation

Completed 2026-10-08. The top-10 corpus and operating point were selected using seed 0. Seeds 1–4 use the **same audited top-10 corpus** and each seed's original base initialization. All five models received 19,715,325 NCA labels and 10M TinyStories tokens under the original schedules. Only seed 0 participated in tier selection; seeds 1–4 are held-out **model initializations**, not fresh corpus blocks.

For each seed, the fixed target is its own scratch validation loss at 10M. Crossing tokens use the first downward linear interpolation of the nine saved evaluation points. Speedup = 10M / crossing tokens; it counts downstream TinyStories tokens only.

## Downstream convergence speedup by synthetic pretraining distribution

![Mean paired-seed downstream convergence speedup, with sample SD whiskers, for the four ordinary NCA gzip bands and Designed NCA](figures/downstream_convergence_speedup_by_distribution.png)

| Synthetic pretraining distribution | Speedup, mean ± sample SD over five seeds |
|:--|--:|
| NCA 20–30% gzip | 1.642 ± 0.212x |
| NCA 30–40% gzip | 1.695 ± 0.059x |
| NCA 40–50% gzip | **1.766 ± 0.206x** |
| NCA 50%+ gzip | 1.559 ± 0.148x |
| **Designed NCA** (top-10% horizontal recurrence) | **1.995 ± 0.030x** |

All five bars use the same paired per-seed scratch target and crossing calculation; [plot data](figures/downstream_convergence_speedup_by_distribution.json), [SVG](figures/downstream_convergence_speedup_by_distribution.svg), and [reproduction script](plot_convergence_distributions.py). The 40–50% band has the highest observed ordinary-NCA mean, although the earlier five-seed band comparison did not establish a sharp gzip optimum. Our original [aggregate plot](../../runs/1m-nca-transfer-v2/plots/english_nca_vs_scratch_five_seeds.png) averaged all four structured NCA regimes first and gave **1.646x** from a mean-curve crossing. That is a different aggregation and is not used in the bars or the design comparison.

| Seed | Scratch target loss | Ordinary 10M loss | Top-10 10M loss | Ordinary crossing | Top-10 crossing | Ordinary speedup | Top-10 speedup | Top-10 − ordinary speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 4.31401 | 4.17338 | 4.15957 | 5.598M | 5.041M | 1.786x | 1.984x | 0.198x |
| 1 | 4.31728 | 4.20435 | 4.13766 | 6.159M | 4.990M | 1.624x | 2.004x | 0.380x |
| 2 | 4.32128 | 4.15847 | 4.15434 | 5.219M | 4.970M | 1.916x | 2.012x | 0.096x |
| 3 | 4.35562 | 4.17271 | 4.15809 | 4.990M | 4.934M | 2.004x | 2.027x | 0.023x |
| 4 | 4.30064 | 4.21540 | 4.13888 | 6.662M | 5.130M | 1.501x | 1.949x | 0.448x |

## Mean ± sample SD across model seeds

| Seeds | Ordinary speedup | Top-10 speedup | Paired speedup difference | Top-10 faster count |
|:--|--:|--:|--:|--:|
| All 5 (seed 0 selected) | 1.766 ± 0.206x | 1.995 ± 0.030x | 0.229 ± 0.182x | 5/5 |
| Held-out seeds 1–4 | 1.761 ± 0.238x | 1.998 ± 0.034x | 0.237 ± 0.209x | 4/4 |

The held-out mean is **1.998 ± 0.034x** for top-10 versus **1.761 ± 0.238x** for ordinary NCA. This supports an approximately 2x downstream-token convergence result across these held-out model initializations on the selected corpus, with no claim of fresh-corpus replication.

## Endpoint and curve checks

| Seed | Ordinary 10M PPL | Top-10 10M PPL | Ordinary 0–10M AUC | Top-10 0–10M AUC | Ordinary − top-10 10M loss |
|--:|--:|--:|--:|--:|--:|
| 0 | 64.93 | 64.04 | 4.90801 | 4.85835 | +0.01382 |
| 1 | 66.98 | 62.66 | 4.94085 | 4.85274 | +0.06669 |
| 2 | 63.97 | 63.71 | 4.88361 | 4.82940 | +0.00412 |
| 3 | 64.89 | 63.95 | 4.91425 | 4.89373 | +0.01462 |
| 4 | 67.72 | 62.73 | 4.96293 | 4.85109 | +0.07652 |

## Interpretation limits

The held-out comparison tests sensitivity to model initialization on one selected corpus. It does not independently validate the top-10 selection across fresh NCA rule populations. Seed 0 was used to select the tier, so the all-five-seed mean is descriptive and the seeds 1–4 row is the cleaner check. Crossings are interpolated between sparse milestones, and NCA pretraining compute is excluded. No unique horizontal-copy mechanism is identified; correlated transition MI, lag-2 recurrence, and other rule properties remain possible.

Sources: [frozen decision](../../runs/nca-data-search/boosted-nca/decision.json), [corpus audit](../../runs/nca-data-search/boosted-nca/sweep-a/audit.json), [seed-0 result](SWEEP_RESULTS.md), [seed-1 run](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_1/downstream/full_10m/metrics.json), [seed-2 run](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_2/downstream/full_10m/metrics.json), [seed-3 run](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_3/downstream/full_10m/metrics.json), [seed-4 run](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_4/downstream/full_10m/metrics.json).
