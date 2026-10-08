# Cross-world patch coverage screen

This is a **descriptive hypothesis screen**, not a language-transfer result. It reads saved `40_50` trajectories only. The [reproduction script](screen_corpus_diversity.py) takes every 20th accepted world among the first 25,000 `40_50` receipts: 1,250 worlds, 1,215,000 patch positions. It uses the original 2×2 tokenizer powers `(1, 10, 100, 1000)` and framewise raster order. No generation or training was run.

```sh
uv run --all-packages python progress/nca_search/screen_corpus_diversity.py
```

## What recurs across worlds?

Unigrams are the 10,000 possible 2×2 patch IDs. Bigrams and trigrams are consecutive patch IDs **within a frame**; frame boundaries are excluded because the actual training sequence inserts marker tokens there. For each world, *cross-world mass* is the share of its n-gram **positions** whose n-gram occurs in at least one other sampled world. *Cross-world vocabulary share* instead gives each distinct n-gram in the world equal weight. These are leave-one-world-out support measurements over the fixed sample, not held-out language prediction. More sampled worlds would raise support.

| Measurement | Observed value |
| --- | ---: |
| Observed patch vocabulary | 10,000 / 10,000 |
| Pooled patch entropy; effective vocabulary | 12.143 bits; 4,522 |
| Mass of 100 most frequent patch IDs | 12.03% |
| Patch IDs appearing in exactly one sampled world | 0 |
| Distinct bigrams; appearing in one world only | 890,396; 777,446 (87.32%) |
| Distinct trigrams; appearing in one world only | 1,135,079; 1,130,298 (99.58%) |

| World-level property | Median [p10, p90] | Spearman ρ with gzip | Within-gzip-bin ρ with cell change |
| --- | ---: | ---: | ---: |
| Bigram cross-world mass | .307 [.188, .466] | −.657 | −.153 |
| Trigram cross-world mass | .0054 [.0011, .0261] | −.475 | −.134 |
| Bigram cross-world vocabulary share | .285 [.177, .408] | −.606 | −.130 |
| Within-world distinct-bigram fraction | .941 [.837, .982] | +.858 | +.084 |

The between-world sharing signal is mostly in **bigrams**. Every patch ID is shared, while only about 0.5% of a median world's trigram *positions* match a trigram in another sampled world. This does not establish that bigram repetition helps transfer: useful learning could require novelty, recurrence, or a balance. The 1250-world sample also covers just 5% of the first 25,000 saved 40–50 worlds, so singleton percentages are sample-size dependent.

Bigram cross-world mass decreases steadily across gzip sub-bins: `.430, .385, .329, .287, .235` for `40–42` through `48–50`. It also tracks world-level spatial excess agreement (raw ρ `+.266`; within gzip bins `+.402`) and transition mutual information (raw `−.238`; within bins `−.281`). Thus selecting for shared bigrams could indirectly select spatial structure or lower cell-transition MI. The within-bin coefficient ranks variables separately in each `.02` gzip sub-bin, centers ranks, and pools them; it is a descriptive control, not causal adjustment. There is still within-bin gzip variation.

## Single-intervention targets

The following are **unvalidated hypotheses**, ranked for ease of testing and distinctness from the existing per-world screen. Freeze any reference lexicon on a disjoint saved subset before sampling candidate worlds; otherwise selection can exploit its own evaluation sample. Keep the original rule family, tokenizer, 40–50 band, five gzip sub-bin quotas, train/validation counts, label budget, model, and TinyStories protocol. Match or report mean change and spatial agreement after selection. The paired seed-0 1M-token probe is a screen; a candidate needs paired seeds and the complete curve before a sample-efficiency claim.

| Priority | Proposed one-property intervention | Why it may matter | Main diagnostic |
| ---: | --- | --- | --- |
| 1 | Higher **held-out-reference bigram overlap**: select worlds whose token bigrams occur in a frozen disjoint lexicon, versus lower overlap. | Reusable short contexts might improve embedding and next-token learning. | Match gzip and spatial excess; inspect actual bigram frequency, not just presence. |
| 2 | Interior bigram overlap: favor a predeclared middle range over both tails. | Excess novelty may be hard; excess repetition may be redundant. | Two-sided selection changes both tails; predeclare thresholds and compare each tail separately. |
| 3 | Higher **corpus marginal coverage**: greedily add candidate worlds bringing the most previously unseen patch IDs, at matched gzip. | A broad patch vocabulary might train more embeddings. | All 10,000 IDs already occur in this sample; likely saturates quickly. Inspect rare-ID counts and reject if there is no residual range. |
| 4 | Higher **corpus bigram coverage**: choose worlds with more new bigrams relative to a frozen selected prefix. | Compositional variety might supply broader local transitions. | Strong gzip link; match within bins and compare NCA validation learning. |
| 5 | Lower corpus bigram coverage at fixed unigram counts. | Repeated transitions could provide more repetitions per learnable relation. | Also shifts spatial agreement and transition MI; quantify both. |
| 6 | Higher **cross-world trigram overlap** at fixed bigram overlap. | Longer repeated motifs may exercise context use beyond pair statistics. | Median overlap is only .0054 in this sample; check feasibility and avoid selecting near duplicates. |
| 7 | Balance patch-ID marginals across the selected corpus by reweighting candidate acceptance, preserving world count. | Avoid spending many NCA labels on common IDs while rare IDs remain undertrained. | Changes many world properties at once; use only after a more isolated selector, or treat as a separate corpus-level policy test. |

The most direct next test is priority 1: it uses a frozen lexicon and a single measurable threshold. It should be compared with the already proposed within-world bigram-repetition target in [TRANSFORMER_TARGETS.md](TRANSFORMER_TARGETS.md): within-world repetition and across-world sharing are different variables. **No correlation between any quantity here and English validation loss has been measured.** The saved language results pool whole corpora, so they cannot identify the value of an individual world's n-grams.
