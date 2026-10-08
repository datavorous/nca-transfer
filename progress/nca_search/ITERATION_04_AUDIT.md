# Iteration 04 corpus audit

Read-only comparison of `iteration-04-high-patch-copy` with the **exact reference selection used for NCA training**: the first 19,975 `40_50` training worlds and first 256 validation worlds in receipt order. The reference packed file contains 25,000 worlds, but the additional worlds are outside this matched selection. No data generation or training was done for this audit. Reproduce with:

```sh
uv run --all-packages python progress/nca_search/audit_iteration_04.py
```

The script scans all 20,231 selected worlds for gzip ratios, rule IDs, and attempt IDs. It computes the more expensive world features on **1,017 worlds per corpus**, taking every 20th selected world separately within each split and each of five 0.02-wide gzip bins. Thus the feature samples have identical split/bin counts. They are deterministic systematic samples, not independent random draws; small feature differences should be treated as descriptive and may change with sampling phase.

## Matching and selection

| Check | Candidate | Reference selection |
| --- | ---: | ---: |
| Training / validation worlds | 19,975 / 256 | 19,975 / 256 |
| Usable NCA training labels | 19,715,325 | 19,715,325 |
| Gzip-bin counts, train | 2,913 / 3,544 / 4,034 / 4,559 / 4,925 | Same |
| Gzip-bin counts, validation | 45 / 42 / 49 / 57 / 63 | Same |
| Full-selection mean gzip | 0.455225 | 0.455195 |
| Full-selection median gzip | 0.457562 | 0.457819 |
| Latest selected rule attempt | 189,402 | 92,286 |

The full empirical gzip CDFs differ by at most **0.00262** in cumulative fraction. Within-bin mean differences are at most **0.000095**. Coarse-bin matching has also preserved the finer gzip distribution closely. This is strong evidence against gzip drift explaining a large transfer difference, although gzip is only a rough proxy for token statistics.

Rule-seed overlap is **10,135 / 20,231** worlds: 10,003 training and 132 validation rules. The remaining candidate rules come later in the same deterministic generator stream. Both sets contain unique `(split, rule seed)` pairs. This shared half reduces independence between corpora; the new half could differ in unmeasured rule properties or by chance. The current contrast cannot isolate patch copying causally from all other properties selected with it.

## World-property comparison

Means below are from the matched **1,017-world-per-corpus feature sample**, except the last row. `Δ` is candidate minus reference. The full-selection patch-copy means come from the recorded sampling/result log and candidate receipt, not from the feature subsample.

| Property | Candidate | Reference | Δ |
| --- | ---: | ---: | ---: |
| Horizontal patch-copy excess, sampled | 0.01144 | 0.00322 | **+0.00822** |
| Horizontal raw patch-copy rate | 0.02538 | 0.01646 | +0.00892 |
| Fixed-count patch-copy chance | 0.01394 | 0.01325 | +0.00069 |
| Cell-change fraction | 0.70639 | 0.70424 | +0.00215 |
| Spatial excess agreement | 0.00012 | 0.00084 | -0.00072 |
| Transition mutual information, bits | 0.17925 | 0.17201 | +0.00724 |
| Patch entropy, bits | 7.26185 | 7.26524 | -0.00339 |
| Rare-patch mass (within-world count ≤2) | 0.27140 | 0.26502 | +0.00638 |
| Same-token repeat in 256-token context | 0.47292 | 0.47416 | -0.00124 |
| Late patch novelty | 0.20645 | 0.20210 | +0.00436 |
| Lag-2 cell recurrence | 0.33128 | 0.32349 | +0.00779 |
| **Horizontal patch-copy excess, full selection** | **0.01163** | **0.00274** | **+0.00889** |

The intended selector clearly moved its target: horizontal copying rose while the chance baseline was nearly unchanged in the sample. Full-selection mean cell change is **0.70588** versus **0.70335** in the experiment log, also close. Spatial agreement, patch entropy, and 256-context repeat show no clear sampled shift. Rare-patch mass, transition MI, late novelty, and lag-2 recurrence move slightly; their effects are small relative to world-to-world spread and are plausible secondary selection confounds. In particular, a gain cannot yet be attributed uniquely to adjacent patch copying. Gzip-conditioned and exact-rule controls would be needed for that claim.

## Packed data and loss positions

The candidate packed file is `(20,231, 1,026)` for both `sequences` and `targets`, `torch.int32`; the reference file is `(25,000, 1,026)`, also `torch.int32`. The candidate SHA-256 is `97dfed5bdc4174d9f9964dd9eebdaca88e53225236a44659432a9c8583a43518`, matching its completion receipt. Both have 27 frames of 38 tokens: start marker `10000`, 36 base-10 2×2 patch IDs, end marker `10001`. Stored `targets` mask both marker positions with `-100`. Split-index order matches receipt/chunk order for the exact 19,975/256 selection. Eight sampled worlds from each corpus matched `pack_one` exactly in both tensors.

The actual `NCAWindows` training path constructs shifted targets from `sequences`: it masks the first 37 targets in the first 256-token window and scores 219 + 256 + 256 + 256 = **987** positions per world. This path does not use stored `targets`; candidate and reference follow the same rule. No observed tensor layout, dtype, marker, target-mask, tokenizer, split-count, or effective-label mismatch explains the iteration 04 result.

**Interpretation:** the sampling intervention produced a real patch-copy shift with tightly matched gzip composition and training budget. A transfer gain, if replicated across model seeds, would support this **corpus selection** as useful; it would not prove patch copying is the causal feature because roughly half the worlds change and some other token/dynamic properties move with selection.
