# Feasibility of matching patch-copy cohorts on temporal nuisances

This is a **read-only sampling screen**, not a transfer experiment. The proposed comparison keeps the original NCA rollout, 40–50 gzip regime, 2×2 patch IDs, 1,026-position packed world, model, and TinyStories protocol fixed. It changes only which generated worlds enter the corpus. Reproduce the numbers with:

```sh
uv run --all-packages python progress/nca_search/screen_patch_copy_matching.py --first 25000 --stride 1
```

The screen reads all 25,000 accepted 40–50 worlds from the original receipts. High horizontal copy means the observed adjacent 2×2 patch equality minus `Σn(n−1)/(972×971)` is at least the frozen reference median, **−0.000374227071**. Cell lag-2 recurrence is the fraction of cells with `state[t] == state[t−2]`. Transition MI is the full 10×10 consecutive-color transition-table estimate in bits/cell used by the earlier feature screen. These scores come from the same unmodified trajectory tensor.

| Comparison | High count | Low count | Lag-2 high − low | MI high − low | Copy-excess high − low |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw 40–50 reference split | 12,504 | 12,496 | +0.03127 | +0.01302 bits | +0.01774 |
| Matched in gzip bin × lag-2 tercile × MI tercile | 10,781 pairs | 10,781 pairs | +0.00437 | +0.00225 bits | +0.01681 |
| Matched in gzip bin × lag-2 quintile × MI quintile | 10,616 pairs | 10,616 pairs | +0.00131 | +0.00121 bits | +0.01672 |

The high-copy share is 49.3–50.8% in each of the five gzip sub-bins. All **45/45** tercile cells and **125/125** quintile cells contain both arms. Tercile matching retains **86.2%** of each arm; quintile matching retains **84.9%**. The two nuisance standardized mean differences fall from **0.307 and 0.122** raw to **0.043 and 0.021** after tercile matching, or **0.013 and 0.011** after quintile matching. The copy-excess contrast remains about **0.017**. The matched high-minus-low gzip means differ by only +0.00015 (terciles) or +0.00017 (quintiles). Thus there is substantial observed overlap: matching these two features does not erase the patch-copy contrast in the saved reference worlds.

The script uses deterministic receipt order within each cell to calculate maximum balanced counts. It estimates *availability*, not unbiased matched outcomes or prospective acceptance rates. The original sample has 25,000 worlds, so its 10,616 matched pairs alone cannot fill a new **19,975-world per-arm** training corpus. Fresh generation would need a larger pool, and the validation quota of 256 per arm is especially sensitive to sparse cells. The observed retention fraction suggests a modest oversample after high/low filtering, but an actual attempt budget must be verified on fresh rule streams; it is not established here.

## Predeclared controlled selector

1. On fresh rule streams, hash-assign each rule to a high or low candidate arm **before** measuring its trajectory, as in the existing replication sampler. Apply only the frozen horizontal-copy threshold (`>=` for high, `<` for low) and the original 40–50 gzip criterion. Do not tune the threshold or use downstream loss to choose a cutoff.
2. Within each original `.02` gzip bin and train/validation split, balance the two arms on **predeclared** lag-2 and MI terciles. Reference-derived cuts are lag-2 `0.2727777778, 0.3616666667` and MI `0.1099513937, 0.1826968063` bits/cell. Set joint-cell target counts from the original reference distribution before new outcomes are inspected, with equal targets for both arms; use deterministic rule-hash ordering within cells. Preserve the exact original five gzip-bin totals in both arms. Declare an attempt cap and report an incomplete cell rather than relaxing the rule after seeing results.
3. Audit the completed arms for lag-2, MI, full gzip CDF, cell-change rate, patch marginals/entropy, vertical copy, lag-38/76 copy, rule overlap, and label masks. Require the original packed tensor shape, dtype, serialization, tokenizer, train/validation split, and NCA label budget. Then run paired 1M TinyStories probes under the fixed model and schedule; corpus blocks, not model seeds, are the independent replication units.

Terciles are the practical primary design because the reference screen shows strong nuisance balance with 45 cells and the validation split is small. Quintiles are a sensitivity design if a larger fresh pool can fill 125 cells per split. The exact joint-cell quotas and attempt cap must be frozen before generation; this note does **not** launch such a corpus.

This would test whether a **high-copy selection policy still transfers after balancing two measured temporal properties**. It cannot establish horizontal copying as the causal mechanism. Matching is still selection on observed worlds, can leave other properties different, and may also control away a temporal pathway through which patch-copy structure matters. A retained downstream gain would narrow some alternative explanations; a vanished gain would not by itself identify which matched property mediated it.
