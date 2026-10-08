# Visible noncopy successor reuse

**Read-only hypothesis screen, not a transfer result.** Reproduce from the repository root:

```sh
uv run --all-packages python progress/nca_search/screen_successor_reuse.py --first 25000 --stride 1
```

The script reads all 25,000 saved accepted 40–50 worlds, in receipt order. It preserves the original 27 frames, 2×2 patch IDs, 38-token frames, and four nonoverlapping 256-token NCA training windows. It considers horizontal pairs `(left patch, right patch)` where the two IDs differ. A scored right-patch target is *eligible* when the same left ID appeared earlier in a completed, nonidentical horizontal pair inside its current window. **Successor reuse** is the fraction of eligible targets whose right ID matches any earlier right ID after that left ID. The first frame can supply history, although its targets are masked. Each measured target and its left neighbor must share a window; the script excludes markers and row-crossing pairs. This describes a visible ordered relation rather than exact adjacent copying.

The frequency-only comparison draws a right ID from the empirical right-ID frequencies of that same window, with the source and its prior successor set held fixed. `successor_reuse_excess` is observed minus this expected match rate, averaged over eligible targets. This is a descriptive, within-world baseline; it is **not** a formal null distribution or proof that a causal transformer learns the relation. The fitted empirical frequencies include the current right ID.

| Feature, 25,000 worlds | Mean | p10 / median / p90 | Spearman ρ with gzip | ρ with horizontal exact-copy excess | Within-gzip-bin ρ with copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eligible source-recurrence targets, count | 328.44 | 250 / 324 / 415 | −.940 | −.185 | −.378 |
| Successor reuse among eligible | .05646 | .01176 / .04154 / .12114 | −.730 | +.030 | +.055 |
| **Frequency-adjusted successor reuse** | **+.01279** | **−.01310 / +.00287 / +.05161** | **−.386** | **+.073** | **+.076** |
| Last-successor match among eligible | .02314 | .00431 / .01813 / .04663 | −.619 | +.027 | +.048 |
| Global noncopy bigram repetition | .06907 | .01617 / .05449 / .14541 | −.871 | −.016 | −.002 |

Raw visible reuse has **ρ +.933** with global noncopy bigram repetition and is highly gzip-linked; it is a poor independent selector. The frequency-adjusted score has ρ **+.597** with global bigram repetition. This leaves some distinct variation but does not isolate a mechanism. Every world has eligible source recurrences; 0.9% have zero observed successor reuse.

## Feasibility within the five gzip sub-bins

The table uses the **frequency-adjusted** score. Each bin has broad overlap on both sides of its own median; the median is a descriptive reference-pool cutoff, not yet a prospective sampling rule.

| Gzip bin | Worlds | p10 | Median | p90 | Above-median share |
| --- | ---: | ---: | ---: | ---: | ---: |
| .40–.42 | 3,640 | −.00819 | +.02456 | +.10099 | .500 |
| .42–.44 | 4,435 | −.01187 | +.01285 | +.07189 | .500 |
| .44–.46 | 5,013 | −.01332 | +.00456 | +.04424 | .500 |
| .46–.48 | 5,766 | −.01373 | −.00063 | +.02626 | .500 |
| .48–.50 | 6,146 | −.01408 | −.00347 | +.01522 | .500 |

Across equal sides of these five within-bin medians, mean adjusted reuse is **+.03173 vs −.00615**, while mean gzip is **.45485 vs .45565**. The high group also has more horizontal copy excess (**+.00511 vs +.00034**), more cell change (**.71171 vs .69481**), and more global noncopy bigram repetition (**.08216 vs .05597**). Thus simple gzip-bin matching would still leave substantial property shifts. The raw reuse median is especially unsuitable: a single global cutoff places 97.0% of the lowest gzip bin but only 11.3% of the highest bin above it.

**Decision.** Keep frequency-adjusted successor reuse as a lower-priority, order-sensitive candidate. It directly counts a relation available to the 256-token objective, but the reference screen does not make it more compelling than the already prioritized visible vertical or boundary-predictability tests. A prospective test should freeze five bin-specific adjusted cutoffs, use disjoint rule blocks and the fixed NCA/English protocol, and audit horizontal copy, cell change, global bigram repetition, eligible count and patch marginals. Match or stratify these if feasible before interpreting any 1M-token difference. No training or new generation was run for this note.
