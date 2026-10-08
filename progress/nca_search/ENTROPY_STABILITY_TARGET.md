# Late patch-entropy stability screen

This is a **read-only saved-data screen**, not a TinyStories transfer test. Reproduce from the repository root:

```sh
uv run --all-packages python progress/nca_search/screen_entropy_stability_target.py --first 25000 --stride 5
```

The script takes every fifth accepted 40–50 gzip world among the first 25,000 receipts (5,000 deterministic reference worlds), plus every twentieth training row from the saved high- and low-horizontal-copy packed corpora (999 each). It reads saved states and packed tensors only. Each frame is 36 raster-ordered 2×2 base-10 patch tokens bracketed by markers. The NCA loss masks frame 0; the primary measure compares **frames 1–9 versus 18–26**, each with 324 scored patch-token positions. Entropy is plug-in Shannon entropy of each 324-token block. The earlier exploratory screen used frames 0–8 for its early block; its median (`−.039` bits) is therefore not the cutoff below.

## Reference-world associations

World-level Spearman correlations describe properties of NCA worlds. There is **no world-level English outcome**.

| Feature | Median [p10, p90] | ρ gzip | ρ cell change | ρ horizontal-copy excess |
| --- | ---: | ---: | ---: | ---: |
| Early scored-patch entropy, bits | 6.834 [6.235, 7.267] | +.896 | +.306 | +.055 |
| Late scored-patch entropy, bits | 6.764 [6.062, 7.247] | +.892 | +.275 | −.006 |
| Late − early entropy, bits | **−.0378** [−.4623, +.2121] | +.126 | −.004 | −.042 |
| Late − early distinct patch IDs | −3 [−27, +15] | +.103 | −.013 | −.048 |
| Late patch tokens unseen in early block | .312 [.194, .420] | +.833 | +.154 | +.053 |
| Late − early visible prior-token fraction | +.0617 [.0000, +.1358] | −.078 | +.050 | +.037 |

The entropy drift has weak rank association with gzip, cell change, and horizontal copy. Its within-gzip-bin rank correlations are **−.025** with cell change and **−.046** with horizontal-copy excess. In contrast, it correlates **−.891** within gzip bins with the change in exact-token reuse visible inside the actual 256-token causal windows. Greater late entropy therefore implies fewer late targets with an identical earlier token in the same window. That is a real training-target tradeoff, not automatically an improvement.

Splitting the reference sample at its median scored-patch entropy drift (`>= −0.037777` bits; 2,500 worlds in each half) yields these high-minus-low mean differences: late entropy **+0.3507** bits, early entropy **−0.1057** bits, distinct-ID drift **+26.12**, late-new-to-early patch mass **+.0494**, and visible-repeat drift **−.0814**. Gzip also rises **+.00555**, cell change **+.00174**, and horizontal-copy excess falls **−.00259**. A controlled selector would need matched gzip sub-bin quotas and audits of these residual shifts.

## Existing corpus direction check

These are deterministic world samples from **different corpora**, not paired worlds; the reference sample spans the 25,000 accepted receipts, while high/low-copy samples take training rows only. The reference and candidate rows are therefore descriptive checks, not a matched causal contrast.

| Corpus | Worlds | Early H | Late H | Late − early H | Distinct-ID drift | Visible-repeat drift | Horizontal-copy excess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 5,000 | 6.7894 | 6.6863 | −.1031 | −5.26 | +.0682 | +.00267 |
| High horizontal copy | 999 | 6.7884 | 6.6627 | **−.1257** | −6.45 | +.0716 | +.01179 |
| Low horizontal copy | 999 | 6.7829 | 6.7088 | **−.0741** | −4.53 | +.0663 | −.00617 |

The high-copy corpus has **more**, not less, late entropy decline than the low-copy corpus, yet its completed five-seed 1M English probe has lower loss in every seed ([experiment log](EXPERIMENT_LOG.md)). This observation weakens a simple monotone claim that sustained late token diversity explains the known gain. It does not rule out entropy stability helping under a different selector, because the corpora differ in many properties and are not independent rule-corpus replications.

**Decision:** keep entropy stability as a distinct, testable secondary target, below fresh rule-corpus replication and the more direct causal-token lag targets. If tested, freeze the median from the full reference selection using the **scored-frame definition here**, match five gzip-bin quotas and ideally cell-change strata, keep the original 27-frame/token budget, and compare high/low arms on NCA window loss and paired TinyStories curves. Audit early and late entropy separately, late novelty, visible reuse, patch marginals, and horizontal copy. Do not interpret entropy drift alone as useful predictability.
