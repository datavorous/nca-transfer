# Two-frame recurrence visible to the transformer

Read-only screen of all **25,000 saved 40–50 gzip worlds**, in accepted receipt order. The [reproduction script](screen_temporal_target.py) reads the saved trajectories and receipts; it generates no worlds and trains no model:

```sh
uv run --all-packages python progress/nca_search/screen_temporal_target.py --first 25000 --stride 1
```

The original tokenizer yields 27 frames × (`START`, 36 raster-ordered 2×2 patch IDs, `END`) = 1,026 tokens. Lag 76 compares the *same patch position two frames apart*. Each world has 900 eligible patch pairs; all 900 targets are scored because their target frames are 2–26. In the original four nonoverlapping 256-token training windows, the source is inside the target's causal window for **687/900 = 76.33%** of pairs. The visibility check uses the actual `NCAWindows.batch` offsets: for a target at packed position `q`, the source must be at or after `floor((q−1)/256)×256`.

For each pair, subtract the chance of a match under independent rearrangements of the two frames' patch IDs: `sum_id count_target(id)×count_source(id)/36²`. Average this excess over a world's pairs. This controls frame patch frequencies, not rule identity, spatial alignment, or nonlinear predictability.

| World-level feature | 10th | Median | 90th | ρ gzip | ρ cell change | ρ horizontal copy excess | ρ lag-38 excess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All lag-76 copy | .001111 | .011111 | .055556 | −.298 | −.201 | +.233 | +.286 |
| All lag-76 excess | −.012623 | −.000123 | +.034355 | +.044 | −.062 | +.236 | +.324 |
| **Visible lag-76 copy** | .001456 | .011645 | .055313 | −.295 | −.199 | +.232 | +.285 |
| **Visible lag-76 excess** | −.012729 | **−.000133092486** | +.034665 | +.045 | −.062 | +.234 | +.323 |
| Invisible lag-76 excess | −.013896 | −.000757 | +.036795 | +.043 | −.052 | +.218 | +.292 |

All correlations are Spearman correlations **between worlds**, not with English loss. Within the five 0.02-wide gzip bins, visible excess has ρ +.008 with gzip, −.097 with change, +.225 with horizontal excess, and +.311 with lag-38 excess. Visible excess correlates **+.989** with whole-world lag-76 excess and **+.837** with invisible lag-76 excess. Restricting to scored positions changes nothing for this feature because every lag-76 target is scored; restricting further to causally visible pairs retains the distribution and ranking almost entirely. The 76.33% figure is a property of fixed packed positions, so it holds for every world.

## Precise candidate selector

For a controlled next screen, accept a world if its **visible lag-76 excess ≥ −0.00013309248606674373**, the full saved-corpus median above. Keep the original 40–50 gzip band, five gzip sub-bin quotas, train/validation counts, rule stream, token budget, tokenizer, model, optimizer, and TinyStories schedule. The script prints the unrounded threshold. Use a seed-0 1M probe only as triage, then paired seeds and the full curve for a credible transfer result.

The median split is feasible in this reference set (12,500 high and 12,500 low worlds). It is **not an isolated lag-76 intervention**: high versus low halves also shift mean cell change by −.01779, horizontal patch-copy excess by +.00473, and lag-38 excess by +.01589, while mean gzip shifts +.00058. The visible lag-76 excess shifts by +.02993 on the whole-world measure. Match gzip bins and audit these residual shifts on any new corpus; a transfer gain would support the *selection policy*, not prove a two-frame memory mechanism. The recurrence is measured from a single existing rule stream, and no language transfer outcome was measured in this screen.
