# Generating horizontal recurrence directly

**Feasible as a generator intervention; downstream value is untested.** The existing NCA rule samples each cell from neural logits. A small additive bias toward a neighboring color did not reliably raise *excess* horizontal 2×2 patch recurrence within the 40–50% gzip band in an informal pilot. A direct patch operation did. This design leaves the frozen Top-10 selection experiment untouched.

## One configurable operation

After each original neural NCA update, partition the new 12×12 grid into its thirty-six aligned 2×2 patches. For each patch except the leftmost in a patch row, independently copy the **newly updated** patch immediately to its left with probability `copy_probability`. Apply all copies simultaneously, so a copied patch cannot be copied again across the same frame. The resulting grid is the input to the next neural update. `copy_probability=0` is exactly the original rule; a fixed positive value changes the generator without selecting worlds by their realized recurrence score.

This is an explicit, stochastic patch-level operator on top of the original neural rule. It is not the original translation-equivariant 3×3 NCA, and it deliberately uses the same 2×2 alignment as the tokenizer. Describe it as a **hybrid patch-copy NCA generator**, not as an untouched sample of the original rule distribution.

The [feature-only probe](probe_patch_copy.py) uses the public `nca.recurrence.HorizontalRecurrenceGame`. It accepts a probability, a rule count, and a rule-stream seed. It retains the original rule, initial-state, and neural-update seeds across probabilities. It saves aggregate feature JSON only; it neither saves corpora nor trains models. Run it from the repository root:

```sh
uv run --locked --all-packages python progress/generator_design/probe_patch_copy.py \
  --seed 2471 --rules 192 --probabilities 0 .01 .03 .05 .10
```

## Cheap feature check

The horizontal score is the observed adjacent 2×2 patch-ID match rate minus the match rate expected from that world's patch-ID counts. The table conditions on the **original 40–50% gzip band**. `n` is the number of eligible worlds among 192 attempted rules, not a corpus acceptance study. No world is filtered by its recurrence score.

| Rule stream | Copy probability | Gzip-eligible `n` | Mean horizontal excess | Mean gzip within band |
| ---: | ---: | ---: | ---: | ---: |
| 2471 | 0 | 48 | 0.00599 | 0.4521 |
| 2471 | 0.01 | 57 | 0.01116 | 0.4532 |
| 2471 | 0.03 | 63 | 0.03223 | 0.4505 |
| 2471 | 0.05 | 66 | 0.05272 | 0.4515 |
| 2471 | 0.10 | 69 | 0.10151 | 0.4532 |
| 8123 | 0 | 44 | 0.00405 | 0.4505 |
| 8123 | 0.01 | 42 | 0.01503 | 0.4565 |
| 8123 | 0.03 | 49 | 0.03610 | 0.4537 |
| 8123 | 0.05 | 50 | 0.05342 | 0.4594 |
| 8123 | 0.10 | 54 | 0.10009 | 0.4588 |

Sources: [stream 2471](probe_seed_2471.json), [stream 8123](probe_seed_8123.json). The three completed frozen Top-10 confirmation corpora have mean horizontal excess **0.03341–0.03353** in their selected 40–50% worlds, so `copy_probability=0.03` produces a similar *mean score* in this small check. The distributions and the transition laws may still differ greatly. The unchanged `copy_probability=0` path reproduced the original rollout exactly for the tested seed. A 3% trajectory retained raw shape `[27,12,12,1]` and the original 1,026-token int32 packing in a direct check.

A direct [distribution comparison with the complete frozen Top-10 confirmation scores](TOP10_DISTRIBUTION_COMPARISON.md) confirms the limit: probability 0.03 matches the mean, but narrows the recurrence distribution and leaves correlated trajectory features different.

The [additional-knob screen](KNOBS.md) tests direct controls for vertical recurrence, temporal structure, and activity. It gets close to several target means while still missing feature distributions, so it remains a design probe rather than a validated corpus recipe.

The [neural-rule optimizer](NEURAL_RULE_OPTIMIZATION.md) instead keeps the original NCA update and changes its neural weights. Its small held-out check produced distinct horizontal-recurrence distributions, though matching the entire selected corpus remains unresolved.

## What this establishes

The generator can raise horizontal recurrence without screening for high-recurrence worlds. The effect appears in two independent, small rule streams and survives the existing gzip filter. It does **not** show better language transfer, equal NCA difficulty, or a causal effect of recurrence. Patch copying also changes the transition law and may alter other properties such as lagged recurrence or transition mutual information. The 192-rule checks are too small to predict production acceptance rates under exact gzip sub-bin quotas.

Before using this as a training corpus, freeze one probability without looking at downstream outcomes. Check all five gzip sub-bin quotas, patch entropy, lag-2 recurrence, transition MI, rule independence, and raw/packed replay. Compare against a contemporaneous Ordinary corpus with the same NCA-label budget and paired language seeds. Do not treat a feature match to Top-10 as a matched corpus or an expected transfer gain.

## Data-contract boundary for the library integration

The existing raw v1 writer saves NumPy trajectory shards and a `manifest.json` whose `implementation` field describes the **original** 3×3 neural update. The feature probe saves no trajectories. The opt-in library generator writes the same raw arrays and manifest structure, with truthful `implementation` values and a `design.json` sidecar recording the probability and gate rule.

The arrays, filenames, shape, dtype, token IDs, marker positions, and split semantics stay unchanged. The sidecar is the one provenance addition. Existing v1 readers ignore it. `nca.reproduce_saved` dispatches to patch copy when the sidecar is present and to the original update when absent. The default generator and frozen corpora remain unchanged. See [the version 1 contract](../../docs/DATA_CONTRACT.md).
