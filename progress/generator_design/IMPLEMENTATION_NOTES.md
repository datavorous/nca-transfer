# Designed NCA generator implementation

## What changed

The standalone `nca` package now offers two opt-in ways to request more horizontal recurrence. `HorizontalRecurrenceGenerator` adds a stochastic left-neighbor patch-copy gate after the original neural update. `optimize_rule` instead searches the weights of the original neural rule, and `NeuralRuleGenerator` rolls out those optimized rules without changing the update algorithm. Both APIs accept a `GameConfig`; the default `NCAGenerator` remains the original path.

## How it works

Patch copy applies to aligned 2×2 patches after each neural step. Every copied patch reads the same pre-copy grid, so updates are simultaneous. Probability zero is the original neural update. The probability controls a mechanism, not an exact measured recurrence score.

Neural-rule optimization evaluates the hard trajectories produced by the existing `Game`. A small evolutionary search perturbs the flattened neural weights and scores horizontal excess recurrence, gzip range, cell change, and patch entropy. Optional targets cover vertical excess recurrence, lag-2 recurrence, and transition mutual information. Candidate rules share training initial-state keys for fair comparisons; separate held-out keys measure whether a rule generalizes. The requested features describe a distribution across worlds, not a guarantee for each world.

## Data and replay

Both generators preserve the raw v1 trajectory arrays, tokenization, and default output path. A designed dataset includes `design.json` to identify its update or weights. Neural-rule datasets also include `optimized-rule-weights.npy`. Their `rules.npy` still contains the original initialization keys, so replay must load the weight sidecar. `reproduce_saved` dispatches from the sidecar and refuses a designed manifest when the replay parameters are missing. These sidecars add provenance without changing the tensors consumed by existing readers. Generated corpora and weight arrays are ignored by Git.

## Checks and limits

The package tests check probability-zero equivalence, deterministic replay, v1 array layout, optimized-rule replay, and failure on a missing sidecar. Feature-only probes compare generated recurrence and gzip distributions with the earlier Top-10 selected corpus; results and limitations are in [DESIGN.md](DESIGN.md), [TOP10_DISTRIBUTION_COMPARISON.md](TOP10_DISTRIBUTION_COMPARISON.md), and [NEURAL_RULE_OPTIMIZATION.md](NEURAL_RULE_OPTIMIZATION.md). No language-model training was run for these generator changes. Matching a few measured features does not establish matching downstream transfer or isolate horizontal recurrence as the cause.
