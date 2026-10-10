# Designing distributions by optimizing the neural NCA rules

**Result:** The original 3×3 convolutional NCA can be steered toward different horizontal-recurrence distributions by optimizing its **686 neural rule weights**. No explicit patch-copy operator is used. Three requested recurrence levels produced ordered means on fresh, unseen rollouts. This establishes coarse feature control in a small probe, not equivalence to the frozen Top-10 corpus or a language-transfer benefit.

## Method

`nca.optimize_rule(config, rule_seed, design, search)` starts from the original `Game.rule` initialization. A small `(1+16)` evolutionary search perturbs the flattened Flax weights for 100 generations. It evaluates **hard categorical full trajectories** from four fixed training initial states. Fitness penalizes distance from the requested horizontal patch recurrence, gzip outside [0.40, 0.50), cell-change target 0.73, and patch-entropy target 7.25. Eight different initial states are measured after search and never enter fitness. The network architecture, stochastic cell update, grid, tokenizer, and trajectory layout are unchanged.

The API can optimize multiple initial neural rules toward one `RuleDesign` and pass them to `NeuralRuleGenerator` to create a distribution of worlds. `NeuralRuleGenerator.save` writes ordinary v1 trajectory arrays plus the learned rule weights and a `design.json` replay sidecar. `nca.reproduce_saved` verifies exact replay. The self-contained [validation script](verify_neural_api.py) reruns the nine searches and generates eight further rollout seeds per optimized rule in a temporary dataset. Its [receipt](neural_api_validation.json) contains the feature summaries; the nine original individual optimization receipts are also in this folder.

## Held-out initial states for each optimized rule

Each cell below is the mean horizontal excess across eight held-out initial states for one optimized rule. The target enters the search; the held-out initial states do not.

| Requested excess | Rule 17 | Rule 23 | Rule 47 |
|---:|---:|---:|---:|
| 0.015 | 0.016 | 0.024 | 0.016 |
| 0.040 | 0.020 | 0.057 | 0.026 |
| 0.070 | 0.067 | 0.063 | 0.049 |

The middle target is visibly less reliable across rule initializations. This is why the requested score should be treated as a distribution-level control, not an exact per-rule or per-world guarantee. Across these nine searches, 68 of 72 held-out optimizer worlds were in gzip [0.40, 0.50); some rules still occasionally leave the band.

## Fresh worlds from the public generator API

The table uses 24 new rollouts per requested level: three optimized neural rules × eight initial seeds derived by the generator, separate from the optimizer's training and held-out seeds.

| Requested horizontal excess | Realized mean ± world SD | Gzip [0.40, 0.50) | Mean gzip | Cell change | Transition MI, bits |
|---:|---:|---:|---:|---:|---:|
| 0.015 | 0.0167 ± 0.0114 | 24/24 | 0.447 | 0.721 | 0.108 |
| 0.040 | 0.0351 ± 0.0267 | 22/24 | 0.440 | 0.757 | 0.194 |
| 0.070 | 0.0711 ± 0.0418 | 21/24 | 0.424 | 0.659 | 0.165 |

This is genuine **rule optimization**, not rejection sampling of high-recurrence worlds. The same original neural architecture generated each cohort. The output distributions are separated in this small check, though they overlap and higher recurrence also changes other properties.

## Multi-feature target check

`RuleDesign` also accepts optional vertical recurrence, cell lag-2 recurrence, and transition-MI targets. A separate [feature-only probe](probe_neural_multifeature.py) optimized three rules for the frozen Top-10 feature means, using 120 generations and 24 perturbations per generation. It generated 24 further worlds, 23 of which were in gzip [0.40, 0.50).

| Feature, mean | Frozen Top-10 sample | Multi-feature neural probe |
|:---|---:|---:|
| Horizontal excess | 0.0324 | 0.0275 |
| Vertical excess | 0.0124 | 0.0102 |
| Cell lag-2 recurrence | 0.3488 | 0.3354 |
| Transition MI, bits | 0.2291 | 0.2128 |
| Cell change fraction | 0.7282 | **0.8257** |
| Patch entropy, bits | 7.2548 | 7.1049 |

Several means moved toward the reference, but cell activity moved away. The neural probe also has narrower feature spreads than the selected corpus. Its [receipt](neural_multifeature_probe.json) is exploratory and does not define a production recipe. The search objective's relative weights, the small number of rule initializations, and conflicts among targets remain unresolved.

## What the package now supports

The standalone package exposes `RuleDesign`, `RuleSearch`, `optimize_rule`, and `NeuralRuleGenerator`. It can request a horizontal recurrence level and gzip range, optimize independent original neural rules, generate trajectories, save them, and replay them. Optional feature targets allow further experiments, but the current evidence supports only **coarse horizontal distribution control**. A full Top-10 replacement would need stronger multi-feature fitting, rule-to-rule diversity, exact gzip sub-bin quotas, independent feature validation, and ultimately a matched downstream transfer test.

No language model was trained. No frozen corpus was changed.
