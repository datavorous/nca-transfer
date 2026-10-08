# Generator and rule-design targets

This is a **design screen**, not a language-transfer result. It proposes changes to how NCA worlds are made, rather than selecting worlds after generation. No new trajectories or model training were run. The reference remains the 40–50 gzip corpus, with 27 frames of 12×12×1 cells, ten colors, 2×2 patch IDs, 38 tokens per frame, and the same 19,715,325 usable NCA labels. Keep that raw shape and tokenizer for every comparison. Apply the original gzip filter or matched five sub-bin quotas *after* each generator change; report acceptance rate and the actual feature distributions.

The default [game](../../packages/nca/src/nca/game.py) initializes one random three-layer convolutional rule for each rule seed. Its first layer sees a periodic 3×3 neighborhood; its 1×1 layers mix channels and emit ten logits. A sampled cell uses `categorical((logits + identity_bias × current_one_hot) / temperature)`. The [rollout](../../packages/nca/src/nca/generate.py) stores 27 frames starting after ten update steps at stride one. Initial colors are drawn from grid-shared random logits, so cells start independently conditional on those logits. These details determine what each knob actually changes.

## Read-only check of starting-grid variation

The [script](screen_rule_design.py) reads saved initial states and trajectories for every fifth accepted world among the first 5,000 reference `40_50` receipts (1,000 worlds). It computes Spearman rank associations across worlds; the right column ranks and centers within the five .02-wide gzip sub-bins. The outcomes here are *trajectory features*, never English validation loss.

```sh
uv run --all-packages python progress/nca_search/screen_rule_design.py
```

| Initial-state property | Median [p10, p90] | Outcome | ρ raw | ρ within gzip bins |
| --- | ---: | --- | ---: | ---: |
| Color entropy, bits | 2.777 [2.346, 3.034] | Final trajectory gzip | -.003 | -.001 |
| Color entropy, bits | 2.777 [2.346, 3.034] | Cell-change fraction | +.010 | +.013 |
| Color entropy, bits | 2.777 [2.346, 3.034] | Cell-transition mutual information | +.073 | +.077 |
| Color entropy, bits | 2.777 [2.346, 3.034] | Horizontal patch-copy excess | +.014 | +.014 |
| Dominant-color share | .306 [.215, .472] | Horizontal patch-copy excess | -.023 | -.029 |
| Initial neighbor agreement | .181 [.129, .278] | Horizontal patch-copy excess | -.027 | -.023 |

The existing **ten-step burn-in** may erase much of the starting-grid variation, or rule differences may dominate it. This makes a simple selector on initial-state entropy a weak next target. It does **not** show that deliberately changing the initial-state distribution has no effect, nor that these features predict transfer.

## One-knob generator experiments

These are speculative. Rank reflects directness to learnable patch-token dependencies and ability to preserve the tokenizer, not measured transfer value. Choose a predeclared pair of knob settings and retain one change per experiment. Reuse base model initializations, downstream data order, optimizer, and 1M checkpoint protocol; promote only replicated candidates to the full 10M curve. For every setting, verify exact packed shape, label count, gzip sub-bin counts, rule split, and source provenance.

| Priority | Change one generator property | Mechanism to test | Important diagnostic or confound |
| ---: | --- | --- | --- |
| 1 | **Identity bias**: compare zero with a small positive logit offset for the current color. | Adds state persistence to an otherwise unchanged local rule; may expose learnable lag-38 dependencies. | At temperature .001, even a small bias can strongly change categorical outcomes. Calibrate by measured change rate and lag-38 copy; avoid selecting near-frozen worlds. This is distinct from filtering on realized change rate because the transition law changes. |
| 2 | **Temperature**: compare .001 with a modestly larger value. | Makes a fixed rule's logits less decisive, potentially supplying nontrivial stochastic conditional prediction. | Changes randomness, gzip acceptance, and potentially NCA irreducible loss. Record conditional transition entropy and repeated-rollout agreement under the same rule. Temperature does not change argmax rule geometry. |
| 3 | **Rule convolution width**: vary `conv_channels` while keeping hidden width 16. | Changes diversity of learned-from-random local 3×3 pattern detectors, which may change reusable patch relations. | Rule family and parameter count change; match gzip and patch entropy and inspect spatial/temporal copy. Model parameter count stays fixed. |
| 4 | **Rule hidden width**: vary `hidden_channels` while keeping convolution width four. | Changes how many nonlinear combinations of local features affect each color. | Reinitializes the rule distribution. Probe whether logits collapse into trivial attractors or diffuse noise; match gzip and change distribution. |
| 5 | **Frames per rule**: two initial states per rule instead of one, while holding world and label counts fixed. | Repeats a transition law across distinct trajectories, letting the transformer see rule-consistent motifs more often. | Fewer distinct rules is the intended tradeoff. The odd 19,975 training-world count requires one singleton or an explicit one-world trim. Split by rule, never by trajectory; otherwise validation leaks rules. |
| 6 | **Burn-in**: change the starting update time while retaining 27 stored frames. | Earlier frames may show more transient, informative change; later frames may have more stable recurring motifs. | In the current implementation, changing total rollout length also changes `jax.random.split` step keys, so identical seeds do not give a common-prefix counterfactual. Compare aggregate trajectory features, not framewise pairs. |
| 7 | **Frame stride**: store every second update instead of every update, retaining 27 frames and the 38-token frame layout. | Spreads visible temporal dependencies across more physical update steps; may expose longer dynamics within a fixed token budget. | The current `stride>1` rollout has a documented numerical difference from upstream. Changing stride also changes split keys. Measure lag-38/76/114 predictability in *packed-token* space; do not compare raw steps as if they were paired. |
| 8 | **Initial color-logit scale**: add one explicit scale to the normal logits before sampling the initial grid. | Changes starting color concentration without changing the transition rule; could control how much local structure develops. | This needs a small code change because the scale is not in `GameConfig`. The saved-data correlations above are weak after burn-in; lower priority. Record initial-state entropy and color marginals, then assess whether the change survives to stored frames. |

Identity bias and temperature are the smallest rule-law changes. Convolution and hidden widths test rule complexity directly. Frames per rule tests **between-world rule reuse** rather than within-world complexity. Burn-in, stride and initial-state scale alter exposure to dynamics and should be labeled as separate hypotheses, not folded into a combined recipe.

The existing [trajectory screens](HYPOTHESES.md), [patch-token screen](TRANSFORMER_TARGETS.md), and [attention-lag screen](ATTENTION_LAGS.md) suggest useful diagnostics, but no saved-world feature is an observed transfer outcome. The current [experiment log](EXPERIMENT_LOG.md) shows that high cell-change and high spatial coherence did not produce a robust improvement in the 1M probes; generator changes may still behave differently because they alter the source distribution. A single seed or one sampled corpus cannot support a claim about generalization. For a promising knob, repeat with independent rule corpora as well as paired model seeds before claiming a data-design effect.
