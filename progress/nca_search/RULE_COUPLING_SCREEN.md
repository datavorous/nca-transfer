# Rule-level spatial coupling screen

Read-only design screen on 5,000 deterministic worlds: every fifth of the first 25,000 accepted reference `40_50` receipts. Reproduce with:

```sh
uv run --all-packages python progress/nca_search/screen_rule_coupling.py --first 25000 --stride 5
```

The script reconstructs each saved random convolutional rule from its logged seed and scores the local derivative of its ten output logits with respect to each possible input color at 32 fixed cells in the first saved frame. It subtracts the mean over input colors, then compares the squared derivative energy of the eight neighbors with the center cell. This estimates **rule-law sensitivity**, not learned transformer attention or causal effect on English. A separate signed score compares horizontal and vertical neighbor energy. The implementation accounts for the existing `jnp.pad(cells, 1, mode="wrap")` behavior, which also duplicates edge color channels. An independent one-cell forward comparison to the saved network gave a maximum logit error of `2.98e-8`.

| Rule score | p10 / median / p90 | ρ horizontal copy excess | ρ lag-2 recurrence | ρ cell-transition MI | ρ gzip | ρ held-out neighbor gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neighbor/center derivative energy | .717 / 1.023 / 1.501 | -.013 | -.027 | -.271 | +.009 | +.049 |
| Mean neighbor derivative energy | .335 / .478 / .677 | -.022 | -.015 | +.018 | -.024 | +.067 |
| Log horizontal/vertical energy | -.344 / .002 / .341 | -.006 | -.006 | -.014 | +.013 | -.017 |

These are Spearman associations across worlds. Ranking both variables within each of five .02-wide gzip sub-bins before pooling barely changes the main observations: neighbor/center energy versus horizontal copy **-.020**, lag-2 **-.030**, transition MI **-.276**, and held-out neighbor gain **+.054**. Directional log energy versus observed horizontal-minus-vertical patch-copy rate is only **+.009** (within-bin **+.008**). The held-out gain compares smoothed cell prediction from `(current color, one cardinal neighbor)` with current-color-only prediction: transition frames 0–17 fit counts, frames 18–25 score them, and four neighbor directions are averaged. It is a cell-level diagnostic, not a patch-token or language-model result.

**Decision:** raw effective neighbor coupling and orientation do not explain the high horizontal patch-copy signal in these saved worlds. Their links to a direct neighbor-prediction diagnostic are very small. The moderate negative association with one-cell transition MI warns that selecting high coupling would also change another measured property. This screen does **not** support spending the next 1M-token probe on a high-coupling rule selector. The independently frozen vertical lag-6 patch-copy comparison remains the more direct transformer-facing test; judge it by its controlled downstream result before choosing another rule-design intervention.

Limits: the derivative is local to one frame and 32 sites per rule; ReLU gates and nonlinear repeated dynamics can change which coupling matters later. Saved worlds have one initial state per rule, so rule parameters and realized trajectory properties are not separately randomized. The number of worlds makes tiny associations easy to estimate but does not make them practically useful or establish transfer. No generation or model training was run and no corpus or data-contract file changed.
