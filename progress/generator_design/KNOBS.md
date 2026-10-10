# Which knobs would make this a designed NCA generator?

The current `copy_probability` is a real generator control: it raises horizontal patch recurrence without selecting worlds by their measured recurrence. It is **not** enough to reproduce the rejected-sample Top-10 cohort. The [distribution comparison](TOP10_DISTRIBUTION_COMPARISON.md) showed that one fixed value matches the mean while missing the score spread and gzip-dependent pattern. This note tests which additional *direct update controls* move the missing features. It does not test language transfer.

## Target and small intervention screen

The target is a fixed thinning of 1,623 worlds from the three completed fresh Top-10 confirmation corpora. The candidate screen uses two new rule streams, 384 attempted rules per stream and condition. Only full trajectories in gzip [0.40, 0.50) enter the comparison. The same rule keys are reused across candidate conditions. All controls act inside the autonomous zero-player update; no candidate is accepted or rejected by a recurrence score. [Code](probe_additional_knobs.py) and [raw summary](additional_knobs.json) record the exact probe.

| Feature, mean | Frozen Top-10 sample | Horizontal copy only | Four-control prototype |
|:---|---:|---:|---:|
| Horizontal patch excess | 0.0324 | 0.0333 | 0.0333 |
| Vertical patch excess | 0.0124 | 0.0036 | 0.0123 |
| Cell change fraction | 0.7282 | 0.7117 | 0.7276 |
| Cell lag-2 recurrence | 0.3488 | 0.3131 | 0.3613 |
| Transition MI, bits | 0.2291 | 0.1639 | 0.2301 |
| Patch entropy, bits | 7.2548 | 7.3230 | 7.2305 |

The four-control prototype uses horizontal patch-copy probability `0.03`, vertical patch-copy probability `0.01`, a `0.10` chance to replace a cell's neural update with the alternate color in a fixed two-cycle, and a `0.10` chance to change a cell if the update would otherwise leave it unchanged. The two-cycle maps color `c` to `(c+5) mod 10`; applying it twice returns `c`. Spatial copies use the newly updated grid and do not cascade within a step. The code is an exploratory simulator, **not a frozen public generator recipe**.

## What each control buys

1. **Horizontal copy strength conditioned on target gzip sub-bin.** This is the primary property control. The exploratory closest values decline from about `0.05` in gzip [0.40, 0.42) to `0.015` in [0.48, 0.50). A generator can choose the intended gzip sub-bin first, set its copy probability, then retain only trajectories in that gzip bin. That uses the existing gzip criterion but no recurrence rejection. One global probability cannot follow the frozen sub-bin pattern.
2. **Vertical patch-copy strength.** At horizontal `0.03`, adding vertical probability `0.01` raises mean vertical excess from `0.0036` to `0.0140`, near the frozen `0.0124`. This controls anisotropy separately from horizontal recurrence.
3. **Temporal transition structure.** A per-cell two-cycle chance of `0.10` raises lag-2 recurrence from `0.3131` to `0.3487` and transition MI from `0.1639` to `0.2120` in this gzip-conditioned screen. This gives a direct temporal control rather than using spatial copies as a proxy for all recurrence.
4. **Activity or change rate.** In the combined temporal/vertical probe, enforcing a change on 10% of cells whose update would otherwise stay the same raises observed change from `0.6969` to `0.7276`, close to the frozen `0.7282`. It also moves MI from `0.2120` to `0.2301`. This knob matters because temporal recurrence and activity need not move together.
5. **Rule-to-rule variation, if distribution matching is the objective.** Fixed controls match several means but make world-to-world features too concentrated. The frozen sample has horizontal/vertical/lag-2/MI standard deviations `0.0226/0.0282/0.1284/0.1337`; the four-control prototype has `0.0116/0.0102/0.0791/0.0917`. A future generator would need variability across rules, such as a specified joint distribution over control values or a conditional neural rule family. Independent per-knob noise is not enough by itself; the joint feature distribution must be checked.

These five are **design requirements suggested by this specific operator family**, not proof that these exact hand-written operations are uniquely necessary. In particular, a neural rule generator optimized toward the same feature targets might replace several explicit gates while staying within the original local-network architecture.

## Why matching means is still insufficient

For the four-control prototype, the empirical KS distances from the frozen feature sample are `0.399` for vertical recurrence, `0.137` for cell change, `0.187` for lag-2 recurrence, and `0.223` for transition MI. Its horizontal score KS distance to all 60,693 frozen selected worlds is `0.283`, and only `85.3%` of its gzip-eligible worlds pass the frozen Top-10 cutoff. The vertical mean looks excellent while its distribution is *less* similar than with horizontal copying alone. Matching six means therefore does not recreate the selected data law.

The current best answer is a **conditional, stochastic generator specification**, not one global scalar. The API should eventually express target gzip bin, horizontal and vertical spatial recurrence, temporal recurrence, activity, and rule-level spread as explicit parameters. The implementation should report the *realized* feature distributions and acceptance counts, not promise exact per-world values. Do not claim the resulting corpus transfers like frozen Top-10 until a matched downstream comparison establishes that.

This screen is small and exploratory. It conditions on realized gzip, so different variants may retain different rule subsets. The candidate controls also change the NCA transition law. No model was trained and no frozen corpus was altered.
