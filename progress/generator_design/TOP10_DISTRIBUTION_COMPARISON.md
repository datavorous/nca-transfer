# Can the recurrence knob generate Top-10-like worlds?

**Partly.** Patch copying can force the average horizontal recurrence into the Top-10 range without rejecting worlds by recurrence. One fixed probability does not reproduce the full selected distribution. Matching the mean score also leaves visible differences in vertical recurrence, cell dynamics, and transition information. This is a feature-only comparison; it says nothing about language transfer.

## Comparison

The frozen reference is the complete Designed arm of each of the three finished [fresh confirmation corpora](../../runs/nca-data-search/top10-confirmation/audit.json): 60,693 worlds total. Each was selected using the same five fixed horizontal-recurrence cutoffs within gzip sub-bins [0.40, 0.42), ..., [0.48, 0.50). The feature comparison below uses all 60,693 saved per-world recurrence scores and a fixed thinning of 1,623 raw worlds for additional features.

The candidate uses the new public `HorizontalRecurrenceGenerator` with the original 12×12, 10-color, 27-frame setup. For each probability, it tries the **same 768 fresh rule seeds** across two streams and keeps only generated worlds whose *full trajectory* falls in gzip [0.40, 0.50). It never screens candidates by horizontal recurrence. The candidate sample is small, so KS values and cutoff fractions are descriptive. The [script](compare_top10.py) and [machine-readable results](comparison.json) reproduce the table without model training or writes to frozen corpora.

| Copy probability | Gzip-eligible worlds / 768 | Horizontal excess mean ± SD | Frozen cutoff pass rate | KS distance to frozen score distribution |
|---:|---:|---:|---:|---:|
| Frozen Top-10 | 60,693 | 0.03345 ± 0.02432 | 100% by selection | 0 |
| 0 | 185 | 0.00376 ± 0.01575 | 11.4% | 0.773 |
| 0.01 | 188 | 0.01241 ± 0.01450 | 29.8% | 0.528 |
| 0.015 | 194 | 0.01827 ± 0.01350 | 51.5% | 0.357 |
| 0.02 | 201 | 0.02576 ± 0.02045 | 66.7% | **0.206** |
| 0.03 | 208 | **0.03441 ± 0.01687** | 82.7% | 0.262 |
| 0.04 | 210 | 0.04334 ± 0.01752 | 93.8% | 0.405 |
| 0.05 | 220 | 0.05472 ± 0.02183 | 97.7% | 0.576 |

`0.03` nearly matches the **mean** but has a narrower spread. Its score 10th/50th/90th percentiles are 0.0185/0.0317/0.0489, versus 0.0114/0.0268/0.0642 for frozen Top-10. `0.02` gives the smallest empirical KS distance in this grid, though its mean is lower. `0.05` makes almost every gzip-eligible world pass the old Top-10 cutoff, but overshoots the selected score distribution. Passing the cutoff is weaker than matching the distribution.

## The required probability changes with gzip sub-bin

The frozen selected score rises as gzip decreases. In this small grid, no constant probability tracks that pattern. Each row below chooses the probability with the smallest *within-bin* empirical KS distance, using **feature outcomes only**. This is a diagnostic calibration, not a validated generator recipe.

| Gzip sub-bin | Frozen mean excess | Closest grid probability | Generated mean excess | Generated cutoff pass rate | Within-bin KS |
|:---|---:|---:|---:|---:|---:|
| 0.40–0.42 | 0.0623 | 0.05 | 0.0652 | 88.1% | 0.170 |
| 0.42–0.44 | 0.0471 | 0.04 | 0.0409 | 87.2% | 0.149 |
| 0.44–0.46 | 0.0336 | 0.03 | 0.0322 | 87.5% | 0.125 |
| 0.46–0.48 | 0.0235 | 0.02 | 0.0225 | 86.7% | 0.203 |
| 0.48–0.50 | 0.0156 | 0.015 | 0.0170 | 94.0% | 0.264 |

This suggests a future **gzip-conditioned copy-probability schedule** could approximate the selected score distribution without recurrence-based rejection. The tested bin samples contain only about 27–52 worlds per probability, and even the best grid values leave some worlds below the frozen cutoffs. More samples and an independent check would be needed before treating such a schedule as matched.

## Other properties remain different

| Feature within gzip 0.40–0.50 | Frozen Top-10 sample | Generated at 0.03 |
|:---|---:|---:|
| Vertical patch recurrence excess | 0.01245 | 0.00180 |
| Cell change fraction | 0.72818 | 0.69520 |
| Cell lag-2 recurrence | 0.34878 | 0.32297 |
| Cell transition mutual information, bits | 0.22906 | 0.16528 |
| Patch entropy, bits | 7.25483 | 7.24097 |

The original Top-10 selector favored worlds with other correlated structure. Directly copying patches mainly increases the horizontal property, so the generated worlds are **not equivalent NCA data** even when their horizontal mean matches. The selected comparison uses 1,623 deterministic thinned worlds for these five features; the generated comparison uses all 208 gzip-eligible 0.03 worlds. These are descriptive samples, not independent model outcomes.

**Answer:** We can directly generate more horizontal recurrence, and a probability around 0.03 matches the old selected cohort's average score in this probe. We cannot claim that one fixed knob force-generates the old Top-10 *distribution* or its downstream benefit. A gzip-conditioned knob is plausible but unvalidated. No model training or production corpus generation was performed.
