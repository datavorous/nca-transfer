# Vertical lag-6 block D: pretraining feature audit

Run `uv run --all-packages python progress/nca_search/screen_vertical_lag6_arms.py --block d` to reproduce [the machine-readable audit](../../runs/nca-data-search/vertical-lag6/block-d/feature_audit.json). This reads the completed, independently audited chunks; it runs no model. It samples every twentieth selected world separately within each arm, split, and gzip sub-bin in chunk order: **1,017 worlds per arm**, including 1,001 training and 16 validation worlds. Arm/split/bin sample counts are identical. Each sampled world's stored vertical score is recomputed exactly. Full gzip statistics below use **all 20,231 worlds per arm**.

| Property (sample mean unless marked full) | High | Unfiltered | Low | High − unfiltered |
| --- | ---: | ---: | ---: | ---: |
| All-pair vertical excess | +.009911 | +.001345 | −.006207 | +.008566 |
| **Visible** vertical excess | +.010034 | +.001322 | −.006210 | +.008712 |
| Vertical exact-copy rate | .025614 | .015257 | .006657 | +.010357 |
| Full gzip mean | .455220 | .455164 | .455194 | +.000056 |
| Horizontal copy excess | +.003574 | +.002871 | +.001911 | +.000703 |
| Visible lag-76 excess | +.009987 | +.005556 | +.001989 | +.004432 |
| Lag-2 cell recurrence | .332138 | .313623 | .304396 | +.018515 |
| Cell change | .701083 | .702534 | .706727 | −.001451 |
| Transition mutual information (bits) | .178246 | .171511 | .158858 | +.006735 |
| Patch entropy (bits) | 7.260418 | 7.263138 | 7.259782 | −.002720 |
| Rare patch mass (world mean) | .270919 | .264865 | .262482 | +.006054 |
| Same-token repetition in 256-context | .472516 | .474394 | .476062 | −.001877 |

The **predeclared manipulation succeeds**: high > unfiltered > low for both all-pair and causally visible vertical excess. The actual input mask includes its first token; therefore **765/810 = 94.44%** of vertical pairs are scored with the predecessor visible. The older read-only screen excluded three valid boundary pairs and reported 762/810. This audit uses the actual `NCAWindows` mask and leaves the frozen all-pair selection score and threshold untouched.

Full gzip p10/median/p90 are **.413837/.457819/.492027** high, **.414095/.457562/.491770** unfiltered, and **.414095/.457562/.492027** low. The largest high-versus-unfiltered CDF difference on a prespecified .005-wide grid over [.40,.50] is **.00405**. The JSON gives the full CDF grid and nine quantiles. Pooled across sampled worlds, all 10,000 patch IDs appear in every arm. The most frequent ten account for **5.60% / 5.07% / 5.10%** of patch tokens in high/unfiltered/low; the JSON lists their IDs and counts. The selected arms differ somewhat in patch frequencies despite similar per-world entropy.

The selected vertical feature is much more separated than horizontal copy excess (+.00871 versus +.00070 high-minus-unfiltered), so this is a distinct token-order intervention from the horizontal selector. It also selects more **visible temporal recurrence** (+.00443), lag-2 cell recurrence (+.01852), and transition MI (+.00674). A later TinyStories gain can support the vertical-score **selection policy**, but this feature audit cannot attribute a gain uniquely to attention over lag-6 predecessors. [The frozen protocol](VERTICAL_LAG6_FROZEN_PROTOCOL.md) specifies the three-arm seed-0 1M probe and decision gate. No English outcome is part of this audit.
