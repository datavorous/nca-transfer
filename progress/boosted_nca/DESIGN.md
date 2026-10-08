# Boosted NCA generator: reference design

The completed [dose sweep](SWEEP_PROTOCOL.md) used this document's fixed top-25/top-10 cutoffs, the existing block-A high corpus as the top-50 anchor, and one model seed through 1M TinyStories tokens. It excluded the optional vertical knob. The horizontal A/B/C replication remains separate and unchanged. **Top-10 was selected and its seed-0 checkpoint resumed to 10M.** See [SWEEP_RESULTS.md](SWEEP_RESULTS.md) for outcomes and limits.

## Generator and selection knobs

Keep the original NCA rule family and rollout, 12×12 ten-color states, 27 saved frames after burn-in 10, and 2×2 patch tokenizer. For each generated world, compute the existing `horizontal_copy_excess`: the fraction of equal adjacent 2×2 patch IDs within each patch row, minus `sum_id n_id(n_id−1)/(972×971)` over all 972 patch IDs. Exclude row boundaries. Filter gzip to **[.40, .50)** and fill the *same* five train/validation sub-bin quotas as the reference. Rank stronger cohorts **within each gzip sub-bin**, using cutoffs frozen from the first 25,000 accepted reference `40_50` worlds in receipt order. Quantiles use NumPy's linear interpolation; accept scores `>=` the tabulated cutoff. Ties can make the realized fraction slightly larger.

| Gzip bin | Reference worlds | Top-50% anchor | Top-25% cutoff | Top-10% cutoff |
|:--|--:|--:|--:|--:|
| .40–.42 | 3,640 | −0.000374227071 | +0.013787809437 | +0.034107767225 |
| .42–.44 | 4,435 | −0.000374227071 | +0.009925705543 | +0.025746229122 |
| .44–.46 | 5,013 | −0.000374227071 | +0.006992070455 | +0.017753705187 |
| .46–.48 | 5,766 | −0.000374227071 | +0.004674553831 | +0.012177637072 |
| .48–.50 | 6,146 | −0.000374227071 | +0.003439879976 | +0.008144630498 |

The top-50% anchor deliberately retains the **exact global-median selector already tested** in the horizontal replication. Every top-25% and top-10% bin cutoff exceeds that anchor, so the stronger tiers are subsets of its eligible worlds. The score and tokenizer remain identical. Only the acceptance threshold changes. Do not infer that a stricter tier transfers better before a downstream screen.

The sweep kept the verified block-A top-50% corpus and continued its JAX rule stream (seed **101**) after the saved cursor. It reused eligible high-arm worlds already saved by block A, then accepted only rules assigned to that same high arm by `person=b"nca-arm"`, with the original `nca-split` split hash. Worlds can occur in more than one tier; the top-25/top-10 overlap was **8,121**. New chunks and receipts are under `runs/nca-data-search/boosted-nca/sweep-a/`; the packed int32 `[world,1026]` contract, markers, masks, 19,975 train worlds, 256 validation worlds, and **19,715,325** usable NCA labels per cohort were audited. Both tiers filled at **2,916,352 attempted rules**, below the frozen **5,000,000-attempt cap**. [Sweep protocol](SWEEP_PROTOCOL.md); [audit receipt](../../runs/nca-data-search/boosted-nca/sweep-a/audit.json).

Vertical, MI, lag-76, matched-copy and other mechanism selectors are outside this sweep.

## Expected acceptance and feasibility

The first 25,000 saved reference worlds give the following conditional eligibility among worlds already in gzip [40,50): the global top-50% anchor passes **49.3–50.8%** across the five bins; bin-specific top-25% passes **25.0%** and top-10% **10.0%** in each bin, up to quantile ties. These are saved-corpus estimates, not guarantees on the continuing block-A stream.

The block-A anchor needed **577,920 total attempted rules** for all three preassigned arms. Within that saved high arm, **10,207** worlds already qualify for top-25% and **4,053** for top-10%. Because the continued sweep keeps the one-third high-arm assignment, a rough inverse-rate estimate puts top-25 fill near **1.2M** and top-10 near **2.8M cumulative attempted rules**; the observed final count is the authority. Both tiers share the continuation stream, so total attempts are governed by the harder top-10 quota, not the sum. Sparse cells can raise cost; **5M** is a ceiling. A global top-10% cutoff of `+0.017141210326` would admit only **2.65%** of worlds in the .48–.50 bin, motivating the bin-specific cutoffs. Audit the achieved gzip CDF and feature distributions before training.

## One-seed 1M screen

The existing block-A top-50% seed-0 model is the anchor. Train each new tier with one seed-0 initialization and the **full 19,715,325-label NCA stage**, then continue the fixed TinyStories stream and original 10M-token learning-rate schedule through **1,001,472 downstream tokens**. Compare first downward crossing times for the frozen loss targets **7.0, 6.5 and 6.25**, summarized by their geometric mean. The [sweep protocol](SWEEP_PROTOCOL.md) specifies the 2%-per-target top-5 gate and 1% conservative tie rule. The one-stream, one-model-seed ranking is exploratory, even though the earlier top-50 policy was replicated across fresh blocks.

## Final canonical 10M-run configuration template

[canonical_10m.template.json](canonical_10m.template.json) records the frozen original NCA budget, architecture, tokenizer, TinyStories data/order, optimizer and evaluation milestones. It remains the predecision design artifact; the actual choice and checkpoint hashes are in [decision.json](../../runs/nca-data-search/boosted-nca/decision.json). Per the user's explicit choice, the final 10M stage **resumed the winning seed-0 model's saved 1M checkpoint**. The original seed-0 scratch and ordinary-NCA 10M curves are historical controls, and a fresh contemporaneous comparison would require a separate experiment.
