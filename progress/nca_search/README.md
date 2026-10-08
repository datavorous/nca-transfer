# NCA data search

**Current phase:** finish the already-running 45-arm horizontal replication, publish [one final blog analysis](BLOG_FINAL_ANALYSIS.md), then stop. Matched-copy, vertical-model, selector-search, horizontal 10M continuation, and the proposed single designed-NCA run are paused; no prepared runner should be launched under the earlier gates. The analysis file is a draft until the horizontal matrix completes.

The fixed comparison is the completed `runs/1m-nca-transfer-v2/` 40–50 gzip condition. Candidate corpora keep the original NCA rule family, 2×2 patch tokenizer, 19,975 training worlds, 256 validation worlds, 19,715,325 usable NCA labels, five gzip sub-bin quotas, ~1M-parameter transformer, and TinyStories stream. New artifacts are under `runs/nca-data-search/`; the original runs are untouched.

## Current result

Selecting trajectories with **above-median horizontal adjacent-patch copy beyond fixed patch-frequency chance** gave lower TinyStories validation loss at the 1M-token checkpoint in all five model seeds. The opposite low-copy selection was worse than the unfiltered reference in all five. Mean 1M losses: **high 5.95409, reference 6.01326, low 6.05601**. The high cohort retained a smaller 10M mean gain (4/5 seeds; **4.16622 versus 4.18486**), with an exploratory interval crossing zero. The result is from one rule stream and was found after several probes; [independent corpus replication](REPLICATION_DESIGN.md) is the next required robustness test.

Three fresh, disjoint rule-corpus blocks have completed their seed-0 1M screens. High-copy beats each block's unfiltered and low-copy arm at the endpoint: unfiltered-minus-high gains are **+0.08426, +0.07415, +0.04103** (mean **+0.06648 ± 0.02261** across blocks). High also beats unfiltered on 0–1M loss AUC in all three, with a small margin in block C. This is promising independent-corpus evidence, still only one model initialization per block in the reported comparison; the [model-seed follow-up](REPLICATION_DESIGN.md) is running. [Live machine-readable results](replication_1m.csv).

Blocks A and B now have all five model seeds. High copy beats unfiltered at 1M in **4/5** A seeds and **5/5** B seeds, with mean paired gains **+0.05686** and **+0.04543** respectively. The early-curve AUC wins are **4/5** and **3/5**; block A seed 2 reverses the endpoint comparison, while block B seed 3 favors low copy and has worse high-copy AUC. Block C seeds 1–4 remain, so the all-block gate has not been decided. Model seeds within a block are not independent corpus draws.

## Records

- [Experiment log](EXPERIMENT_LOG.md): six completed controlled iterations and measured outcomes.
- [Decision and stopping rule](STOPPING_RULE.md): finite gates for the ongoing three-block follow-up, matched-copy mechanism test, and vertical candidate.
- [1M machine-readable probes](experiments.csv), [high-copy 10M curves](full_10m.csv), and [high/reference/low control](patch_copy_direction.csv).
- [Candidate audit](ITERATION_04_AUDIT.md): gzip, other feature shifts, rule overlap, packed tensor contract.
- [Reference world hypotheses](HYPOTHESES.md), [transformer token targets](TRANSFORMER_TARGETS.md), [attention-lag targets](ATTENTION_LAGS.md), [cross-world diversity](CORPUS_DIVERSITY.md), and [generator-design targets](RULE_DESIGN.md).
- Focused read-only screens: [vertical lag-6](VERTICAL_LAG_TARGET.md), [visible two-frame recurrence](TEMPORAL_TARGET.md), [cell-transition information at visible patch targets](TRANSITION_INFO_TARGET.md), and [late patch-entropy stability](ENTROPY_STABILITY_TARGET.md). These report feature associations and probe designs, not language-transfer outcomes.
- [Prioritized next targets](NEXT_TARGETS.md) and [independent-corpus replication design](REPLICATION_DESIGN.md).
- [New transfer targets](NEW_TRANSFER_TARGETS.md): visible vertical copying, held-out horizontal boundary prediction, and visible two-frame recurrence; these are read-only hypotheses.
- [Visible successor-reuse screen](SUCCESSOR_REUSE_TARGET.md): an order-sensitive relation with residual variation after frequency correction, but substantial gzip and bigram confounding; lower priority.
- [Frozen vertical lag-6 protocol](VERTICAL_LAG6_FROZEN_PROTOCOL.md): full 25,000-world reference cutoff, feasibility and a fixed next sampling/probe plan; block D sampling and contract audit are complete.
- [Vertical sampler](sample_vertical_lag6.py) and [independent audit](audit_vertical_lag6.py): block D passed exact quotas, rule isolation, tensor layout and packed-hash checks.
- [Block D feature audit](VERTICAL_LAG6_BLOCK_D_FEATURE_AUDIT.md): visible vertical copying separates as intended at closely matched gzip, while lag-76 recurrence, lag-2 recurrence and transition MI also shift.
- [Vertical 1M probe](probe_vertical_lag6.py), [visible-target diagnostic](diagnose_vertical_lag6_nca.py), and [summary](summarize_vertical_lag6.py): fixed training path and read-only diagnostic, ready after sampling/audit; no vertical model has started.
- [Corpus-level predictor audit](TRANSFER_PREDICTOR_AUDIT.md): horizontal copy, lag-2 recurrence, and transition MI all track the fresh-arm ordering, so their causal roles remain unresolved.
- [Rule-coupling screen](RULE_COUPLING_SCREEN.md): effective neighbor sensitivity of saved NCA rules shows little relation to patch copying or held-out neighbor prediction; a high-coupling selector ranks below the frozen vertical target.
- [Fresh-rule block sampler](sample_patch_replication.py) and [rule-index audit](audit_patch_replication.py) for the ongoing high-copy replication.
- [Matched-copy feasibility screen](PATCH_COPY_MECHANISM_FEASIBILITY.md): estimates whether a later high/low test can balance lag-2 recurrence and transition MI while retaining the copy contrast; no transfer result is claimed.
- [Controlled-copy protocol](CONTROLLED_COPY_PROTOCOL.md): a prospective matched high/low test after the active replication gate.
- [Matched-copy implementation](MATCHED_COPY_IMPLEMENTATION.md): frozen 45-cell quotas, sampler, independent audit, and fixture checks for three new rule streams; no matched corpus or transfer model has run.
- [Matched-copy 1M probe](probe_matched_copy.py) and [paired summary](summarize_matched_copy.py): prepared seed-0 transfer path, gated on all three passing corpus audits; no model has run.
- [Gated horizontal 10M continuation](gate_patch_replication_10m.py): checks all 45 completed 1M runs and the block-level decision before any continuation. [Continuation runner](run_patch_replication_10m.sh) is prepared but has not started.
- [Horizontal closeout table](render_horizontal_closeout.py): read-only, refuses incomplete 45-arm data and prints every paired seed with block-aware endpoint, AUC, and perplexity summaries for the blog document.
- [Full-curve audit and summary](summarize_patch_replication_10m.py): verifies every completed 10M receipt and exact original milestones before reporting paired endpoints and AUC; no 10M data exist yet.
- [Packed metadata compatibility](PACKED_METADATA_COMPATIBILITY.md): records the search-only provenance key differences and read-time compatibility expressions.

The read-only screen scripts alongside these documents reproduce their descriptive statistics. `sample_low_change.py` generates filtered corpora with `--high`, `--spatial-high`, `--patch-high`, `--patch-low`, or `--transition-high`; `probe_low_change.py` runs the matching 1M probe with the same selector flag and optional `--seed 0..4`. `continue_patch_10m.py --seed 0..4` resumes the high-copy probes to 10M. Each script uses `uv run --all-packages python` from the repository root and skips completed stages. Iteration 06's high-transition-MI corpus lost the seed-0 1M probe by 0.05833 validation-loss units; its full record is in the experiment log.
