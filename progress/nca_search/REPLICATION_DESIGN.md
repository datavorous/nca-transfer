# Independent-corpus replication of horizontal patch-copy selection

The iteration-04 high-copy cohort improves the 1M TinyStories endpoint in 5/5 model seeds (mean paired loss gain +0.05917), but it is **one corpus** chosen after several probes. About half its rules overlap the original reference selection. Iteration 05 is a useful opposite-direction control on the same seed-0 stream; it is not an independent corpus replicate. This protocol freezes the selector and tests whether its apparent benefit survives new rule draws. No generation or training was performed for this design note.

## Locked comparison

Use **three fresh corpus blocks**, each with three arms: `high`, `unfiltered`, and `low`. Within a block, generate from one new JAX rule stream. Assign each attempted rule to an arm by a domain-separated hash of its rule seed **before** computing or using the feature; assign its train/validation/test split with the existing `split_for` hash. This makes arm membership independent of the measured feature and ensures no rule is shared between arms. Use different, recorded nonzero stream seeds for blocks A/B/C. Check that no rule seed occurs in another block or in any earlier reference/search corpus; if a duplicate appears, skip it and record it.

The frozen horizontal-copy score is the existing `horizontal_copy_excess`: horizontal adjacent 2×2 patch-copy fraction minus `sum_id n_id(n_id-1)/(972*971)`. Its threshold is **-0.000374227071**, fixed from the original 25,000-world reference scan. `high` accepts score >= threshold; `low` accepts score <= threshold; `unfiltered` ignores score. Keep the original 40–50 gzip interval and exact train/validation quotas per 0.02-wide gzip bin:

| Split | .40–.42 | .42–.44 | .44–.46 | .46–.48 | .48–.50 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 2,913 | 3,544 | 4,034 | 4,559 | 4,925 | 19,975 |
| Validation | 45 | 42 | 49 | 57 | 63 | 256 |

Each arm therefore has **19,715,325 usable NCA labels** and the original 1,026-token world layout, int32 packed tensors, markers, masks, tokenizer, and split semantics. Keep the original GameConfig except for the recorded rule-stream seed. Do not alter the NCA rule distribution, trajectory seed derivation, model, optimizer, or TinyStories corpus. Record full empirical gzip distributions, feature means, counts and rule IDs. If a block cannot fill all cells at a prespecified attempt cap, report it as incomplete rather than relaxing the threshold or quotas after seeing results.

## Training and decision sequence

1. **Freeze before running:** commit/record this protocol, threshold, arm hash, stream seeds, quotas, output metrics, code SHA-256, and candidate ordering. Do not tune the threshold on these new blocks. The earlier 1M loss is discovery evidence only.
2. **Sample and audit block A first, then B/C:** verify exact quotas, 987 labels per training world, packed-data hash, `pack_one` agreement, disjoint rule IDs, and train/validation split isolation. Inspect feature separation and within-bin gzip means/CDFs. If the selector also shifts other token properties, report them; matching gzip does not remove those confounds.
3. **Cheap paired screen:** for every completed block, pre-pretrain all three arms using the *same* original seed-0 base initialization and NCA budget, then run the original full 10M TinyStories scheduler only through the logged **1,001,472-token** checkpoint. Reuse the exact TinyStories train/validation bins and order seed 17, training seed 23, evaluation code, and checkpoint steps. The primary replication contrast is `validation_loss(unfiltered) - validation_loss(high)` at that checkpoint, computed separately within each block. Secondary contrasts are `low - high` and `unfiltered - low`; also report normalized 0–1M loss AUC and each milestone. Do not compare NCA validation losses across filtered arms as a transfer metric: they have different held-out distributions.
4. **Confirm only after all block-A/B/C seed-0 results are visible:** if high beats unfiltered in all three blocks and low in at least two, run the remaining original model initializations **1–4** for every arm in those same blocks. This costs up to 45 NCA/1M runs total including seed 0; the staged first pass costs nine. If the seed-0 replication fails, retain all results and stop the expensive follow-up instead of picking a favorable block.
5. **Full-curve check:** if the three-block 1M endpoint pattern survives model-seed replication, continue the saved downstream optimizer states to 10M under the unchanged schedule. Report the complete milestone curves and 0–10M loss AUC as distinct outcomes. A 1M endpoint gain alone does not establish an early-curve or final advantage.

Treat the **corpus block** as the independent replicate. Five initialization seeds within a block describe model-seed sensitivity but do not create five independent corpus observations. Report each block's mean paired arm gain across the five seeds, their three-block mean and range, win counts at both block and model-seed levels, and the corresponding AUC and 10M summaries. Three blocks are a minimum robustness screen, not enough for a precise population confidence interval; avoid a seed-level t test that treats 15 runs as independent. The previous exploratory selection makes any unadjusted significance claim optimistic.

## Proposed artifact layout and implementation boundary

Add a dedicated sampler and new output paths; reuse the unchanged NCA/TinyStories probe runner with explicit replication flags. A concrete layout is:

```text
progress/nca_search/sample_patch_replication.py
progress/nca_search/probe_low_change.py
runs/nca-data-search/patch-replication/{block-a,block-b,block-c}/
  sampling_config.json
  rule_index.jsonl
  {high,unfiltered,low}/packed.pt
  {high,unfiltered,low}/done.json
  {high,unfiltered,low}/seeds/seed_{0,1,2,3,4}/...
```

Expected command shape after implementation:

```sh
uv run --all-packages python progress/nca_search/sample_patch_replication.py --block a
uv run --all-packages python progress/nca_search/probe_low_change.py --block a --replication-arm high --seed 0
```

The sampler uses the same rollout, `gzip_ratio`, `horizontal_copy_excess`, `split_for`, quota and packing rules as `sample_low_change.py`; `probe_low_change.py` reuses its unchanged `train_stage` and TinyStories schedule for the new arm paths. New output roots, config receipts and content hashes make runs resume-safe without touching earlier artifacts. The arm-assignment hash must use a domain/person distinct from `nca-split`; record its exact bytes. Verify checkpoint token, step, learning-rate and text-validation equality between arms before comparing losses.

Run `uv run --all-packages python progress/nca_search/audit_patch_replication.py --block a` after sampling and before training. It rebuilds `rule_index.jsonl`, checks each selected seed against the saved JAX stream and arm/split hashes, verifies counts and threshold directions, and rejects any overlap with original receipts, earlier search corpora, or other blocks. If overlap is found, invalidate and resample the block with duplicate skipping before interpreting it.

## Frozen block streams and first implementation

Before starting block A, the rule-stream seeds were fixed to **101, 202, and 303** for blocks A, B, and C. The new [sampler](sample_patch_replication.py) uses `jax.random.PRNGKey(stream_seed)` with the original rollout and trajectory seed derivation. Arm assignment uses `blake2b(rule_seed.tobytes(), digest_size=8, person=b"nca-arm")`, scaled to `[0,1)` and divided into equal thirds for `high`, `unfiltered`, and `low`; `split_for` keeps `person=b"nca-split"`. Arm and split are determined before trajectory generation or feature measurement. The high/low threshold stays **−0.000374227071**. Its resume-safe `sampling_config.json` records these choices and the original per-arm gzip quotas. The original iteration-04/05 and reference corpora remain untouched.

## What this can and cannot establish

If high repeatedly beats both independently drawn controls, the evidence supports **selecting high-copy NCA corpora** as a useful data policy for this fixed transformer/TinyStories setup. It would quantify sensitivity to corpus draw and initialization. It still would not isolate adjacent copying as the causal mechanism: high-copy selection changes entire rules and may move unmeasured correlated properties. A mechanism test would require a later intervention that changes copy structure while holding rule identity or other token statistics as fixed as possible. All inference remains specific to this NCA generator, gzip band, patch tokenizer, model scale and downstream dataset until replicated outside them.
