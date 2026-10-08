# Paper-ratio NCA warm-up pivot — frozen seed-0 plan

This replaces the broad seed-0 dose sweep as the active experiment. The old `dose-sweep` runner and plotter were stopped. Completed 10M TinyStories probes for Ordinary and Designed at 9,869,186 and 29,602,341 NCA labels, plus the partially completed Ordinary 49,347,373-label probe, remain on disk. Scratch and the original 197,153,250-label runs remain unchanged. No old branch is queued to resume.

## Audit and work count

Both Ordinary and Designed NCA arms have a saved **step-0 checkpoint**, then the next saved checkpoint at **9,869,186** scored labels (step 5,000). Neither has a checkpoint near 2M or 4M. The frozen corpus and shuffled order first cross the new targets here:

| Target | Optimizer step | Exact scored NCA labels |
|--:|--:|--:|
| 2M | 1,013 | 2,001,179 |
| 4M | 2,025 | 4,001,087 |

The new NCA runs resume their arm's original step-0 **model and optimizer**. Designed runs once through 4,001,087 labels and saves both 2M and 4M checkpoints. Ordinary NCA training is deferred until a Designed dose has been chosen, then stops at the matching 2,001,179 or 4,001,087 labels. Thus new scored NCA labels total **6,002,266–8,002,174** across both arms; no corpus is generated. The short-dose saves are **checkpoint-only**: the original NCA validation schedule, full-197M optimizer schedule, and update order remain intact.

The independent variable is **NCA switch dose** (warm-up duration): the scored-label count at which the same frozen NCA optimization trajectory switches to language. The 2M and 4M models are early checkpoints of the original full-197M learning-rate schedule, not independently optimized short runs. The report will show the switch-time curve, `NCA switch dose → TinyStories validation loss at fixed downstream token counts`. Seed 0 is exploratory/model-selection data because it chooses the dose. A later confirmation would freeze that dose and evaluate seeds 1–4 without further selection; those runs are outside this plan and are not queued.

There are **three new downstream branches**: Designed 2M, Designed 4M, and one matched-dose Ordinary. The two Designed branches first run to the existing 10,000,384-token checkpoint on the unchanged 100M-token optimizer schedule. One or both then resume to the full 100M TinyStories tokens; the one Ordinary control runs to the same full horizon. Scratch is reused. Completed seed-0 timings imply roughly **2.6–3.7 hours** including continuations, plus small startup costs. All runs execute sequentially on the shared MPS device.

## Transfer-rule audit

The shared transfer path loads the saved random **language** initialization, then replaces only the Transformer-body tensors with the NCA checkpoint. It excludes `input_proj.*` and `output_proj.*` from NCA loading. A direct comparison of the completed seed-0 Designed transfer at downstream step 0 found all three language embedding/head tensors equal to the language initialization, all 73 body tensors equal to the NCA checkpoint, and none of those body tensors equal to the language initialization. The input and output weights are tied. All model parameters require gradients and the downstream Adam optimizer contains all of them. The paper-ratio branches call this same transfer path; the NCA embeddings and output head are discarded at transfer.

## Fixed selection and stopping rule

Compare Designed 2M and 4M at the six logged TinyStories checkpoints near **1M, 2M, 3M, 5M, 7.5M, and 10M**, with Scratch and normalized 0–10M AUC. Extend both Designed conditions unless one is worse than the other at **at least five of six** checkpoints by ≥0.02 loss **and** worse at the 10M endpoint and in AUC by ≥0.02. Also extend only the other if one is worse than both it and Scratch at **all six** checkpoints by ≥0.03 loss and has AUC worse than the other by ≥0.03. This rule never uses one isolated checkpoint to reject a condition.

After the selected Designed condition(s) reach 100M TinyStories tokens, choose the matched Ordinary dose by lowest Designed 100M validation loss. When losses differ by ≤0.01, use lower normalized 0–100M AUC; any remaining tie favors 2M. Train **one Ordinary NCA checkpoint and one Ordinary downstream control** at that dose. Do not start seeds 1–4, new selectors, corpus generation, or any further branches automatically.

The [runner](paper_ratio.py) uses the original training function, transfer operation, saved TinyStories stream, batch sizes, tokenizer, validation milestones, architecture, model seed, and full-stage LR schedules. It pauses immediately after the standard downstream 10M checkpoint; model and optimizer are resumed for the 100M continuation. The [read-only report](paper_ratio_report.py) writes `PAPER_RATIO_RESULTS.md` and figures after the two Designed probes, then again after the matched Ordinary control. New artifacts are isolated under `runs/scale-transfer-10m/paper-ratio/`.

The [live plot](paper_ratio_live.html) refreshes the saved validation-loss curves during the run. Its [plotter](paper_ratio_live_plot.py) reads only existing `metrics.json` files, including reused Scratch and older Designed doses; it does not affect training. A static snapshot is at `runs/scale-transfer-10m/paper-ratio/live_loss.png`.
