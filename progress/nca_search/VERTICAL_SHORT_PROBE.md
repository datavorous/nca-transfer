# Vertical lag-6 short transfer screen

**Paused by the user's latest instruction. Do not resume or launch any arm.** The high arm's full NCA stage completed before the instruction changed. Its TinyStories stage wrote an initial validation/checkpoint at zero downstream tokens, then was interrupted before any later checkpoint. The unfiltered and low models were not started. There is no vertical transfer result.

This protocol reuses the audited block-D high, unfiltered, and low corpora and the existing `probe_vertical_lag6.py` path. It does not change the NCA rule stream, corpus, transformer, base initialization, optimizer, TinyStories data/order, or learning-rate horizon. The initial NCA pretraining stage consumes the full **19,715,325 usable NCA labels per arm**. Only the downstream TinyStories run is shortened.

For seed 0, run each arm to 500K TinyStories tokens using:

```sh
uv run --all-packages python progress/nca_search/probe_vertical_lag6.py --block d --arm high --seed 0 --stop-tokens 500000
uv run --all-packages python progress/nca_search/probe_vertical_lag6.py --block d --arm unfiltered --seed 0 --stop-tokens 500000
uv run --all-packages python progress/nca_search/probe_vertical_lag6.py --block d --arm low --seed 0 --stop-tokens 500000
```

The actual deterministic checkpoints on the fixed seed-17 TinyStories order are **100,352** (step 49), **251,904** (step 123), and **501,760** (step 245) scored tokens. Validation uses the original full 262,144-token validation stream and causal mask. The 100K evaluation is saved separately as `early_100k.json` so the canonical history remains at 0, 250K, 500K, and eventually 1M. All checkpoints stay in the existing `downstream/probe_1m/checkpoints/` directory. A later explicitly authorized `--stop-tokens 1000000` resumes the same optimizer and original 4,883-step 10M schedule; the 1M checkpoint remains at step 489 and 1,001,472 scored tokens. No `done.json` is written by a short run.

The short screen is **paired within this one corpus block and seed**. Compare validation loss at each checkpoint as `unfiltered − high` and `low − high`, with lower loss better. A positive high-copy result in the separate horizontal experiment does not establish vertical additivity; this screen only ranks vertical arms. The original ordinary-NCA reference has logged checkpoints at 251,904 and 501,760 but **no matched 100K point**. Do not interpolate its 0–250K curve and call that a measured 100K baseline. The 100K triplet is an arm-to-arm early screen; the 250K and 500K points can also be compared directly with the original seed-0 reference. One block and seed are triage evidence, not a replicated effect.

**Decision rule fixed before reading outcomes:** regard vertical selection as a promising recipe component only if high is lower than both contemporaneous controls at 501,760 tokens, its unfiltered-minus-high advantage is at least **0.03 validation loss**, and high also beats unfiltered at 251,904 tokens and on normalized trapezoidal 0–501,760 AUC (using 0, 251,904, and 501,760 points). Report 100K separately. This threshold is a practical screening rule, not a significance test. If it passes, repeat the high-versus-unfiltered contrast with another model seed before making a combined corpus. A combination must be tested against contemporaneous horizontal-high and unfiltered controls under the same gzip quotas and NCA budget; a positive vertical-only screen does not imply additivity. Do not retune the cutoff or select a favorable checkpoint after seeing results.

Use the existing block-D audit and feature-audit receipts to report vertical manipulation, gzip balance, and correlated horizontal recurrence, lag-2 recurrence, transition MI, and lag-76 recurrence. The screen cannot identify a unique causal mechanism. Do not infer an advantage from whichever of the three checkpoints is best; report all three and the curve direction.
