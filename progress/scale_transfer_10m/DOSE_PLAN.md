# Frozen seed-0 NCA-dose diagnostic

This reuses the completed seed-0 10M-scale run. The five-seed replication was stopped at the user's request; this diagnostic does not resume or alter it. Both packed NCA corpora, the 100M+1-token TinyStories stream, validation data, model, transfer function, optimizer, full 100M-token learning-rate schedule, and evaluation milestones remain frozen. No NCA training or corpus generation is scheduled.

## Saved seed-0 NCA checkpoints

Ordinary and Designed each have all **30** checkpoints below at the same scored-label counts. The files live in `runs/scale-transfer-10m/nca/{ordinary,designed}/checkpoints/model_{step}_iter.pth`. Counts come from the frozen shuffled training-order prefix, not rounded filenames.

| Step | Scored labels | Step | Scored labels |
|--:|--:|--:|--:|
| 0 | 0 | 50,661 | 100,001,521 |
| 5,000 | 9,869,186 | 55,000 | 108,567,040 |
| 10,000 | 19,732,563 | 60,000 | 118,437,225 |
| 10,136 | 20,000,398 | 60,793 | 120,001,904 |
| 15,000 | 29,602,341 | 65,000 | 128,308,520 |
| 20,000 | 39,480,925 | 70,000 | 138,173,488 |
| 20,263 | 40,000,568 | 70,927 | 140,001,647 |
| 25,000 | 49,347,373 | 75,000 | 148,042,896 |
| 30,000 | 59,210,750 | 80,000 | 157,915,856 |
| 30,401 | 60,001,658 | 81,055 | 160,000,424 |
| 35,000 | 69,080,195 | 85,000 | 167,789,186 |
| 40,000 | 78,956,670 | 90,000 | 177,658,335 |
| 40,529 | 80,000,731 | 91,187 | 180,000,918 |
| 45,000 | 88,823,858 | 95,000 | 187,529,112 |
| 50,000 | 98,694,265 | 99,875 | 197,153,250 |

The first positive checkpoint is 9.87M labels, so there is no distinct 5M dose. The selected grid uses the closest saved checkpoint to each target:

| Requested | Actual scored labels | Step | New downstream run per arm? |
|--:|--:|--:|:--|
| 10M | 9,869,186 | 5,000 | Yes |
| 25M | 29,602,341 | 15,000 | Yes |
| 50M | 49,347,373 | 25,000 | Yes |
| 100M | 100,001,521 | 50,661 | Yes |
| 150M | 148,042,896 | 75,000 | Yes |
| 200M | 197,153,250 | 99,875 | No; original history reused |

## Execution count and gate

- Reuse the existing seed-0 Scratch, Ordinary 197.15M, and Designed 197.15M downstream histories and all saved NCA checkpoints. **No baseline reruns.**
- Run **10 new seed-0 downstream branches**: five shorter doses × two NCA arms. Each first reaches the existing 10,000,384-token evaluation checkpoint while retaining the original **100M-token optimizer schedule**. This is about 1.3 hours of downstream training at the completed seed-0 10M-stage rate, excluding startup and contention.
- After all ten probes, extend at most **two shorter doses per arm** to 100M TinyStories tokens. A shorter dose qualifies if its 10M loss or normalized 0–10M AUC is at least **0.01 lower** than that arm's original 197.15M result. Choose the best qualifying dose by 10M loss and the best by AUC; ties favor the smaller NCA dose. If neither metric qualifies, extend none for that arm. The fixed rule limits continuation to at most four runs and uses no new NCA feature selection.
- Run downstream branches sequentially on the shared MPS device. Concurrent GPU training with the prior replication or among dose branches would add contention; there is no practical speed gain from parallel execution here.
- Write tables and plots after the 10M probes and again after any selected continuations. Results remain seed-0 diagnostics. There is **no automatic seeds 1–4 dose validation**.

The [dose runner](dose_sweep.py) verifies the seed-0 config, corpus hashes, checkpoint counts and model availability before training. It calls the original downstream training function and pauses it immediately after its normal 10M evaluation/checkpoint; resumption uses the saved model and optimizer on the unchanged full-stage schedule. The [report](dose_report.py) is read-only and writes `DOSE_RESULTS.md` plus three figures. A dedicated `launchd` queue writes only under `runs/scale-transfer-10m/dose-sweep/`.

The [live validation-loss plotter](dose_live_plot.py) polls only saved `metrics.json` files and writes `runs/scale-transfer-10m/dose-sweep/live_loss.png`. Its two panels show Ordinary and Designed doses separately, each against the reused Scratch and 197.15M-label anchors. It can also render one snapshot with `uv run --locked --all-packages python progress/scale_transfer_10m/dose_live_plot.py --once`.
