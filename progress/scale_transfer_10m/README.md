# Frozen 10× model scale transfer

One model seed (0), three downstream conditions, and no selector search:

1. Scratch
2. Ordinary NCA from the original 40–50% gzip packed corpus
3. Designed NCA from the frozen Top-10 horizontal-recurrence packed corpus

The text model is the same tied-embedding causal Transformer family as the 1M experiment, scaled to 8 layers, 4 heads, width 144, context 256: **9,943,921 trainable text parameters**. Both NCA arms train for ten shuffled passes through their fixed 19,975-world training split. Each therefore processes exactly **197,153,250 scored synthetic labels**. Reusing the frozen corpora holds their empirical distributions and quota/threshold recipes fixed; this is a training-budget scale test, not a fresh-rule-population test. Every downstream arm processes 100,000,000 scored TinyStories tokens in the same order and uses the original GPT-2 tokenizer, dataset revision, validation stream, initialization seed, Adam settings, and effective batch eight.

The queue is `run.py`: prepare the 100M-token text stream, save one common text initialization, train ordinary and designed NCA stages, then train scratch, ordinary NCA, and Designed NCA downstream in that order. Stages resume from their latest complete checkpoint. All results stay under ignored `runs/scale-transfer-10m/`. `--smoke` runs two real MPS updates per condition with tiny validation slices in a temporary directory. The full queue starts only after smoke passes.

The training code logs token counts, loss/PPL, per-interval tokens/sec, and elapsed stage time. `telemetry.csv` samples process CPU/RSS and battery voltage/current once a minute. On this Mac, `powermetrics` requires administrator privileges and is unavailable to the unprivileged overnight runner; the machine is on AC, so battery current is **not** total system power. `summary.py` will report energy-to-target only if a valid measured energy series is available; otherwise it reports `null` with the reason. No power value is fabricated.

The full experiment is unattended and may run longer than one night. It launches exactly these three downstream conditions and starts no follow-up experiment.

## Live loss plot

The runner logs training loss every 100 optimizer steps. Poll it into an updating PNG:

```sh
uv run --locked --all-packages python progress/scale_transfer_10m/live_plot.py
```

Open `runs/scale-transfer-10m/live_loss.png` in the IDE. Add `--show` for a live Matplotlib window, or `--once` to save one snapshot and exit. The two panels keep NCA and TinyStories losses separate; colored lines are exponential moving averages and faint lines are the logged values. The plotter reads logs only and does not start or interrupt training.
