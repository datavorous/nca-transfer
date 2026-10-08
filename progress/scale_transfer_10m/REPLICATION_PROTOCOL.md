# Frozen 10M-scale model-seed replication

Frozen before seeds 1–4. Seed 0 remains under `runs/scale-transfer-10m/`; new runs write only under `runs/scale-transfer-10m/seeds/seed_N/`. The existing [seed-0 config](../../runs/scale-transfer-10m/config.json), [ordinary packed corpus](../../runs/1m-nca-transfer-v2/data/nca/40_50/done.json), [Designed Top-10 receipt](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/done.json), and [TinyStories metadata](../../runs/scale-transfer-10m/data/tinystories/metadata.json) are the frozen inputs. No corpus or language data is regenerated.

| Frozen input | SHA-256 or setting |
|:--|:--|
| Ordinary 40–50 packed `.pt` | `1ba5e8ce68c70faf19183170ac107959d5348e8db220898baa20bb9668d7ffbf` |
| Designed Top-10 packed `.pt` | `a992f78a255b7c896eee6a8e77fad57631fc866239af8ab1b05edb1ea9573824` |
| TinyStories 100M+1-token train `.bin` | `175daf984ce5a5ed0d2fa7b2d03652e6eeeefdf1b0b87959b738faf7b9c6d05a` |
| Original validation `.bin` | `851de2118d56178237f2b87270d02803433dd7fb6b4b5c841bb05afb4a5385e5` |
| Top-10 within-bin copy thresholds | `.034107767225 / .025746229122 / .017753705187 / .012177637072 / .008144630498` |
| Model | 8 layers, 4 heads, width 144, context 256, tied embeddings; text model 9,943,921 parameters |
| Training | Adam, LR .0003, weight decay 0, warmup 100, cosine over full stage, effective batch 8, NCA/text physical batches 4/2 |
| Order and update seeds | 17 / 23; fresh paired model initialization seeds 1–4 |
| NCA labels | 197,153,250 scored labels per arm, ten shuffled passes over the same frozen 19,975-world split |
| Downstream tokens | 100,000,000 per condition in identical order; same validation data and 15 checkpoints as seed 0 |

For each seed, save one initial text model and use it for Scratch and both NCA transfer arms. The NCA model starts from its matching text body's weights. The downstream Ordinary and Designed text models start from the same seed's text weights, then load the corresponding NCA body. All three conditions share the same TinyStories train order and validation stream. NCA and downstream validation schedules, token budgets, architecture, transfer operation, and optimizer are identical to seed 0. Seed 0 is read only.

## Seed-0 intermediate NCA checkpoint inventory

Both Ordinary and Designed have all **30** saved checkpoints below. The label count is the deterministic shuffled training-order prefix at the checkpoint's optimizer step, not a rounded filename. These files remain in place and are not modified.

| Step | Scored NCA labels | Step | Scored NCA labels |
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

New seeds retain the same periodic and evaluation checkpoints, plus checkpoint-only saves at the first optimizer steps reaching **25M, 50M, 100M, and 150M** scored labels. The final 197,153,250-label checkpoint represents the approximately 200M-label budget. The extra saves do **not** add validation evaluations or alter training updates. Per-seed `checkpoint_labels.json` files record exact counts.

## Measurement plan

The [read-only summarizer](summarize_replication.py) runs only after all four new seeds complete. It reports every logged validation loss/PPL, paired D−S/O−S/D−O differences, first Designed–Scratch curve crossing, and normalized AUC over **0–1M, 0–10M, 0–100M, and 10–100M** downstream tokens. The fixed target-loss grid is **3.0, 2.5, 2.25, 2.0, 1.9**; first downward crossing uses only linear interpolation between logged checkpoints. Mean ± sample SD and paired win counts use the five model seeds. It draws each seed faintly and the mean prominently. No interim outcome determines any additional training or setting.

The [sequential launcher](run_replication_overnight.sh) starts seeds 1–4 through [replicate_seeds.py](replicate_seeds.py) under `launchd`, with automatic resume from each arm's last complete checkpoint if interrupted. The queue then invokes the summarizer once; it does not schedule a subsequent experiment. The log is `runs/scale-transfer-10m/replication/runner.log` and the final report will be `progress/scale_transfer_10m/REPLICATION_RESULTS.md`.
