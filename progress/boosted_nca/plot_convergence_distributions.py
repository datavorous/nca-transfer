"""Paired-seed convergence speedup for four ordinary NCA bands and designed NCA."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_seed_validation import BOOSTED, REPO, SOURCE, CORPUS, crossing, history


BANDS = (
    ("20–30%", "nca_20_30"),
    ("30–40%", "nca_30_40"),
    ("40–50%", "nca_40_50"),
    ("50%+", "nca_50_plus"),
    ("Designed NCA", "designed"),
)
OUT = REPO / "progress/boosted_nca/figures"


def main() -> None:
    assert json.loads((BOOSTED / "decision.json").read_text())["selected_tier"] == "top_10"
    records = []
    for label, name in BANDS:
        values = []
        targets = []
        crossings = []
        for seed in range(5):
            source = SOURCE / f"seed_{seed}/downstream"
            target = history(source / "scratch/metrics.json")[-1]["validation_loss"]
            if name == "designed":
                run = (BOOSTED / "final-10m" if seed == 0 else
                       CORPUS / f"seeds/seed_{seed}/downstream/full_10m")
            else:
                run = source / name
            done = json.loads((run / "done.json").read_text())
            assert done["tokens_consumed"] == 10_000_000
            curve = history(run / "metrics.json")
            assert curve[-1]["validation_loss"] == done["final_validation_loss"]
            t = crossing(curve, target)
            assert t is not None and t > 0, (name, seed)
            targets.append(target)
            crossings.append(t)
            values.append(10_000_000/t)
        records.append({"label": label, "condition": name,
                        "speedup_by_seed": values,
                        "scratch_target_loss_by_seed": targets,
                        "crossing_tokens_by_seed": crossings,
                        "mean": statistics.mean(values),
                        "sample_sd": statistics.stdev(values)})

    assert round(records[2]["mean"], 3) == 1.766
    assert round(records[4]["mean"], 3) == 1.995
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "For each seed: target=scratch seed validation loss at 10M; first downward linear crossing of condition's nine logged checkpoints; speedup=10M/crossing tokens. Mean and sample SD across five seeds.",
        "note": "The designed top-10 distribution was selected using seed 0; seeds 1–4 are held-out model initializations on the same corpus. Ratios exclude NCA training compute.",
        "conditions": records,
    }
    (OUT / "downstream_convergence_speedup_by_distribution.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 240})
    ink, grid, teal = "#233042", "#e8edf1", "#087f88"
    grays = ("#a6b0bc", "#909daa", "#768696", "#b7c0c9")
    positions = np.array((0, 1, 2, 3, 4.55), dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 6.1), facecolor="white")
    fig.subplots_adjust(left=.09, right=.97, top=.84, bottom=.20)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=grid, linewidth=1)
    ax.spines["left"].set_color("#cbd4dc")
    ax.spines["bottom"].set_color("#cbd4dc")
    ax.tick_params(axis="both", colors=ink, length=0, pad=8)
    ax.set_ylim(0, 2.35)
    ax.set_xlim(-.62, 5.17)
    ax.set_yticks((0, .5, 1, 1.5, 2))
    ax.set_ylabel("Speedup (x)", color=ink, labelpad=10)
    ax.axhline(1, color="#bac4cd", linewidth=1, linestyle=(0, (3, 4)))
    ax.axvline(3.76, color="#dce3e8", linewidth=1, ymin=.03, ymax=.94)
    ax.set_xticks(positions, [r["label"] for r in records])
    ax.text(positions[:4].mean(), -.105, "ordinary NCA · gzip band",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            color="#657181", fontsize=8.5, clip_on=False)
    ax.text(positions[-1], -.105, "top-10% horizontal recurrence",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            color="#657181", fontsize=8.5, clip_on=False)
    for index, (x, row) in enumerate(zip(positions, records)):
        color = teal if index == 4 else grays[index]
        ax.bar(x, row["mean"], width=.69, color=color, edgecolor="none", zorder=3)
        ax.errorbar(x, row["mean"], yerr=row["sample_sd"], fmt="none",
                    ecolor=ink if index == 4 else "#536273",
                    elinewidth=1.35, capsize=5, capthick=1.35, zorder=5)
        ax.text(x, row["mean"] + row["sample_sd"] + .075,
                f"{row['mean']:.2f} ± {row['sample_sd']:.2f}x",
                ha="center", va="bottom", color=teal if index == 4 else ink,
                fontsize=10)
    fig.text(.09, .94, "Downstream convergence speedup by synthetic pretraining distribution",
             ha="left", va="top", color=ink, fontsize=16)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"downstream_convergence_speedup_by_distribution.{suffix}",
                    facecolor="white")
    plt.close(fig)
    for row in records:
        print(f"{row['label']}: {row['mean']:.3f} ± {row['sample_sd']:.3f}x")


if __name__ == "__main__":
    main()
