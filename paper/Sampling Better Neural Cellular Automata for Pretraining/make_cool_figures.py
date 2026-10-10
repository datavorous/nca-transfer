"""Plot frozen 1M-parameter results only.

Run from SampleEff: uv run --locked --all-packages python
    'paper/Sampling Better Neural Cellular Automata for Pretraining/make_cool_figures.py'
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE / "figs_cool"
RUN = PROJECT / "runs/1m-nca-transfer-v2/seeds"
# Exact gray and green from the original five-seed 1M comparison figure.
INK, BASE, GRID, SPINE = "#233042", "#FFFFFF", "#e8edf1", "#cbd4dc"
GRAY, GREEN = "#89919e", "#087f88"
SCRATCH = SHUFFLED = LOW = EXPLORATORY = GRAY
ORDINARY = HIGH = GREEN

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": BASE, "axes.facecolor": BASE,
    "savefig.facecolor": BASE,
})


def baseline(condition):
    curves = []
    for seed in range(5):
        path = RUN / f"seed_{seed}/downstream/{condition}/metrics.json"
        rows = json.loads(path.read_text())["history"]
        x = np.array([r["tokens"] for r in rows], dtype=float) / 1e6
        y = np.array([r["validation_loss"] for r in rows], dtype=float)
        curves.append((x, y))
    assert all(np.array_equal(curves[0][0], x) for x, _ in curves[1:])
    return curves[0][0], np.stack([y for _, y in curves])


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(length=3, width=0.7)


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    for extension in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}",
                    bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def reproduction():
    curves = [
        ("scratch", "Scratch", SCRATCH),
        ("nca_40_50", "Ordinary NCA", ORDINARY),
        ("shuffled_nca", "Shuffled NCA", SHUFFLED),
    ]
    endpoints = [
        ("scratch", "Scratch", SCRATCH),
        ("shuffled_nca", "Shuffled", SHUFFLED),
        ("nca_20_30", "20–30%", ORDINARY),
        ("nca_30_40", "30–40%", ORDINARY),
        ("nca_40_50", "40–50%", ORDINARY),
        ("nca_50_plus", "50%+", ORDINARY),
    ]
    fig_curves, left = plt.subplots(figsize=(5.6, 3.4))
    plotted = []
    for condition, label, color in curves:
        x, values = baseline(condition)
        marker = {"scratch": "^", "nca_40_50": "o", "shuffled_nca": "s"}[condition]
        linestyle = "--" if condition == "shuffled_nca" else "-"
        zorder = 4 if condition == "shuffled_nca" else 3
        face = BASE if condition == "shuffled_nca" else color
        mean = values.mean(axis=0)
        plotted.append((x, mean, color, marker, linestyle, face, zorder))
        left.plot(x, mean, color=color, lw=1.4, linestyle=linestyle,
                  marker=marker, markerfacecolor=face, markeredgewidth=1.0,
                  markersize=3.7, label=label, zorder=zorder)
    left.set(xlabel="TinyStories tokens (M)", ylabel="Validation loss",
             xlim=(0, 10))
    left.set_xticks([0, 1, 2, 3, 5, 7.5, 10])
    left.legend(frameon=False, fontsize=7.1, loc="upper left",
                bbox_to_anchor=(0.15, 0.98))
    style(left)
    zoom = left.inset_axes([0.51, 0.50, 0.46, 0.42])
    for x, mean, color, marker, linestyle, face, zorder in plotted:
        zoom.plot(x, mean, color=color, lw=1.1, linestyle=linestyle,
                  marker=marker, markerfacecolor=face, markeredgewidth=0.9,
                  markersize=3.5, zorder=zorder)
    zoom.set(xlim=(5, 10), ylim=(4.15, 4.56))
    zoom.set_xticks([5, 7.5, 10])
    zoom.set_yticks([4.2, 4.4])
    zoom.tick_params(labelsize=6.5, length=2, pad=1.5)
    zoom.set_title("5–10M tokens", fontsize=7, pad=2)
    style(zoom)

    fig_bands, right = plt.subplots(figsize=(4.9, 3.4))
    for i, (condition, _, color) in enumerate(endpoints):
        final = baseline(condition)[1][:, -1]
        marker = {"scratch": "^", "shuffled_nca": "s"}.get(condition, "o")
        face = color if condition in ("scratch", "nca_40_50") else BASE
        right.errorbar(final.mean(), i, xerr=final.std(ddof=1), fmt=marker,
                       markersize=4.8, markerfacecolor=face,
                       markeredgecolor=color, markeredgewidth=1.0,
                       capsize=2.3, capthick=0.7, elinewidth=0.7,
                       color=color, ecolor=color, zorder=3)
    right.set_yticks(range(len(endpoints)), [row[1] for row in endpoints])
    right.invert_yaxis()
    right.set(xlabel="Validation loss at 10M tokens", xlim=(4.14, 4.39))
    right.set_xticks([4.15, 4.20, 4.25, 4.30, 4.35])
    style(right, axis="x")
    fig_curves.subplots_adjust(left=0.13, right=0.985, bottom=0.16, top=0.94)
    fig_bands.subplots_adjust(left=0.23, right=0.985, bottom=0.16, top=0.94)
    save(fig_curves, "fig1_reproduction_curves")
    save(fig_bands, "fig1_reproduction_bands")


def recurrence():
    rows = list(csv.DictReader(
        (PROJECT / "progress/nca_search/experiments.csv").open()))
    labels = [
        "Lower cell change", "Higher cell change",
        "Higher spatial agreement", "Higher horizontal recurrence",
        "Lower horizontal recurrence", "Higher transition MI",
    ]
    effects, counts = [], []
    for experiment in range(1, 7):
        selected = [r for r in rows if r["experiment"] == f"{experiment:02d}"]
        effects.append(np.mean([float(r["reference_loss"]) -
                                float(r["candidate_loss"]) for r in selected]))
        counts.append(len(selected))
    assert counts == [1, 5, 1, 5, 5, 1]
    records = list(csv.DictReader(
        (PROJECT / "progress/nca_search/replication_1m.csv").open()))
    fig_selectors, left = plt.subplots(figsize=(6.0, 3.55))
    colors = [HIGH if i == 3 else LOW if i == 4 else EXPLORATORY
              for i in range(len(labels))]
    left.barh(range(len(labels)), effects, height=0.52, color=colors)
    left.axvline(0, color=INK, lw=0.9)
    left.set_yticks(range(len(labels)), labels, fontsize=7.7)
    left.invert_yaxis()
    left.set(xlabel="1M loss gain vs ordinary NCA", xlim=(-0.071, 0.083))
    left.set_xticks([-0.06, -0.03, 0, 0.03, 0.06])
    style(left, axis="x")
    fig_replicates, right = plt.subplots(figsize=(4.9, 3.55))
    for block, linestyle in [("a", "-"), ("b", "--"), ("c", ":")]:
        values = [np.mean([float(r["validation_loss"]) for r in records
                           if r["block"] == block and r["arm"] == arm])
                  for arm in ("high", "unfiltered", "low")]
        assert values[0] < values[1] < values[2]
        right.plot(range(3), values, color=GRAY, lw=0.8,
                   linestyle=linestyle, zorder=2,
                   label=f"Rule population {block.upper()}")
        for x, (value, color, marker, face) in enumerate(zip(
                values, (HIGH, ORDINARY, LOW), ("D", "o", "s"),
                (HIGH, BASE, LOW))):
            right.plot(x, value, linestyle="none", marker=marker,
                       markersize=4.8, markerfacecolor=face,
                       markeredgecolor=color, markeredgewidth=1.0, zorder=3)
    right.set_xticks(range(3), ["High", "Ordinary", "Low"])
    right.set(xlabel="Horizontal recurrence selection",
              ylabel="Validation loss at 1M tokens", xlim=(-0.1, 2.1))
    right.legend(frameon=False, fontsize=7, loc="upper left")
    style(right)
    fig_selectors.subplots_adjust(left=0.39, right=0.985, bottom=0.19, top=0.94)
    fig_replicates.subplots_adjust(left=0.16, right=0.985, bottom=0.19, top=0.94)
    save(fig_selectors, "fig2_recurrence_selectors")
    save(fig_replicates, "fig2_recurrence_replicates")


if __name__ == "__main__":
    reproduction()
    recurrence()
    print(f"Saved Figures 1 and 2 as four PDF/SVG panels in {OUT}")
