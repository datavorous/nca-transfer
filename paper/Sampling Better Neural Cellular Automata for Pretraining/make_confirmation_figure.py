"""Plot the frozen 30-run Top-10 confirmation from completed metrics only."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUN = PROJECT / "runs/nca-data-search/top10-confirmation"
SCRATCH = PROJECT / "runs/1m-nca-transfer-v2/seeds"
OUT = HERE / "figs_cool"

INK = "#233042"
GRAY = "#89919e"
GREEN = "#087f88"
GRID = "#e8edf1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def losses(block: str, arm: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    stage = RUN / f"block-{block}/{arm}/seeds/seed_{seed}/downstream/full_10m"
    receipt = json.loads((stage / "done.json").read_text())
    assert receipt["tokens_consumed"] == 10_000_000
    history = json.loads((stage / "metrics.json").read_text())["history"]
    tokens = np.array([row["tokens"] for row in history], dtype=int)
    assert tokens[0] == 0 and tokens[-1] == 10_000_000
    return tokens, np.array([row["validation_loss"] for row in history])


def crossing(tokens: np.ndarray, values: np.ndarray, target: float) -> float:
    for i in range(len(tokens) - 1):
        if values[i] >= target >= values[i + 1]:
            fraction = (values[i] - target) / (values[i] - values[i + 1])
            return (tokens[i] + fraction * (tokens[i + 1] - tokens[i])) / 1_000_000
    raise ValueError("Scratch endpoint was not reached")


def main() -> None:
    replicate_means = []
    crossing_means = []
    seed_crossings = []
    tokens = None
    for block in "abc":
        paired = []
        corpus_crossings = []
        for seed in range(5):
            ordinary_tokens, ordinary = losses(block, "ordinary", seed)
            designed_tokens, designed = losses(block, "designed", seed)
            assert np.array_equal(ordinary_tokens, designed_tokens)
            if tokens is None:
                tokens = ordinary_tokens
            else:
                assert np.array_equal(tokens, ordinary_tokens)
            paired.append(ordinary - designed)
            scratch_history = json.loads((SCRATCH / f"seed_{seed}/downstream/scratch/metrics.json").read_text())["history"]
            assert np.array_equal(tokens, [row["tokens"] for row in scratch_history])
            target = scratch_history[-1]["validation_loss"]
            corpus_crossings.append((crossing(tokens, ordinary, target),
                                     crossing(tokens, designed, target)))
        replicate_means.append(np.mean(paired, axis=0))
        seed_crossings.append(np.asarray(corpus_crossings))
        crossing_means.append(np.mean(corpus_crossings, axis=0))

    means = np.stack(replicate_means)
    x = tokens / 1_000_000
    fig_loss, left = plt.subplots(figsize=(5.5, 3.5))
    for i, (curve, linestyle) in enumerate(zip(means, ("-", "--", ":")), start=1):
        left.plot(x, curve, color=GRAY, linestyle=linestyle, linewidth=0.9,
                  label=f"Corpus {i}", zorder=2)
    left.plot(x, means.mean(axis=0), color=GREEN, linewidth=1.9,
              marker="o", markersize=3.1, label="Mean", zorder=3)
    left.axhline(0, color=GRAY, linewidth=0.7, zorder=1)
    left.set(xlabel="TinyStories tokens (M)", ylabel="Loss gain vs. Ordinary",
             xlim=(0, 10), ylim=(-0.02, 0.205))
    left.set_xticks([0, 1, 2, 3, 5, 7.5, 10])
    left.legend(frameon=False, ncol=2, loc="upper right", fontsize=7)

    fig_tokens, right = plt.subplots(figsize=(5.0, 3.5))
    for i, (seed_pairs, mean_pair) in enumerate(zip(seed_crossings, crossing_means)):
        positions = (i - 0.12, i + 0.12)
        for ordinary, designed in seed_pairs:
            right.plot(positions, (ordinary, designed), color=GRAY,
                       linewidth=0.5, alpha=0.28, zorder=1)
        right.scatter(np.full(5, positions[0]), seed_pairs[:, 0], color=GRAY,
                      s=10, alpha=0.4, zorder=2)
        right.scatter(np.full(5, positions[1]), seed_pairs[:, 1], color=GREEN,
                      s=10, alpha=0.4, zorder=2)
        right.plot(positions, mean_pair, color=GRAY, linewidth=0.9, zorder=3)
        right.plot(positions[0], mean_pair[0], marker="o", color=GRAY,
                   markersize=5.5, linestyle="none", zorder=4)
        right.plot(positions[1], mean_pair[1], marker="D", color=GREEN,
                   markersize=5.5, linestyle="none", zorder=4)
    right.plot([], [], marker="o", color=GRAY, linestyle="none", label="Ordinary")
    right.plot([], [], marker="D", color=GREEN, linestyle="none", label="Designed")
    right.set(xlabel="Corpus replicate", ylabel="Tokens to Scratch endpoint (M)",
              xlim=(-0.45, 2.45), ylim=(4.45, 7.45))
    right.set_xticks(range(3), ["1", "2", "3"])
    right.legend(frameon=False, ncol=2, loc="upper right", fontsize=7)

    for ax in (left, right):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#cbd4dc")
        ax.tick_params(length=3, width=0.7)
    fig_loss.tight_layout(pad=0.8)
    fig_tokens.tight_layout(pad=0.8)
    OUT.mkdir(exist_ok=True)
    for fig, stem in ((fig_loss, "fig3_confirmation_loss"),
                      (fig_tokens, "fig3_confirmation_tokens")):
        for ext in ("pdf", "svg"):
            fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    print(f"Saved both Figure 3 panels from all 30 completed runs to {OUT}")


if __name__ == "__main__":
    main()
