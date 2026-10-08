"""Rebuild the paper-regime post-hoc analysis from completed v2 stages.

Read-only with respect to the experiment. Run from the repository root with:
    uv run --with numpy --with scipy --with matplotlib python progress/part1/analyze.py
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/1m-nca-transfer-v2"
HERE = Path(__file__).resolve().parent
OUT = HERE / "paper_regime"
OUT.mkdir(exist_ok=True)
SEEDS = tuple(range(5))
BANDS = ("nca_20_30", "nca_30_40", "nca_40_50", "nca_50_plus")
CONDITIONS = ("scratch", "shuffled_nca", *BANDS)
LABELS = {"scratch": "Scratch", "shuffled_nca": "Shuffled NCA", "nca_20_30": "20–30",
          "nca_30_40": "30–40", "nca_40_50": "40–50", "nca_50_plus": "50+"}
COLORS = {"scratch": "#87919f", "nca_20_30": "#88a5a6", "nca_30_40": "#3f9ca2",
          "nca_40_50": "#087f88", "nca_50_plus": "#d49a52"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_sha(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def describe(values: list[float] | np.ndarray) -> dict:
    a = np.asarray(values, dtype=float)
    n = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else None
    half = float(stats.t.ppf(.975, n-1) * sd / math.sqrt(n)) if n > 1 else None
    return {"n": n, "mean": float(a.mean()), "sample_sd": sd,
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "t95_ci": [float(a.mean()-half), float(a.mean()+half)] if half is not None else None}


def paired_stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    boot = np.asarray([np.mean(choice) for choice in itertools.product(a, repeat=len(a))])
    wins, losses, ties = int((a > 0).sum()), int((a < 0).sum()), int((a == 0).sum())
    decisive = wins + losses
    sign_p = min(1., 2 * stats.binom.cdf(min(wins, losses), decisive, .5)) if decisive else None
    return {**describe(a), "bootstrap95_ci": np.quantile(boot, [.025, .975]).tolist(),
            "wins": wins, "losses": losses, "ties": ties, "sign_test_two_sided_p": sign_p}


def crossing(tokens: np.ndarray, losses: np.ndarray, threshold: float) -> float | None:
    # First observed crossing, linear between adjacent checkpoints. Never extrapolate.
    if losses[0] <= threshold:
        return float(tokens[0])
    for i in range(1, len(tokens)):
        if losses[i] <= threshold <= losses[i-1]:
            portion = (losses[i-1]-threshold) / (losses[i-1]-losses[i])
            return float(tokens[i-1] + portion*(tokens[i]-tokens[i-1]))
    return None


def auc(tokens: np.ndarray, losses: np.ndarray, start: int) -> float:
    assert tokens[0] <= start < tokens[-1]
    xs = np.r_[start, tokens[tokens > start]]
    ys = np.r_[np.interp(start, tokens, losses), losses[tokens > start]]
    return float(np.trapezoid(ys, xs / 1e6))  # loss × million downstream tokens


def csv_out(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def f(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


config_path = RUN / "config/experiment.json"
config = read_json(config_path)
summary_marker = read_json(RUN / "summary/done.json")
summary_sha_ok = file_sha(RUN / "summary/summary.json") == summary_marker["summary_sha256"]
global_config_sha_ok = config_sha(config) == summary_marker["config_sha256"]
assert config["seeds"] == list(SEEDS) and config["conditions"] == list(CONDITIONS)
assert summary_sha_ok and global_config_sha_ok

records: dict[tuple[int, str, str], dict] = {}
completion = {"expected_nca": len(SEEDS) * (len(BANDS)+1),
              "expected_downstream": len(SEEDS) * len(CONDITIONS),
              "completed_nca": 0, "completed_downstream": 0,
              "missing_or_invalid": [], "completed_by_condition": {},
              "summary_done": True, "summary_hash_valid": summary_sha_ok,
              "global_config_hash_valid": global_config_sha_ok}
milestones = {}

for seed in SEEDS:
    seed_dir = RUN / "seeds" / f"seed_{seed}"
    seed_config_path = seed_dir / "config/experiment.json"
    seed_config_sha = config_sha(read_json(seed_config_path))
    base_done = read_json(seed_dir / "config/base_random_init/done.json")
    assert base_done["config_sha256"] == seed_config_sha
    for stage, conditions in (("nca", ("shuffled_nca", *BANDS)),
                              ("downstream", CONDITIONS)):
        for condition in conditions:
            directory = seed_dir / stage / condition
            done_path, metric_path = directory / "done.json", directory / "metrics.json"
            key = f"{stage}/{condition}"
            valid = False
            try:
                done, metrics = read_json(done_path), read_json(metric_path)
                history = metrics["history"]
                final = history[-1]
                budget = (config["language_tokens"] if stage == "downstream"
                          else config["nca_usable_training_tokens_per_band"])
                valid = (done["config_sha256"] == seed_config_sha
                         and done["tokens_consumed"] == budget
                         and final["tokens"] == budget
                         and final["validation_loss"] == done["final_validation_loss"]
                         and final["perplexity"] == done["final_perplexity"]
                         and (directory / done["final_checkpoint"]).is_file()
                         and all(0 <= a["tokens"] < b["tokens"]
                                 for a, b in zip(history, history[1:]))
                         and all(math.isfinite(row["validation_loss"])
                                 and math.isclose(math.exp(row["validation_loss"]), row["perplexity"], rel_tol=1e-10)
                                 for row in history))
                if stage == "downstream":
                    valid &= len(history) == len(config["language_milestones"])
                else:
                    valid &= len(history) == len(config["nca_milestones"])
            except (OSError, KeyError, ValueError, TypeError, IndexError) as error:
                completion["missing_or_invalid"].append(f"seed {seed} {key}: {error}")
            if not valid:
                if not any(f"seed {seed} {key}:" in item for item in completion["missing_or_invalid"]):
                    completion["missing_or_invalid"].append(f"seed {seed} {key}: failed marker/metric/checkpoint consistency")
                continue
            records[seed, stage, condition] = {"done": done, "history": history,
                                                "metrics_path": str(metric_path.relative_to(ROOT)),
                                                "done_path": str(done_path.relative_to(ROOT))}
            completion[f"completed_{stage}"] += 1
            completion["completed_by_condition"].setdefault(key, []).append(seed)
            milestones.setdefault(stage, {})[f"seed_{seed}/{condition}"] = [r["tokens"] for r in history]

assert not completion["missing_or_invalid"], completion["missing_or_invalid"]
assert completion["completed_nca"] == completion["expected_nca"]
assert completion["completed_downstream"] == completion["expected_downstream"]

final_rows = []
per_seed = {}
curves = {}
for seed in SEEDS:
    scratch_final = records[seed, "downstream", "scratch"]["history"][-1]
    per_seed[str(seed)] = {}
    for condition in CONDITIONS:
        stage = records[seed, "downstream", condition]
        history = stage["history"]
        tok = np.asarray([r["tokens"] for r in history], dtype=float)
        loss = np.asarray([r["validation_loss"] for r in history], dtype=float)
        curves[seed, condition] = {"tokens": tok, "loss": loss}
        final = history[-1]
        row = {"seed": seed, "condition": condition, "final_train_loss": final["train_loss"],
               "final_validation_loss": final["validation_loss"],
               "final_validation_perplexity": final["perplexity"],
               "loss_gain_vs_scratch": scratch_final["validation_loss"]-final["validation_loss"],
               "perplexity_gain_vs_scratch": scratch_final["perplexity"]-final["perplexity"],
               "source_metrics": stage["metrics_path"], "source_done": stage["done_path"]}
        per_seed[str(seed)][condition] = row
        final_rows.append(row)
csv_out("final_by_seed.csv", final_rows)

aggregates = {}
paired = {}
for condition in CONDITIONS:
    items = [per_seed[str(seed)][condition] for seed in SEEDS]
    aggregates[condition] = {key: describe([row[key] for row in items])
                             for key in ("final_train_loss", "final_validation_loss",
                                         "final_validation_perplexity")}
    if condition != "scratch":
        paired[condition] = {"loss": paired_stats([row["loss_gain_vs_scratch"] for row in items]),
                             "perplexity": paired_stats([row["perplexity_gain_vs_scratch"] for row in items]),
                             "by_seed": {str(row["seed"]): {"loss": row["loss_gain_vs_scratch"],
                                                             "perplexity": row["perplexity_gain_vs_scratch"]}
                                         for row in items}}

learning = {}
for condition in CONDITIONS:
    arr = np.asarray([curves[seed, condition]["loss"] for seed in SEEDS])
    token_arr = np.asarray([curves[seed, condition]["tokens"] for seed in SEEDS])
    learning[condition] = [{"planned_tokens": planned, "actual_tokens_min": int(token_arr[:, i].min()),
                            "actual_tokens_max": int(token_arr[:, i].max()),
                            "mean_loss": float(arr[:, i].mean()),
                            "sample_sd_loss": float(arr[:, i].std(ddof=1)),
                            "seed_losses": arr[:, i].tolist()}
                           for i, planned in enumerate(config["language_milestones"])]

threshold_rows = []
efficiency = {}
for condition in CONDITIONS:
    rows = []
    for seed in SEEDS:
        target = per_seed[str(seed)]["scratch"]["final_validation_loss"]
        curve = curves[seed, condition]
        token = crossing(curve["tokens"], curve["loss"], target)
        row = {"seed": seed, "condition": condition, "target_scratch_loss": target,
               "tokens_to_threshold": token,
               "token_saving_fraction": None if token is None else 1-token/10_000_000,
               "speedup_x": None if token is None or token == 0 else 10_000_000/token}
        rows.append(row)
        threshold_rows.append(row)
    valid = [r for r in rows if r["tokens_to_threshold"] is not None]
    mean_tokens = np.asarray([row["actual_tokens_min"] + (row["actual_tokens_max"] - row["actual_tokens_min"])/2
                              for row in learning[condition]], dtype=float)
    mean_loss = np.asarray([row["mean_loss"] for row in learning[condition]])
    mean_target = aggregates["scratch"]["final_validation_loss"]["mean"]
    visual_cross = crossing(mean_tokens, mean_loss, mean_target)
    efficiency[condition] = {"by_seed": {str(r["seed"]): r for r in rows},
                             "reached_count": len(valid),
                             "tokens": describe([r["tokens_to_threshold"] for r in valid]) if valid else None,
                             "saving_fraction": describe([r["token_saving_fraction"] for r in valid]) if valid else None,
                             "speedup_x": describe([r["speedup_x"] for r in valid]) if valid else None,
                             "mean_curve_visualization_only": {
                                 "tokens_to_threshold": visual_cross,
                                 "token_saving_fraction": None if visual_cross is None else 1-visual_cross/10_000_000,
                                 "speedup_x": None if visual_cross is None else 10_000_000/visual_cross}}
csv_out("threshold_by_seed.csv", threshold_rows)

auc_rows = []
aucs = {}
for condition in CONDITIONS:
    by_seed = {}
    for seed in SEEDS:
        curve = curves[seed, condition]
        by_seed[str(seed)] = {str(start): auc(curve["tokens"], curve["loss"], start)
                              for start in (0, 500_000, 1_000_000)}
        auc_rows.append({"seed": seed, "condition": condition, **{f"auc_from_{start}": value
                          for start, value in by_seed[str(seed)].items()}})
    aucs[condition] = {"by_seed": by_seed,
                       "aggregate": {str(start): describe([by_seed[str(seed)][str(start)] for seed in SEEDS])
                                     for start in (0, 500_000, 1_000_000)}}
csv_out("auc_by_seed.csv", auc_rows)
for condition in BANDS:
    aucs[condition]["paired_improvement_vs_scratch"] = {
        str(start): paired_stats([aucs["scratch"]["by_seed"][str(seed)][str(start)]
                                  - aucs[condition]["by_seed"][str(seed)][str(start)]
                                  for seed in SEEDS]) for start in (0, 500_000, 1_000_000)}

ranks = {}
for metric in ("final_loss", "auc_500000", "tokens_to_threshold"):
    values = {}
    for seed in SEEDS:
        value = {condition: (per_seed[str(seed)][condition]["final_validation_loss"] if metric == "final_loss"
                             else aucs[condition]["by_seed"][str(seed)]["500000"] if metric == "auc_500000"
                             else efficiency[condition]["by_seed"][str(seed)]["tokens_to_threshold"])
                 for condition in BANDS}
        values[str(seed)] = {name: int(rank) for name, rank in zip(BANDS,
                             stats.rankdata([value[name] for name in BANDS], method="average"))}
    ranks[metric] = {"by_seed": values,
                     "mean_rank": {name: float(np.mean([values[str(seed)][name] for seed in SEEDS]))
                                   for name in BANDS}}

pairwise = {}
for left, right in itertools.combinations(BANDS, 2):
    differences = [per_seed[str(seed)][right]["final_validation_loss"]
                   - per_seed[str(seed)][left]["final_validation_loss"] for seed in SEEDS]
    pairwise[f"{left}_vs_{right}"] = {"positive_favors": left,
                                     "by_seed": dict(zip(map(str, SEEDS), differences)),
                                     "stats": paired_stats(differences)}

nca_loss_rows = []
for seed in SEEDS:
    for condition in BANDS:
        nca_loss_rows.append({"seed": seed, "condition": condition,
                              "nca_validation_loss": records[seed, "nca", condition]["history"][-1]["validation_loss"],
                              "downstream_gain": per_seed[str(seed)][condition]["loss_gain_vs_scratch"],
                              "downstream_final_loss": per_seed[str(seed)][condition]["final_validation_loss"],
                              "auc_500000": aucs[condition]["by_seed"][str(seed)]["500000"],
                              "tokens_to_threshold": efficiency[condition]["by_seed"][str(seed)]["tokens_to_threshold"]})
csv_out("nca_loss_transfer.csv", nca_loss_rows)
correlations = {}
for target in ("downstream_gain", "downstream_final_loss", "auc_500000", "tokens_to_threshold"):
    x_all = np.asarray([r["nca_validation_loss"] for r in nca_loss_rows])
    y_all = np.asarray([r[target] for r in nca_loss_rows])
    x_center = np.asarray([r["nca_validation_loss"] - np.mean([q["nca_validation_loss"]
                            for q in nca_loss_rows if q["condition"] == r["condition"]]) for r in nca_loss_rows])
    y_center = np.asarray([r[target] - np.mean([q[target] for q in nca_loss_rows
                            if q["condition"] == r["condition"]]) for r in nca_loss_rows])
    correlations[target] = {"pooled_pearson_r": float(stats.pearsonr(x_all, y_all).statistic),
                            "pooled_spearman_rho": float(stats.spearmanr(x_all, y_all).statistic),
                            "within_band_centered_pearson_r": float(stats.pearsonr(x_center, y_center).statistic),
                            "within_band_centered_spearman_rho": float(stats.spearmanr(x_center, y_center).statistic)}

baseline_ppl = aggregates["scratch"]["final_validation_perplexity"]["mean"]
ppl_reductions = {band: {"absolute": baseline_ppl-aggregates[band]["final_validation_perplexity"]["mean"],
                         "fraction": (baseline_ppl-aggregates[band]["final_validation_perplexity"]["mean"])/baseline_ppl}
                  for band in BANDS}
summary = {"source_run": str(RUN.relative_to(ROOT)), "source_config_sha256": config_sha(config),
           "method": {"paired_unit": "initialization seed", "n": 5,
                      "ci": "t95 for unpaired summaries; exact 5^5 percentile bootstrap for paired means",
                      "threshold": "each seed's own scratch@10M validation loss; first linear crossing; no extrapolation",
                      "auc": "trapezoidal integral over actual tokens, divided by 1e6; interpolated common start",
                      "perplexity": "mean of logged per-seed perplexities, not exp(mean loss)"},
           "completion": completion, "milestones": milestones, "per_seed_final": per_seed,
           "aggregate_final": aggregates, "paired_vs_scratch": paired,
           "learning_curves": learning, "sample_efficiency": efficiency,
           "aucs": aucs, "band_rankings": ranks, "pairwise_band_final_loss": pairwise,
           "nca_loss_transfer_rows": nca_loss_rows, "nca_loss_correlations": correlations,
           "perplexity_reductions": ppl_reductions}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


# Figures use the same white / slate / teal visual grammar as the existing two-curve plot.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220, "svg.hashsalt": "sampleeff-paper-regime-v1"})
INK, GRID = "#233042", "#e8edf1"


def style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor("white")
    ax.set_xlabel(xlabel, color=INK, labelpad=8)
    ax.set_ylabel(ylabel, color=INK, labelpad=8)
    ax.tick_params(colors=INK, length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=.9)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd4dc")


def save_fig(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", facecolor="white", bbox_inches="tight",
                metadata={"Date": None})
    plt.close(fig)


def plot_curve(ax, condition: str, *, ribbon=True, label=None, width=2.5) -> None:
    rows = learning[condition]
    x = np.asarray([(row["actual_tokens_min"]+row["actual_tokens_max"])/2e6 for row in rows])
    mean = np.asarray([row["mean_loss"] for row in rows])
    sd = np.asarray([row["sample_sd_loss"] for row in rows])
    color = COLORS[condition]
    if ribbon:
        ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.20, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=width, marker="o" if condition != "scratch" else "s",
            markersize=4.4, markeredgecolor="white", markeredgewidth=.5,
            label=label or LABELS[condition])


fig, ax = plt.subplots(figsize=(10.8, 6.2), facecolor="white")
fig.subplots_adjust(left=.09, right=.97, top=.94, bottom=.14)
for condition in ("scratch", "nca_40_50"):
    plot_curve(ax, condition, label="NCA 40–50" if condition == "nca_40_50" else None, width=2.8)
style(ax, "TinyStories tokens (millions)", "Validation loss")
ax.set_xlim(-.2, 10.2)
ax.set_ylim(3.85, 11.65)
ax.set_xticks([0, 1, 2, 3, 5, 7.5, 10])
ax.legend(loc="upper left", bbox_to_anchor=(.20, .99), frameon=False, labelcolor=INK)
inset = ax.inset_axes([.58, .52, .39, .43])
for condition in ("scratch", "nca_40_50"):
    plot_curve(inset, condition, width=2)
style(inset, "", "")
inset.set_xlim(4.85, 10.2)
inset.set_ylim(4.02, 4.67)
inset.set_xticks([5, 7.5, 10])
inset.set_yticks([4.2, 4.4, 4.6])
inset.tick_params(labelsize=8)
mean_cross = efficiency["nca_40_50"]["mean_curve_visualization_only"]["tokens_to_threshold"]/1e6
target = aggregates["scratch"]["final_validation_loss"]["mean"]
inset.hlines(target, mean_cross, 10, color="#a5afbb", linestyle=(0, (3, 3)), lw=1)
inset.vlines([mean_cross, 10], 4.075, target, color="#a5afbb", linestyle=(0, (2, 3)), lw=.8)
inset.annotate("", xy=(10, 4.075), xytext=(mean_cross, 4.075),
               arrowprops={"arrowstyle": "<->", "color": COLORS["nca_40_50"], "lw": 1.3})
inset.text((mean_cross+10)/2, 4.105,
           f"~{10/mean_cross:.2f}x (mean curves)", ha="center", va="bottom",
           color=COLORS["nca_40_50"], fontsize=8.4)
save_fig(fig, "01_scratch_vs_nca_40_50")

fig, ax = plt.subplots(figsize=(10.8, 6.2), facecolor="white")
fig.subplots_adjust(left=.09, right=.97, top=.94, bottom=.14)
for condition in ("scratch", *BANDS):
    plot_curve(ax, condition, width=2.8 if condition == "scratch" else 2.2)
style(ax, "TinyStories tokens (millions)", "Validation loss")
ax.set_xlim(-.2, 10.2)
ax.set_ylim(3.85, 11.65)
ax.set_xticks([0, 1, 2, 3, 5, 7.5, 10])
ax.legend(loc="upper left", bbox_to_anchor=(.18, .99), frameon=False,
          ncol=2, labelcolor=INK, fontsize=9)
save_fig(fig, "02_all_structured_bands")


def points_plot(name: str, ylabel: str, by_seed_value, *, hline=None, ylimit=None) -> None:
    fig, ax = plt.subplots(figsize=(9.3, 5.6), facecolor="white")
    fig.subplots_adjust(left=.11, right=.97, top=.95, bottom=.15)
    style(ax, "Gzip band", ylabel)
    offsets = np.linspace(-.17, .17, len(SEEDS))
    for i, condition in enumerate(BANDS):
        ys = np.asarray([by_seed_value(seed, condition) for seed in SEEDS])
        ax.scatter(i+offsets, ys, s=35, color=COLORS[condition], alpha=.65,
                   edgecolor="white", linewidth=.6, zorder=3)
        ax.errorbar(i, ys.mean(), yerr=ys.std(ddof=1), fmt="D", markersize=7,
                    capsize=5, color=COLORS[condition], ecolor=INK, elinewidth=1.1,
                    markeredgecolor="white", markeredgewidth=.7, zorder=5)
    if hline is not None:
        ax.axhline(hline, color="#b7c0c9", lw=1, linestyle=(0, (4, 3)))
    ax.set_xticks(range(4), [LABELS[band] for band in BANDS])
    ax.set_xlim(-.5, 3.5)
    if ylimit:
        ax.set_ylim(*ylimit)
    save_fig(fig, name)


points_plot("03_final_loss_gain", "Paired final validation-loss gain vs scratch",
            lambda s, c: per_seed[str(s)][c]["loss_gain_vs_scratch"], hline=0)
points_plot("04_convergence_speedup", "Paired token speedup to scratch@10M loss (x)",
            lambda s, c: efficiency[c]["by_seed"][str(s)]["speedup_x"], hline=1, ylimit=(.95, 2.12))
points_plot("05_auc_gain", "Paired AUC improvement vs scratch (loss × M tokens)",
            lambda s, c: aucs["scratch"]["by_seed"][str(s)]["500000"]
                         - aucs[c]["by_seed"][str(s)]["500000"],
            hline=0)


# Write prose only after all metrics, checks, CSVs, and figures have been produced.
doc = []


def add(*lines: str) -> None:
    doc.extend(lines)


def table(headers: list[str], rows: list[list[str]]) -> None:
    add(md_table(headers, rows), "")


def mean_sd(stat: dict, digits=3) -> str:
    return f"{f(stat['mean'], digits)} ± {f(stat['sample_sd'], digits)}"


best = "nca_40_50"
best_eff = efficiency[best]
best_paired = paired[best]["loss"]
best_ppl = ppl_reductions[best]
best_ppl_pct = 100*best_ppl["fraction"]
all_wins = sum(paired[band]["loss"]["wins"] for band in BANDS)
mean_band_range = (max(aggregates[band]["final_validation_loss"]["mean"] for band in BANDS)
                   - min(aggregates[band]["final_validation_loss"]["mean"] for band in BANDS))

add("# Paper-regime post-hoc analysis: NCA pre-pre-training at ~1M parameters", "",
    "## Executive summary", "",
    f"- **Core effect:** yes. All four structured NCA bands beat paired scratch at 10M TinyStories tokens in all {all_wins}/20 band–seed comparisons; these are five independent seeds, not 20 independent replications.",
    f"- **Sample efficiency:** the 40–50 band reaches each seed's scratch@10M loss using {100*best_eff['saving_fraction']['mean']:.1f}% fewer downstream tokens on average (five-seed paired mean).",
    f"- **Convergence:** its mean paired token ratio is {best_eff['speedup_x']['mean']:.2f}x; this is a downstream-token ratio, not wall-clock speed or total pretraining cost.",
    f"- **Final quality:** the 40–50 band's mean validation loss improves by {best_paired['mean']:.3f}; mean perplexity falls from {aggregates['scratch']['final_validation_perplexity']['mean']:.2f} to {aggregates[best]['final_validation_perplexity']['mean']:.2f} ({best_ppl_pct:.1f}% lower).",
    "- **Five-seed consistency:** each structured band wins 5/5 paired final-loss comparisons; the two-sided exact sign-test p is 0.0625 per band.",
    f"- **Gzip resolution:** 40–50 leads on five-seed mean final loss, but its mean edge over 30–40 is only {aggregates['nca_30_40']['final_validation_loss']['mean']-aggregates[best]['final_validation_loss']['mean']:.3f}, and seed rankings change. The coarse bands do not establish a sharp universal optimum.",
    "",
    "## Scope, sources, and formulas", "",
    "This is a post-hoc analysis of completed `runs/1m-nca-transfer-v2/` artifacts. It covers structured NCA versus scratch and coarse gzip-band differences. Shuffled-NCA is retained only in raw final tables; no mechanistic inference uses it. The experimental model has 1,081,177 trainable text parameters; each structured NCA stage used 19,715,325 usable synthetic training labels before 10M downstream TinyStories tokens. Seeds 0–4 share initialization and downstream setup within each seed; they are the independent analysis units. The source [configuration](../../runs/1m-nca-transfer-v2/config/experiment.json), [completion marker](../../runs/1m-nca-transfer-v2/summary/done.json), and per-stage `done.json` plus `metrics.json` are checked by [`analyze.py`](analyze.py).",
    "",
    "Formulas: loss gain = paired scratch loss − NCA loss; perplexity reduction = paired scratch perplexity − NCA perplexity. Final perplexity summaries average logged per-seed values, rather than exponentiating mean loss. For each seed, target = its own scratch validation loss at 10M; first NCA threshold crossing is linearly interpolated between adjacent logged checkpoints, without extrapolation. Token saving = 1 − NCA tokens/10M; speedup = 10M/NCA tokens. AUC is trapezoidal loss integration over each run's actual consumed-token positions, divided by 1M, with linear interpolation at the common interval start. Reported SD is sample SD. Descriptive 95% t intervals use df=4; paired bootstrap intervals enumerate all 5^5 seed-resampling combinations. These intervals are exploratory with n=5 and do not correct for multiple comparisons.",
    "",
    "## A. Completion and sanity", "",
    f"The [summary marker](../../runs/1m-nca-transfer-v2/summary/done.json) matches the summary artifact hash and the canonical JSON configuration hash. Every stage below has a valid marker, metrics history, final checkpoint, budget, and final loss/perplexity matching the marker. No missing or corrupt metrics were found. Intended configuration and completed evidence agree on five seeds and six downstream conditions.",
    "")
table(["Stage", "Expected", "Completed", "Seeds / conditions"], [
    ["Structured + shuffled NCA", "25", str(completion["completed_nca"]), "0–4 × 5"],
    ["Downstream", "30", str(completion["completed_downstream"]), "0–4 × 6"],
    ["Structured NCA only", "20", "20", "0–4 × 4"],
])
add("The intended language milestones and actual consumed-token positions are shown below. Actual positions are identical across all 30 downstream stages but differ slightly from the rounded intended milestones; analysis uses the logged positions. All final language runs consumed exactly 10,000,000 tokens. The planned NCA checkpoints are 0, 2M, 4M, 6M, 8M, 10M, 12M, 14M, 16M, 18M, and 20M; their final actual count is 19,715,325 usable labels.", "")
milestone_rows = []
for i, planned in enumerate(config["language_milestones"]):
    actual = [v[i] for v in milestones["downstream"].values()]
    milestone_rows.append([f"{planned:,}", f"{min(actual):,}", f"{max(actual):,}"])
table(["Planned tokens", "Actual min", "Actual max"], milestone_rows)
add("**Metadata inconsistency:** global `config/experiment.json` retains `one_seed_exploratory: true` from the earlier setup, while the completed five-seed summary records `false`. The five seed directories and their completed markers establish the actual run. No training artifact was altered.", "")

add("## B. Raw final results", "",
    "At 10M downstream tokens; positive gains favor the condition over paired scratch. The shuffled rows are inventory only.", "")
table(["Seed", "Condition", "Train loss", "Val loss", "Val PPL", "Loss gain", "PPL gain"], [
    [str(row["seed"]), LABELS[row["condition"]], f(row["final_train_loss"], 4),
     f(row["final_validation_loss"], 4), f(row["final_validation_perplexity"], 2),
     f(row["loss_gain_vs_scratch"], 4), f(row["perplexity_gain_vs_scratch"], 2)]
    for row in final_rows])
add("Across-seed summaries below include sample SD, median, range, and a descriptive t-based 95% CI. Full-precision values are in [`summary.json`](paper_regime/summary.json).", "")
aggregate_rows = []
for condition in CONDITIONS:
    a = aggregates[condition]
    aggregate_rows.append([LABELS[condition], mean_sd(a["final_train_loss"]),
                           mean_sd(a["final_validation_loss"]),
                           f(a["final_validation_loss"]["median"]),
                           f(a["final_validation_loss"]["min"])+"–"+f(a["final_validation_loss"]["max"]),
                           "["+", ".join(f(v) for v in a["final_validation_loss"]["t95_ci"])+"]",
                           mean_sd(a["final_validation_perplexity"], 2),
                           f(a["final_validation_perplexity"]["median"], 2),
                           f(a["final_validation_perplexity"]["min"], 2)+"–"+f(a["final_validation_perplexity"]["max"], 2),
                           "["+", ".join(f(v, 2) for v in a["final_validation_perplexity"]["t95_ci"])+"]"])
table(["Condition", "Train mean±SD", "Val mean±SD", "Val median", "Val min–max", "Val 95% CI",
       "PPL mean±SD", "PPL median", "PPL min–max", "PPL 95% CI"], aggregate_rows)
table(["Condition", "Final train median", "Final train min–max", "Final train 95% CI"], [
    [LABELS[condition], f(aggregates[condition]["final_train_loss"]["median"]),
     f(aggregates[condition]["final_train_loss"]["min"])+"–"+f(aggregates[condition]["final_train_loss"]["max"]),
     "["+", ".join(f(v) for v in aggregates[condition]["final_train_loss"]["t95_ci"])+"]"]
    for condition in CONDITIONS])

add("## C. Paired final effects versus scratch", "",
    "All effects use each seed's own scratch run. Exact two-sided sign-test p=0.0625 for each 5/5 structured-band direction; it is not evidence for a conventional 0.05 claim. The bootstrap interval is the exact percentile distribution over 5^5 resamples of five paired seeds.", "")
paired_rows = []
for condition in BANDS:
    p = paired[condition]
    l, ppl = p["loss"], p["perplexity"]
    paired_rows.append([LABELS[condition], ", ".join(f(p["by_seed"][str(s)]["loss"], 3) for s in SEEDS),
                        mean_sd(l), f(l["median"]),
                        "["+", ".join(f(v) for v in l["bootstrap95_ci"])+"]",
                        f"{l['wins']}/{l['losses']}/{l['ties']}", mean_sd(ppl, 2),
                        "["+", ".join(f(v, 2) for v in ppl["bootstrap95_ci"])+"]"])
table(["Band", "Seed loss gains 0→4", "Loss mean±SD", "Median", "Bootstrap 95%",
       "W/L/T", "PPL gain mean±SD", "PPL bootstrap 95%"], paired_rows)

add("## D. Full downstream learning curves", "",
    "The table gives mean validation loss ± sample SD across seeds at each planned checkpoint; the preceding milestone table gives the actual token positions. The 40–50 band is the representative figure because it matches the established replication view and has the lowest five-seed mean final loss among these four bands. This does not establish that it is universally best.", "",
    "![Scratch versus NCA 40–50 with a late-stage inset](paper_regime/01_scratch_vs_nca_40_50.png)", "",
    "![All four structured NCA gzip bands and scratch](paper_regime/02_all_structured_bands.png)", "")
curve_rows = []
for i, planned in enumerate(config["language_milestones"]):
    curve_rows.append([f"{planned/1e6:g}M"] + [mean_sd({"mean": learning[c][i]["mean_loss"],
                                      "sample_sd": learning[c][i]["sample_sd_loss"]})
                                                for c in ("scratch", *BANDS)])
table(["Planned", "Scratch", "20–30", "30–40", "40–50", "50+"], curve_rows)
add("All structured bands begin with worse loss at zero language tokens after the text embedding/head is newly initialized. Every band overtakes scratch by the first logged nonzero checkpoint (~0.252M tokens) and remains ahead in mean loss at all later checkpoints. At that first checkpoint, 20–30, 30–40, and 40–50 win 5/5 paired seeds; 50+ wins 4/5. From ~0.502M onward, all four bands win 5/5 at every logged checkpoint. The early loss gap is larger than the final gap, so the observed advantage shrinks in absolute loss units but persists through 10M.", "")

add("## E. Sample efficiency and convergence", "",
    "The primary threshold is **each seed's own scratch@10M validation loss**. Tokens to threshold come from the first linear crossing of the logged NCA curve. Every structured run reaches its paired target within observed checkpoints; no extrapolation is used. The inset in Figure 1 uses the mean curves only to illustrate the crossing; its ~1.74x label is distinct from the mean of five paired seed-level speedups. The existing run summary's 40–50 threshold of 5.747M tokens is likewise a mean-curve crossing; this report's primary paired mean is 5.726M. These differ because interpolation and averaging do not commute.", "")
table(["Seed", "Band", "NCA tokens (M)", "Fewer tokens", "Speedup"], [
    [str(seed), LABELS[band], f(efficiency[band]["by_seed"][str(seed)]["tokens_to_threshold"]/1e6, 3),
     f(100*efficiency[band]["by_seed"][str(seed)]["token_saving_fraction"], 1)+"%",
     f(efficiency[band]["by_seed"][str(seed)]["speedup_x"], 2)+"x"]
    for seed in SEEDS for band in BANDS])
table(["Band", "Tokens M mean±SD", "Median M", "95% t CI M", "Savings mean±SD", "Speedup mean±SD",
       "Mean-curve x (visual only)"], [
    [LABELS[band], mean_sd({"mean": efficiency[band]["tokens"]["mean"]/1e6,
                            "sample_sd": efficiency[band]["tokens"]["sample_sd"]/1e6}),
     f(efficiency[band]["tokens"]["median"]/1e6),
     "["+", ".join(f(v/1e6) for v in efficiency[band]["tokens"]["t95_ci"])+"]",
     f(100*efficiency[band]["saving_fraction"]["mean"], 1)+"% ± "+f(100*efficiency[band]["saving_fraction"]["sample_sd"], 1)+"%",
     mean_sd(efficiency[band]["speedup_x"], 2)+"x",
     f(efficiency[band]["mean_curve_visualization_only"]["speedup_x"], 2)+"x"]
    for band in BANDS])
table(["Band", "Savings median", "Savings 95% t CI", "Speedup median", "Speedup 95% t CI"], [
    [LABELS[band], f(100*efficiency[band]["saving_fraction"]["median"], 1)+"%",
     "["+", ".join(f(100*v, 1)+"%" for v in efficiency[band]["saving_fraction"]["t95_ci"])+"]",
     f(efficiency[band]["speedup_x"]["median"], 2)+"x",
     "["+", ".join(f(v, 2)+"x" for v in efficiency[band]["speedup_x"]["t95_ci"])+"]"]
    for band in BANDS])
add("![Paired token speedup to each seed's scratch@10M loss](paper_regime/04_convergence_speedup.png)", "")

add("## F. Area under the learning curve", "",
    "Lower AUC means lower validation loss across the downstream trajectory. Units are loss × million tokens. Full-range AUC includes the zero-token evaluation; the 0.5M and 1M starts are interpolated to common token positions. Paired improvements subtract each NCA AUC from its seed's scratch AUC.", "")
auc_summary_rows = []
for condition in ("scratch", *BANDS):
    auc_summary_rows.append([LABELS[condition]] +
        [mean_sd(aucs[condition]["aggregate"][str(start)], 3)
         for start in (0, 500_000, 1_000_000)] +
        (["—"]*3 if condition == "scratch" else
         [mean_sd(aucs[condition]["paired_improvement_vs_scratch"][str(start)], 3)
          for start in (0, 500_000, 1_000_000)]))
table(["Condition", "AUC 0–10M", "AUC 0.5–10M", "AUC 1–10M",
       "Gain full", "Gain from 0.5M", "Gain from 1M"], auc_summary_rows)
add("![Paired AUC improvement from 0.5M to 10M](paper_regime/05_auc_gain.png)", "")

add("## G. Early, middle, and late transfer", "",
    "Entries are mean paired scratch loss − NCA loss. These are logged checkpoints near the named token amounts, not exactly rounded token positions.", "")
selected_indices = (2, 3, 4, 6, 8)
table(["Band", "~0.5M", "~1M", "~2M", "~5M", "10M"], [
    [LABELS[band]] + [f(np.mean([curves[seed, "scratch"]["loss"][i]
                                 - curves[seed, band]["loss"][i] for seed in SEEDS]), 3)
                      for i in selected_indices] for band in BANDS])
add("At ~0.5M, mean paired gains span " + f(min(np.mean([curves[s,"scratch"]["loss"][2]-curves[s,b]["loss"][2] for s in SEEDS]) for b in BANDS), 3)
    + "–" + f(max(np.mean([curves[s,"scratch"]["loss"][2]-curves[s,b]["loss"][2] for s in SEEDS]) for b in BANDS), 3)
    + " loss units; by 10M they span " + f(min(paired[b]["loss"]["mean"] for b in BANDS), 3)
    + "–" + f(max(paired[b]["loss"]["mean"] for b in BANDS), 3)
    + ". This is a strong early-learning advantage with a smaller persistent final-quality advantage. The observed curves do not establish an asymptotic advantage beyond 10M.", "")

add("## H. Gzip-band analysis", "",
    "Rank 1 is best among the four structured bands for the specified metric. Final-loss rankings change by seed; AUC and threshold-token rankings are also shown. The 40–50 band is best on the five-seed mean of all three metrics, but it has the best final loss in only two of five seeds.", "")
table(["Band", "Final loss ranks, seeds 0→4", "Mean rank", "0.5M AUC ranks", "Threshold-token ranks",
       "Final mean loss"], [
    [LABELS[band], ", ".join(str(ranks["final_loss"]["by_seed"][str(s)][band]) for s in SEEDS),
     f(ranks["final_loss"]["mean_rank"][band], 1),
     ", ".join(str(ranks["auc_500000"]["by_seed"][str(s)][band]) for s in SEEDS),
     ", ".join(str(ranks["tokens_to_threshold"]["by_seed"][str(s)][band]) for s in SEEDS),
     f(aggregates[band]["final_validation_loss"]["mean"], 4)] for band in BANDS])
add("Pairwise effects below are right-band loss − left-band loss, so positive values favor the left band. Within-seed SD measures variation across the five paired differences.", "")
table(["Left vs right", "Seed differences 0→4", "Mean±SD", "95% bootstrap", "Left wins"], [
    [LABELS[left]+" vs "+LABELS[right],
     ", ".join(f(pairwise[f"{left}_vs_{right}"]["by_seed"][str(s)], 3) for s in SEEDS),
     mean_sd(pairwise[f"{left}_vs_{right}"]["stats"]),
     "["+", ".join(f(v) for v in pairwise[f"{left}_vs_{right}"]["stats"]["bootstrap95_ci"])+"]",
     str(pairwise[f"{left}_vs_{right}"]["stats"]["wins"])+"/5"]
    for left, right in itertools.combinations(BANDS, 2)])
mean_structured_gain = np.mean([paired[b]["loss"]["mean"] for b in BANDS])
mean_structured_seed_sd = np.mean([aggregates[b]["final_validation_loss"]["sample_sd"] for b in BANDS])
add(f"The largest gap between structured-band mean final losses is {mean_band_range:.4f}, versus mean structured NCA–scratch gain {mean_structured_gain:.4f}; the band spread is {mean_band_range/mean_structured_gain:.2f} of that main effect. Mean across-band seed SD of final loss is {mean_structured_seed_sd:.4f}. The 40–50 versus 30–40 mean edge is {aggregates['nca_30_40']['final_validation_loss']['mean']-aggregates[best]['final_validation_loss']['mean']:.4f}, smaller than their paired-difference SD {pairwise['nca_30_40_vs_nca_40_50']['stats']['sample_sd']:.4f}. The data motivate more precise regime tests; they do not isolate a sharp Goldilocks optimum.", "",
    "![Final paired loss gains by gzip band](paper_regime/03_final_loss_gain.png)", "")

add("## I. NCA training loss and downstream transfer", "",
    "Final NCA validation loss is measured on held-out NCA rules within each band. The NCA targets have different difficulty distributions across bands, so raw NCA losses are not directly comparable as transfer scores.", "")
table(["Band", "NCA val loss mean±SD", "Downstream gain mean±SD"], [
    [LABELS[band], mean_sd(describe([r["nca_validation_loss"] for r in nca_loss_rows
                                    if r["condition"] == band])), mean_sd(paired[band]["loss"])]
    for band in BANDS])
table(["Downstream quantity", "Pooled Pearson", "Pooled Spearman", "Band-centered Pearson", "Band-centered Spearman"], [
    [target.replace("_", " "), f(correlations[target]["pooled_pearson_r"], 2),
     f(correlations[target]["pooled_spearman_rho"], 2),
     f(correlations[target]["within_band_centered_pearson_r"], 2),
     f(correlations[target]["within_band_centered_spearman_rho"], 2)]
    for target in correlations])
add("These 20 band–seed rows are clustered in five seeds and four bands, not 20 independent draws. The pooled association mainly reflects band difficulty and changes or nearly disappears after band centering. Lower NCA validation loss is therefore not established as a predictor of better TinyStories transfer. Correlations are descriptive; no p-values are used.", "")

add("## J. Variance and robustness", "",
    f"The main result is more stable than the fine band ranking: each structured band wins 5/5 final paired comparisons against scratch, but the top final-loss band varies across seeds (40–50 in seeds 0 and 2; 30–40 in seeds 1 and 4; 20–30 in seed 3). The four mean final losses span {mean_band_range:.4f}; across-seed SD within structured bands spans {min(aggregates[b]['final_validation_loss']['sample_sd'] for b in BANDS):.4f}–{max(aggregates[b]['final_validation_loss']['sample_sd'] for b in BANDS):.4f}. Comparisons among bands are correlated because the seeds and downstream streams are paired. The five-seed sample supports effect consistency more strongly than precise band ordering.",
    "Seed dependence is visible: seed 3 has the highest scratch final loss (4.356) and the largest 20–30 gain (0.184); seed 4 gives 40–50 its smallest gain (0.085) and ranks it third among bands. All five seeds are retained.", "")

add("## K. Perplexity interpretation", "",
    "Percent reduction uses (mean scratch PPL − mean NCA PPL) / mean scratch PPL. This is not a percentage change in loss. Perplexity is exp(validation loss) within each run; the table averages those five logged values.", "")
table(["Band", "Scratch mean PPL", "NCA mean PPL", "Absolute reduction", "Percent reduction"], [
    [LABELS[band], f(baseline_ppl, 2),
     f(aggregates[band]["final_validation_perplexity"]["mean"], 2),
     f(ppl_reductions[band]["absolute"], 2),
     f(100*ppl_reductions[band]["fraction"], 1)+"%"] for band in BANDS])

add("## L. Relation to the original paper", "",
    "The [original paper](https://arxiv.org/html/2603.10055v1) reports improved language modeling and faster convergence after NCA pre-pre-training, complexity-sensitive transfer, and additional results at much larger model scale and on other corpora and tasks. Our setup is a tiny-scale, one-corpus probe, not an exact reproduction of its model size, downstream data, or full baselines.", "")
table(["Original-paper claim", "Our tiny-scale evidence", "Status"], [
    ["NCA PPT improves language modeling vs scratch", "Four bands win 5/5 paired seeds at 10M TinyStories tokens", "Qualitatively reproduced"],
    ["NCA PPT increases downstream token efficiency", "All bands reach paired scratch@10M before 10M", "Qualitatively reproduced"],
    ["Advantage persists during downstream training", "Band mean curves lead at every nonzero checkpoint, including 10M", "Qualitatively reproduced through 10M"],
    ["Gzip complexity modulates transfer", "Mean final losses span "+f(mean_band_range, 3)+"; 50+ underperforms 40–50 in all five seeds", "Partially supported"],
    ["Optimal complexity can differ by domain", "One downstream corpus only; no cross-domain contrast", "Not tested"],
    ["Relative effect changes with model scale", "One ~1M-parameter scale only", "Not tested"],
    ["Attention layers carry transfer", "No component-transfer ablations", "Not tested"],
    ["Reasoning benchmark gains", "No downstream reasoning evaluation", "Not tested"],
    ["C4 or Dyck comparison", "Neither baseline run in this analysis", "Not tested"],
])

add("## What we can safely say", "",
    "1. At ~1M parameters, NCA pre-pre-training consistently improves downstream TinyStories learning relative to scratch across five paired seeds and four structured bands.",
    f"2. For the representative 40–50 band, five-seed paired estimates give **{100*best_eff['saving_fraction']['mean']:.1f}% fewer downstream tokens** and **{best_eff['speedup_x']['mean']:.2f}x mean convergence speedup** to each seed's own scratch@10M validation loss; these are not wall-clock or total-compute speedups.",
    f"3. At 10M, the 40–50 band's mean paired validation-loss improvement is **{best_paired['mean']:.3f}**, and mean perplexity is **{best_ppl_pct:.1f}% lower** than scratch.",
    "4. The benefit is strongest early and remains at the final logged checkpoint; behavior after 10M is unknown.",
    "5. Coarse gzip bands differ enough to motivate deeper work, but five seeds do not establish a universal optimum.", "")

add("## What we cannot say", "",
    "- Which gzip band is universally optimal, or whether a narrow Goldilocks peak exists.",
    "- Why transfer happens, whether latent-rule inference is causal, or which model component carries the effect.",
    "- That downstream wall-clock time or total compute is reduced after accounting for NCA pre-pretraining.",
    "- That the effect generalizes to larger models, other language corpora, reasoning tasks, or other synthetic/natural pre-pretraining corpora.",
    "- That this experiment reproduces the original paper's C4/Dyck comparisons, attention-transfer mechanism, cross-domain gzip matching, or reasoning gains.", "")

add("## Reproduction files", "",
    "- [`analyze.py`](analyze.py): deterministic analysis; reads completed stages and writes this report and all outputs.",
    "- [`paper_regime/summary.json`](paper_regime/summary.json): full-precision derived data, completion inventory, and source paths.",
    "- [`paper_regime/final_by_seed.csv`](paper_regime/final_by_seed.csv), [`threshold_by_seed.csv`](paper_regime/threshold_by_seed.csv), [`auc_by_seed.csv`](paper_regime/auc_by_seed.csv), and [`nca_loss_transfer.csv`](paper_regime/nca_loss_transfer.csv): compact tables.",
    "- Five PNG figures above have SVG counterparts in `paper_regime/`.",
    "",
    "```sh",
    "uv run --with numpy --with scipy --with matplotlib python progress/part1/analyze.py",
    "```", "")

report = "\n".join(doc)
(HERE / "PAPER_REGIME_POSTHOC.md").write_text(report)
mirror = RUN / "analysis/PAPER_REGIME_POSTHOC.md"
mirror.parent.mkdir(exist_ok=True)
if mirror.is_symlink():
    mirror.unlink()
if mirror.exists() and not mirror.read_text().startswith("# Paper-regime post-hoc analysis:"):
    raise ValueError(f"refusing to overwrite unrelated analysis: {mirror}")
mirror.write_text(report.replace("../../runs/1m-nca-transfer-v2/", "../")
                   .replace("(analyze.py)", "(../../../progress/part1/analyze.py)"))
