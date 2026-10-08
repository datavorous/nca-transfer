"""Read saved seed-0 dose histories and write the bounded diagnostic tables and plots."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from progress.scale_transfer_10m import summarize_replication as calc

RUN = REPO / "runs/scale-transfer-10m"
SWEEP = RUN / "dose-sweep/seed_0"
OUT = Path(__file__).with_name("DOSE_RESULTS.md")
FIGURES = OUT.parent / "figures"
ARMS = ("ordinary", "designed")
DOSES = ((10, 9_869_186), (25, 29_602_341), (50, 49_347_373),
         (100, 100_001_521), (150, 148_042_896), (200, 197_153_250))
TARGETS = (3.0, 2.5, 2.25, 2.0, 1.9)
COLORS = ("#96D4C3", "#D7B8EF", "#F0BFA6", "#A9D0F4", "#E6D58D", "#F3D38A")
BACKGROUND, PANEL, TEXT, MUTED, GRID = "#111722", "#192230", "#E8EDF4", "#A9B5C4", "#344255"
CHECKPOINTS = (250_000, 500_000, 1_000_000, 2_000_000, 3_000_000,
               5_000_000, 7_500_000, 10_000_000, 15_000_000, 20_000_000,
               30_000_000, 50_000_000, 75_000_000, 100_000_000)


def history(arm: str, requested: int) -> list[dict]:
    if requested == 200:
        path = RUN / "downstream" / arm / "metrics.json"
    else:
        path = SWEEP / "downstream" / f"{arm}_dose_{requested}m" / "metrics.json"
    return json.loads(path.read_text())["history"]


def fmt(value: float | None, digits: int = 5) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def table(lines: list[str], headers: list[str], rows: list[list[object]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join([":--"] + ["--:" for _ in headers[1:]]) + "|")
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def at_or_none(rows: list[dict], tokens: int) -> float | None:
    return calc.at(rows, tokens) if rows[-1]["tokens"] >= tokens else None


def auc_or_none(rows: list[dict], tokens: int) -> float | None:
    return calc.auc(rows, 0, tokens) if rows[-1]["tokens"] >= tokens else None


def setup_axis(axis, xlabel: str, ylabel: str) -> None:
    axis.set_facecolor(PANEL)
    axis.grid(color=GRID, linewidth=.8, alpha=.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color(GRID)
    axis.tick_params(colors=MUTED)
    axis.set_xlabel(xlabel, color=TEXT)
    axis.set_ylabel(ylabel, color=TEXT)


def legend(axis, **kwargs) -> None:
    item = axis.legend(frameon=False, **kwargs)
    for label in item.get_texts():
        label.set_color(TEXT)


def trajectory_plot(arm: str, histories: dict, scratch: list[dict]) -> list[str]:
    fig, axis = plt.subplots(figsize=(10, 5.4), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(axis, "TinyStories tokens (M)", "Validation loss")
    axis.plot([r["tokens"]/1e6 for r in scratch],
              [r["validation_loss"] for r in scratch],
              color="#A9D0F4", linewidth=1.45, alpha=.82, linestyle="--", label="Scratch")
    for i, (target, labels) in enumerate(DOSES):
        rows = histories[arm][target]
        axis.plot([r["tokens"]/1e6 for r in rows],
                  [r["validation_loss"] for r in rows],
                  color=COLORS[i], linewidth=1.45, alpha=.82,
                  label=f"{labels/1e6:.2f}M NCA")
    axis.set_xlim(0, 10)
    legend(axis, fontsize=8, ncol=4)
    filename = f"dose_{arm}_vs_scratch.png"
    fig.savefig(FIGURES / filename, dpi=180, facecolor=BACKGROUND)
    filenames = [filename]
    if any(histories[arm][target][-1]["tokens"] >= 100_000_000 for target, _ in DOSES[:-1]):
        axis.set_xlim(0, 100)
        full = f"dose_{arm}_vs_scratch_full.png"
        fig.savefig(FIGURES / full, dpi=180, facecolor=BACKGROUND)
        filenames.append(full)
    plt.close(fig)
    return filenames


def response_plot(histories: dict) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    milestone_colors = ("#A9D0F4", "#EFA7D0", "#F3D38A")
    for row_index, arm in enumerate(ARMS):
        for col_index, metric_name in enumerate(("loss", "AUC")):
            axis = axes[row_index, col_index]
            setup_axis(axis, "Scored NCA labels (M)",
                       "TinyStories validation loss" if col_index == 0 else "Normalized AUC")
            axis.set_title(arm.title(), color=TEXT, loc="left", fontsize=11)
            for milestone, color in zip((1, 5, 10), milestone_colors):
                xs, ys = [], []
                for target, labels in DOSES:
                    rows = histories[arm][target]
                    value = (at_or_none(rows, milestone*1_000_000) if col_index == 0 else
                             auc_or_none(rows, milestone*1_000_000))
                    if value is not None:
                        xs.append(labels/1e6)
                        ys.append(value)
                axis.plot(xs, ys, color=color, linewidth=1.45, alpha=.82,
                          marker="o", markersize=3, label=f"{milestone}M text")
            legend(axis, fontsize=8)
    filename = "dose_response.png"
    fig.savefig(FIGURES / filename, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)
    return filename


def main() -> None:
    scratch = json.loads((RUN / "downstream/scratch/metrics.json").read_text())["history"]
    histories = {arm: {target: history(arm, target) for target, _ in DOSES}
                 for arm in ARMS}
    FIGURES.mkdir(parents=True, exist_ok=True)
    figures = {arm: trajectory_plot(arm, histories, scratch) for arm in ARMS}
    response = response_plot(histories)
    lines = ["# Seed-0 NCA dose diagnostic", "",
             "Frozen 10M-parameter model, corpora, transfer, tokenizer, TinyStories order, optimizer, full 100M-token LR schedule, and validation milestones. The 10M probes pause after a saved checkpoint on the original full schedule. The existing Scratch and 197.15M-label histories are reused. This is one model seed, not a replication.", "",
             "![Dose response](figures/" + response + ")", ""]
    lines += ["## Dose comparison against the original 197.15M setting", "",
              "Differences below are shorter-dose minus original 197.15M-dose validation metrics within the same NCA arm. Negative values mean the shorter dose measured lower loss or AUC. The Scratch comparator is shared across doses, so the same loss difference is the change in that arm's paired gap versus Scratch.", ""]
    summary=[]
    for arm in ARMS:
        baseline=histories[arm][200]
        for endpoint in (1_000_000,5_000_000,10_000_000,100_000_000):
            observed=[(target,labels,at_or_none(histories[arm][target],endpoint))
                      for target,labels in DOSES[:-1]]
            available=[row for row in observed if row[2] is not None]
            if not available:
                continue
            best=min(available,key=lambda row:(row[2],row[1]))
            reference=calc.at(baseline,endpoint)
            summary.append([arm.title(),f"{endpoint/1e6:g}M loss",
                            f"{best[1]/1e6:.3f}M",fmt(best[2]),fmt(reference),
                            f"{best[2]-reference:+.5f}"])
        for endpoint in (1_000_000,5_000_000,10_000_000):
            observed=[(target,labels,auc_or_none(histories[arm][target],endpoint))
                      for target,labels in DOSES[:-1]]
            available=[row for row in observed if row[2] is not None]
            best=min(available,key=lambda row:(row[2],row[1]))
            reference=calc.auc(baseline,0,endpoint)
            summary.append([arm.title(),f"0–{endpoint/1e6:g}M AUC",
                            f"{best[1]/1e6:.3f}M",fmt(best[2]),fmt(reference),
                            f"{best[2]-reference:+.5f}"])
    table(lines,["Arm","Metric","Best shorter NCA dose","Shorter value","197.15M value",
                 "Shorter − 197.15M"],summary)
    for arm in ARMS:
        lines += [f"## {arm.title()} NCA", ""]
        lines += [f"![Trajectories](figures/{figure})" for figure in figures[arm]]
        lines += ["",
                  "The dose label is the exact scored-NCA-label count of the saved seed-0 checkpoint. Doses at 10, 25, 50, 100, and 150M are new downstream branches. The 197.15M row is the original saved run.", ""]
        rows = []
        for target, labels in DOSES:
            h = histories[arm][target]
            for row in h:
                match = next(s for s in scratch if s["tokens"] == row["tokens"])
                rows.append([f"{labels/1e6:.3f}M", f"{row['tokens']/1e6:.3f}M",
                             fmt(row["validation_loss"]), fmt(row["perplexity"], 2),
                             fmt(row["validation_loss"]-match["validation_loss"])])
        table(lines,["NCA labels","Text tokens","Validation loss","PPL","Loss − Scratch"],rows)
        rows=[]
        for target, labels in DOSES:
            h=histories[arm][target]
            crossing,direction=calc.first_crossing(h,scratch[:len(h)])
            rows.append([f"{labels/1e6:.3f}M",f"{h[-1]['tokens']/1e6:.3f}M",
                         fmt(auc_or_none(h,5_000_000)),fmt(auc_or_none(h,10_000_000)),
                         "—" if crossing is None else f"{crossing/1e6:.3f}M ({direction})"])
        lines += ["### AUC and first Scratch crossing", ""]
        table(lines,["NCA labels","Text observed","0–5M AUC","0–10M AUC",
                     "First loss-curve crossing"],rows)
        lines += ["### First downward crossing of fixed absolute losses", "",
                  "Linear interpolation between logged validation checkpoints only; no extrapolation. An em dash means the target was not reached within that branch's observed downstream budget.", ""]
        rows=[]
        for target, labels in DOSES:
            h=histories[arm][target]
            rows.append([f"{labels/1e6:.3f}M",*["—" if (v:=calc.target_tokens(h,loss)) is None
                        else f"{v/1e6:.3f}M" for loss in TARGETS]])
        table(lines,["NCA labels",*[f"Loss {loss:g}" for loss in TARGETS]],rows)
        lines += ["### Fixed downstream milestones", "",
                  "Each cell is validation loss followed by the paired difference versus Scratch in parentheses. Values use only interpolation within the observed curve.", ""]
        rows=[]
        for target, labels in DOSES:
            h=histories[arm][target]
            values=[]
            for milestone in CHECKPOINTS:
                value=at_or_none(h,milestone)
                values.append("—" if value is None else
                              f"{value:.4f} ({value-calc.at(scratch,milestone):+.4f})")
            rows.append([f"{labels/1e6:.3f}M",*values])
        table(lines,["NCA labels",*[f"{x/1e6:g}M" for x in CHECKPOINTS]],rows)
    decision=SWEEP/"extension_decision.json"
    if decision.exists():
        selection=json.loads(decision.read_text())
        lines += ["## Frozen 100M extension decision", "",
                  selection["rule"], "",
                  f"- Ordinary: {selection['selected']['ordinary'] or 'none'}", 
                  f"- Designed: {selection['selected']['designed'] or 'none'}", ""]
    lines += ["## Scope", "",
              "All rows reuse the original seed-0 NCA checkpoints and fixed packed corpora. The 10M probe uses the 100M schedule and the same transfer function as the original run. Selected branches resume their saved model and optimizer. This is a seed-0 dose diagnostic; any apparent optimum needs independent model seeds. Downstream-token results do not include NCA training compute.", ""]
    OUT.write_text("\n".join(lines))
    print(f"WROTE {OUT}",flush=True)


if __name__ == "__main__":
    main()
