"""Read-only tables and figures for the seed-0 paper-ratio dose pivot."""

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
ROOT = RUN / "paper-ratio/seed_0"
OLD = RUN / "dose-sweep/seed_0"
OUT = Path(__file__).with_name("PAPER_RATIO_RESULTS.md")
FIGURES = OUT.parent / "figures"
TARGETS = (3.0, 2.5, 2.25, 2.0, 1.9)
MILESTONES = (0, 251_904, 501_760, 1_001_472, 2_000_896, 3_000_320,
              5_001_216, 7_501_824, 10_000_384, 15_001_600, 20_000_768,
              30_001_152, 50_001_920, 75_001_856, 100_000_000)
BACKGROUND, PANEL, TEXT, MUTED, GRID = "#111722", "#192230", "#E8EDF4", "#A9B5C4", "#344255"
COLORS = {"scratch": "#A9D0F4", "designed_2": "#9DD9D2", "designed_4": "#C9B4E9",
          "designed_9": "#F3B49F", "designed_197": "#F3D38A", "ordinary": "#EFAAD0"}


def load(path: Path) -> list[dict] | None:
    return json.loads(path.read_text())["history"] if path.is_file() else None


def histories() -> dict[str, list[dict]]:
    paths = {
        "scratch": RUN / "downstream/scratch/metrics.json",
        "designed_2": ROOT / "downstream/designed_2m/metrics.json",
        "designed_4": ROOT / "downstream/designed_4m/metrics.json",
        "designed_9": OLD / "downstream/designed_dose_10m/metrics.json",
        "designed_197": RUN / "downstream/designed/metrics.json",
        "ordinary_9": OLD / "downstream/ordinary_dose_10m/metrics.json",
        "ordinary_197": RUN / "downstream/ordinary/metrics.json",
    }
    result = {name: rows for name, path in paths.items() if (rows := load(path)) is not None}
    for dose in (2, 4):
        key = f"ordinary_{dose}"
        rows = load(ROOT / f"downstream/ordinary_{dose}m/metrics.json")
        if rows is not None:
            result[key] = rows
    return result


def dose_labels(name: str) -> int:
    if name == "scratch": return 0
    return {"2": 2_001_179, "4": 4_001_087,
            "9": 9_869_186, "197": 197_153_250}[name.rsplit("_", 1)[1]]


def fmt(value: float | None, digits: int = 5) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def table(lines: list[str], headers: list[str], rows: list[list[object]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join([":--"] + ["--:" for _ in headers[1:]]) + "|")
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def at(rows: list[dict], tokens: int) -> float | None:
    return calc.at(rows, tokens) if rows[-1]["tokens"] >= tokens else None


def area(rows: list[dict], tokens: int) -> float | None:
    return calc.auc(rows, 0, tokens) if rows[-1]["tokens"] >= tokens else None


def axis_style(axis, xlabel: str, ylabel: str) -> None:
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


def dose_response(h: dict[str, list[dict]]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    keys = [name for name in ("scratch", "designed_2", "designed_4",
                              "designed_9", "designed_197") if name in h]
    x = [dose_labels(name) / 1e6 for name in keys]
    for axis, metric in zip(axes, ("loss", "AUC")):
        axis_style(axis, "NCA switch dose (M scored labels)",
                   "TinyStories validation loss" if metric == "loss" else "Normalized AUC")
        for milestone, color in ((1_000_000, "#A9D0F4"),
                                 (5_000_000, "#EFAAD0"),
                                 (10_000_000, "#F3D38A"),
                                 (100_000_000, "#9DD9D2")):
            points = [(position, (at(h[name], milestone) if metric == "loss" else
                                  area(h[name], milestone))) for name, position in zip(keys, x)]
            available = [(a, b) for a, b in points if b is not None]
            if available:
                axis.plot([a for a, _ in available], [b for _, b in available],
                          color=color, linewidth=1.45, alpha=.82, marker="o", markersize=3,
                          label=f"{milestone/1e6:g}M text")
        axis.set_xscale("symlog", linthresh=2)
        legend(axis, fontsize=8)
    filename = "paper_ratio_dose_response.png"
    fig.savefig(FIGURES / filename, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)
    return filename


def trajectories(h: dict[str, list[dict]], matched: int | None) -> list[str]:
    fig, axis = plt.subplots(figsize=(10, 5.4), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    axis_style(axis, "TinyStories tokens (M)", "Validation loss")
    keys = ["scratch", "designed_2", "designed_4", "designed_9", "designed_197"]
    if matched is not None:
        keys.append(f"ordinary_{matched}")
    for name in keys:
        if name not in h:
            continue
        rows = h[name]
        axis.plot([row["tokens"] / 1e6 for row in rows],
                  [row["validation_loss"] for row in rows],
                  color=COLORS.get(name, COLORS["ordinary"]), linewidth=1.45,
                  alpha=.82, linestyle="--" if name == "scratch" else "-",
                  label=name.replace("_", " ").title())
    legend(axis, fontsize=8, ncol=3)
    filenames = []
    axis.set_xlim(0, 10)
    short = "paper_ratio_trajectories_10m.png"
    fig.savefig(FIGURES / short, dpi=180, facecolor=BACKGROUND)
    filenames.append(short)
    if any(h[name][-1]["tokens"] >= 100_000_000 for name in ("designed_2", "designed_4")
           if name in h):
        axis.set_xlim(0, 100)
        full = "paper_ratio_trajectories_100m.png"
        fig.savefig(FIGURES / full, dpi=180, facecolor=BACKGROUND)
        filenames.append(full)
    plt.close(fig)
    return filenames


def main() -> None:
    h = histories()
    assert "scratch" in h and "designed_2" in h and "designed_4" in h
    FIGURES.mkdir(parents=True, exist_ok=True)
    decision_path = ROOT / "final_decision.json"
    matched = json.loads(decision_path.read_text())["matched_ordinary_dose"] if decision_path.is_file() else None
    response = dose_response(h)
    plots = trajectories(h, matched)
    lines = ["# Paper-ratio NCA pre-pretraining dose: seed 0", "",
             "A fixed 100M-token TinyStories stream and ~10M-parameter model are used throughout. NCA corpora, selector, architecture, tokenizer, transfer operation, optimizer, and evaluation milestones are frozen. The intervention is **NCA switch dose**: when training leaves the same frozen NCA optimization trajectory for language. The 2M and 4M checkpoints use the original full-197M NCA learning-rate schedule; they are not independently optimized short runs. Seed 0 selects a dose, so this is an exploratory single-seed diagnostic rather than held-out confirmation.", "",
             "## Switch-time curve", "",
             "Each column below holds downstream TinyStories tokens fixed and varies only the scored NCA labels before transfer. Values are validation loss; `—` means the saved curve has not reached that downstream checkpoint. The 9.87M and 197.15M arms are existing context runs.", "",
             f"![NCA switch dose versus downstream validation loss](figures/{response})", ""]
    switch_rows = []
    for name in ("scratch", "designed_2", "designed_4", "designed_9", "designed_197"):
        if name not in h:
            continue
        switch_rows.append([name.replace("_", " ").title(), f"{dose_labels(name):,}",
                            *[fmt(at(h[name], token)) for token in
                              (1_001_472, 5_001_216, 10_000_384, 100_000_000)]])
    table(lines, ["Condition", "NCA switch dose", "1M text", "5M text", "10M text",
                  "100M text"], switch_rows)
    for plot in plots:
        lines += [f"![Downstream trajectories](figures/{plot})", ""]
    lines += ["## Synthetic-to-language ratio", "",
              "Ratios below use the fixed **planned** 100M TinyStories-token horizon; the observed horizon is listed separately. The 9.87M and 197.15M context arms reuse saved histories.", ""]
    ratio_rows=[]
    for name in ("scratch", "designed_2", "designed_4", "designed_9",
                 "designed_197", "ordinary_2", "ordinary_4", "ordinary_9", "ordinary_197"):
        if name not in h:
            continue
        labels=dose_labels(name)
        ratio_rows.append([name.replace("_", " ").title(),f"{labels:,}",
                           "0" if not labels else f"1:{100_000_000/labels:.2f}" if labels <= 100_000_000 else
                           f"{labels/100_000_000:.3f}:1",
                           f"{h[name][-1]['tokens']:,}"])
    table(lines,["Condition","Scored NCA labels","NCA:text at 100M text","Text observed"],ratio_rows)
    lines += ["## Early validation loss: paired conditions", "",
              "These are the exact logged downstream checkpoints; lower is better. A missing cell means training has not reached it.", ""]
    rows=[]
    for name in ("scratch", "designed_2", "designed_4", "designed_9", "designed_197",
                 "ordinary_2", "ordinary_4", "ordinary_197"):
        if name not in h: continue
        d={row["tokens"]:row["validation_loss"] for row in h[name]}
        rows.append([name.replace("_", " ").title(),
                     *[fmt(d.get(token)) for token in (1_001_472,2_000_896,3_000_320,
                                                      5_001_216,7_501_824,10_000_384)]])
    table(lines,["Condition","1M","2M","3M","5M","7.5M","10M"],rows)
    lines += ["## Every observed validation checkpoint", "",
              "PPL is `exp(loss)`. Differences are paired to the same seed-0 Scratch checkpoint. Designed minus Ordinary is shown only at the matched small NCA dose.", ""]
    rows=[]
    for name in ("scratch", "designed_2", "designed_4", "designed_9",
                 "designed_197", "ordinary_2", "ordinary_4", "ordinary_9", "ordinary_197"):
        if name not in h: continue
        for row in h[name]:
            scratch_loss=next(s["validation_loss"] for s in h["scratch"]
                              if s["tokens"]==row["tokens"])
            counterpart=None
            if matched is not None and name == f"designed_{matched}" and f"ordinary_{matched}" in h:
                other=h[f"ordinary_{matched}"]
                counterpart=next((s["validation_loss"] for s in other
                                  if s["tokens"]==row["tokens"]),None)
            rows.append([name.replace("_", " ").title(),f"{row['tokens']:,}",
                         fmt(row["validation_loss"]),fmt(row["perplexity"],2),
                         f"{row['validation_loss']-scratch_loss:+.5f}",
                         "—" if counterpart is None else f"{row['validation_loss']-counterpart:+.5f}"])
    table(lines,["Condition","Text tokens","Loss","PPL","Loss − Scratch",
                 "Designed − matched Ordinary"],rows)
    lines += ["## AUC, fixed-loss crossings, and Scratch crossover", "",
              "AUC uses trapezoidal integration and boundary interpolation. Fixed-loss crossings use only the first downward linear interpolation between logged checkpoints. No extrapolation.", ""]
    rows=[]
    for name in ("scratch", "designed_2", "designed_4", "designed_9",
                 "designed_197", "ordinary_2", "ordinary_4", "ordinary_9", "ordinary_197"):
        if name not in h:continue
        curve=h[name]
        crossing,direction=calc.first_crossing(curve,h["scratch"][:len(curve)])
        rows.append([name.replace("_", " ").title(),fmt(area(curve,10_000_000)),
                     fmt(area(curve,100_000_000)),
                     "—" if crossing is None else f"{crossing/1e6:.3f}M ({direction})",
                     *["—" if (value:=calc.target_tokens(curve,target)) is None else
                       f"{value/1e6:.3f}M" for target in TARGETS]])
    table(lines,["Condition","0–10M AUC","0–100M AUC","First Scratch crossing",
                 *[f"Loss {target:g}" for target in TARGETS]],rows)
    for filename,title in (("early_decision.json","Early extension decision"),
                           ("final_decision.json","Matched Ordinary dose")):
        path=ROOT/filename
        if path.is_file():
            lines += [f"## {title}", "", "```json", path.read_text().strip(), "```", ""]
    lines += ["## Limits", "",
              "A paper-like NCA:text ratio is a tested regime, not an established optimum. The 2M and 4M checkpoints are early stops on one fixed NCA rule stream, initialization, and full-stage optimizer schedule. Seed 0 is used to select the dose; confirmation would freeze it and test held-out model seeds 1–4, which this runner does not launch. Downstream-token advantages omit synthetic pretraining compute and wall-clock cost. Designed-versus-Scratch and Designed-versus-matched-Ordinary are separate questions. The observed switch-time curve alone does not establish a causal mechanism for any advantage.", ""]
    OUT.write_text("\n".join(lines))
    print(f"WROTE {OUT}",flush=True)


if __name__ == "__main__":
    main()
