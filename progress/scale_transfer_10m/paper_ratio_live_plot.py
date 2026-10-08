"""Refresh the seed-0 NCA switch-dose validation curves from saved metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/scale-transfer-10m"
ROOT = RUN / "paper-ratio/seed_0"
OLD = RUN / "dose-sweep/seed_0"
OUTPUT = RUN / "paper-ratio/live_loss.png"
COLORS = {"scratch": "#A9D0F4", "designed_2": "#9DD9D2",
          "designed_4": "#C9B4E9", "designed_9": "#F3B49F",
          "designed_197": "#F3D38A", "ordinary_2": "#EFAAD0",
          "ordinary_4": "#EFAAD0"}
LABELS = {"scratch": "Scratch", "designed_2": "Designed · 2.001M",
          "designed_4": "Designed · 4.001M", "designed_9": "Designed · 9.869M",
          "designed_197": "Designed · 197.153M",
          "ordinary_2": "Ordinary · 2.001M", "ordinary_4": "Ordinary · 4.001M"}
BACKGROUND, PANEL, TEXT, MUTED, GRID = "#111722", "#192230", "#E8EDF4", "#A9B5C4", "#344255"


def sources() -> dict[str, Path]:
    paths = {
        "scratch": RUN / "downstream/scratch/metrics.json",
        "designed_2": ROOT / "downstream/designed_2m/metrics.json",
        "designed_4": ROOT / "downstream/designed_4m/metrics.json",
        "designed_9": OLD / "downstream/designed_dose_10m/metrics.json",
        "designed_197": RUN / "downstream/designed/metrics.json",
    }
    paths.update({f"ordinary_{dose}": ROOT / f"downstream/ordinary_{dose}m/metrics.json"
                  for dose in (2, 4)})
    return paths


def read(path: Path) -> list[dict]:
    return json.loads(path.read_text())["history"] if path.is_file() else []


def style(axis) -> None:
    axis.set_facecolor(PANEL)
    axis.set_xlabel("TinyStories tokens (M)", color=TEXT)
    axis.set_ylabel("Validation loss", color=TEXT)
    axis.tick_params(colors=MUTED)
    axis.grid(color=GRID, linewidth=.7, alpha=.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color(GRID)


def curve(axis, name: str, rows: list[dict], *, labeled: bool = True) -> None:
    if not rows:
        return
    axis.plot([row["tokens"] / 1e6 for row in rows],
              [row["validation_loss"] for row in rows],
              color=COLORS[name], linewidth=1.35, alpha=.78,
              linestyle="--" if name == "scratch" else "-",
              label=LABELS[name] if labeled else None)


def draw(fig, axis, paths: dict[str, Path]) -> None:
    axis.clear()
    histories = {name: read(path) for name, path in paths.items()}
    style(axis)
    for name in paths:
        curve(axis, name, histories[name])
    full = any(histories[name] and histories[name][-1]["tokens"] > 10_000_384
               for name in ("designed_2", "designed_4", "ordinary_2", "ordinary_4"))
    axis.set_xlim(0, 100 if full else 10)
    axis.set_ylim(bottom=1.5)
    handles, labels = axis.get_legend_handles_labels()
    legend = axis.legend(handles, labels, frameon=False, loc="upper right",
                         ncol=2, fontsize=8)
    for label in legend.get_texts():
        label.set_color(TEXT)
    if full:
        inset = axis.inset_axes([.51, .43, .46, .40])
        style(inset)
        inset.set_xlabel("")
        inset.set_ylabel("")
        inset.tick_params(labelsize=7)
        for name in paths:
            curve(inset, name, histories[name], labeled=False)
        inset.set_xlim(0, 10)
        inset.set_ylim(1.5, 5)


def complete() -> bool:
    log = RUN / "paper-ratio/runner.log"
    if not log.is_file():
        return False
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 4096))
        return b"PAPER-RATIO DIAGNOSTIC COMPLETE" in stream.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--until-run-ends", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.output.suffix.lower() != ".png":
        parser.error("--interval must be positive and --output must be a .png path")
    paths = sources()
    fig, axis = plt.subplots(figsize=(10, 5.5), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    signature = None
    try:
        while True:
            current = tuple((path.stat().st_mtime_ns, path.stat().st_size)
                            if path.is_file() else None for path in paths.values())
            if args.once or current != signature:
                draw(fig, axis, paths)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_name(f".{args.output.stem}.{os.getpid()}.tmp.png")
                fig.savefig(temporary, dpi=160, facecolor=BACKGROUND)
                os.replace(temporary, args.output)
                print(f"Updated {args.output}", flush=True)
                signature = current
            if args.once or args.until_run_ends and complete():
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
