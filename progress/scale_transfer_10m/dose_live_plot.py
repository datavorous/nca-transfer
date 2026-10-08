"""Poll saved seed-0 dose histories and draw live validation-loss curves."""

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
SWEEP = RUN / "dose-sweep/seed_0"
DOSES = ((10, "9.87M"), (25, "29.60M"), (50, "49.35M"),
         (100, "100.00M"), (150, "148.04M"))
DOSE_COLORS = ("#9DD9D2", "#C9B4E9", "#F3B49F", "#ABD49F", "#EFAAD0")
SCRATCH, FULL = "#A9D0F4", "#F3D38A"
BACKGROUND, PANEL, TEXT, MUTED, GRID = "#111722", "#192230", "#E8EDF4", "#A9B5C4", "#344255"


def sources() -> list[Path]:
    paths = [RUN / "downstream/scratch/metrics.json"]
    for arm in ("ordinary", "designed"):
        paths.append(RUN / "downstream" / arm / "metrics.json")
        paths.extend(SWEEP / "downstream" / f"{arm}_dose_{dose}m" / "metrics.json"
                     for dose, _ in DOSES)
    return paths


def read(path: Path) -> list[dict]:
    return json.loads(path.read_text())["history"] if path.is_file() else []


def style(axis, title: str, xlim: int) -> None:
    axis.set_facecolor(PANEL)
    axis.set_title(title, loc="left", fontsize=12, color=TEXT)
    axis.set_xlabel("TinyStories tokens (M)", color=TEXT)
    axis.set_ylabel("Validation loss", color=TEXT)
    axis.tick_params(colors=MUTED)
    axis.grid(color=GRID, linewidth=.8, alpha=.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color(GRID)
    axis.set_xlim(0, xlim)


def curve(axis, rows: list[dict], color: str, label: str, *, dashed: bool = False) -> None:
    if rows:
        axis.plot([row["tokens"] / 1e6 for row in rows],
                  [row["validation_loss"] for row in rows],
                  color=color, linewidth=1.45, alpha=.82,
                  linestyle="--" if dashed else "-", label=label)


def draw(fig, axes, paths: list[Path]) -> None:
    scratch = read(paths[0])
    for arm_index, (axis, arm) in enumerate(zip(axes, ("ordinary", "designed"))):
        for child in list(axis.child_axes):
            child.remove()
        axis.clear()
        anchor = read(paths[1 + arm_index * 6])
        branches = [read(path) for path in paths[2 + arm_index * 6:7 + arm_index * 6]]
        extended = any(rows and rows[-1]["tokens"] > 10_000_384 for rows in branches)
        style(axis, "Ordinary NCA" if arm == "ordinary" else "Designed NCA",
              100 if extended else 10)
        curve(axis, scratch, SCRATCH, "Scratch", dashed=True)
        curve(axis, anchor, FULL, "197.15M NCA")
        for (_, label), rows, color in zip(DOSES, branches, DOSE_COLORS):
            curve(axis, rows, color, f"{label} NCA")
        legend = axis.legend(frameon=False, loc="upper right", ncol=4, fontsize=8)
        for label in legend.get_texts():
            label.set_color(TEXT)
        if extended:
            inset = axis.inset_axes([.54, .40, .42, .43])
            inset.set_facecolor(PANEL)
            inset.tick_params(colors=MUTED, labelsize=7)
            inset.grid(color=GRID, linewidth=.6, alpha=.5)
            for spine in inset.spines.values():
                spine.set_color(GRID)
            curve(inset, scratch, SCRATCH, "", dashed=True)
            curve(inset, anchor, FULL, "")
            for rows, color in zip(branches, DOSE_COLORS):
                curve(inset, rows, color, "")
            inset.set_xlim(0, 10)
            inset.set_ylim(1.8, 4.4)


def complete(log: Path) -> bool:
    if not log.is_file():
        return False
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 4096))
        return b"DOSE DIAGNOSTIC COMPLETE" in stream.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN / "dose-sweep/live_loss.png")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--until-run-ends", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be a .png path")
    paths = sources()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    last_signature = None
    try:
        while True:
            signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else None
                              for p in paths)
            if args.once or signature != last_signature:
                draw(fig, axes, paths)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_name(f".{args.output.stem}.{os.getpid()}.tmp.png")
                fig.savefig(temporary, dpi=150, facecolor=BACKGROUND)
                os.replace(temporary, args.output)
                print(f"Updated {args.output}", flush=True)
                last_signature = signature
            if args.once or args.until_run_ends and complete(RUN / "dose-sweep/runner.log"):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
