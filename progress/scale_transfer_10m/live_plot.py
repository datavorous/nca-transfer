"""Poll frozen scale-transfer logs and plot seed trajectories plus their shared-range mean."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/scale-transfer-10m"
PATTERN = re.compile(
    r"^PROGRESS (?P<stage>(?:nca|downstream)/(?:scratch|ordinary|designed)) "
    r"step=\d+/\d+ tokens=(?P<tokens>\d+) train_loss=(?P<loss>[-+\d.eE]+)"
)
SEED_PATTERN = re.compile(r"^SEED (?P<seed>[1-4]) ")
COLORS = {"scratch": "#A9D0F4", "ordinary": "#EFA7D0", "designed": "#F3D38A"}
LABELS = {"scratch": "Scratch", "ordinary": "Ordinary NCA", "designed": "Designed NCA"}
BACKGROUND = "#111722"
PANEL = "#192230"
TEXT = "#E8EDF4"
MUTED = "#A9B5C4"
GRID = "#344255"


def read_curves(seed0_log: Path, replication_log: Path) -> dict[tuple[int, str], list[tuple[int, float]]]:
    """Use the latest logged value at each token count after a resumed attempt."""
    points: dict[tuple[int, str], dict[int, float]] = {}
    for path, default_seed in ((seed0_log, 0), (replication_log, None)):
        if not path.is_file():
            continue
        current_seed = default_seed
        with path.open(errors="replace") as stream:
            for line in stream:
                seed_match = SEED_PATTERN.match(line)
                if seed_match:
                    current_seed = int(seed_match["seed"])
                match = PATTERN.match(line)
                if match and current_seed is not None:
                    points.setdefault((current_seed, match["stage"]), {})[
                        int(match["tokens"])] = float(match["loss"])
    return {key: sorted(rows.items()) for key, rows in points.items()}


def shared_mean(rows_by_seed: dict[int, list[tuple[int, float]]]) -> list[tuple[int, float]]:
    """Average only token positions reached by every currently available seed."""
    maps = [{tokens: loss for tokens, loss in rows} for rows in rows_by_seed.values()]
    common = set(maps[0]).intersection(*(set(values) for values in maps[1:]))
    return [(tokens, sum(values[tokens] for values in maps) / len(maps))
            for tokens in sorted(common)]


def smooth(values: list[float], alpha: float) -> list[float]:
    result: list[float] = []
    for value in values:
        result.append(value if not result else alpha * value + (1 - alpha) * result[-1])
    return result


def draw(axes, curves: dict[tuple[int, str], list[tuple[int, float]]], alpha: float) -> None:
    for axis, prefix, title, xlabel, arms in (
        (axes[0], "nca", "NCA", "Scored NCA labels (M)", ("ordinary", "designed")),
        (axes[1], "downstream", "TinyStories", "TinyStories tokens (M)",
         ("scratch", "ordinary", "designed")),
    ):
        axis.clear()
        axis.set_facecolor(PANEL)
        axis.set_title(title, loc="left", fontsize=12, pad=8, color=TEXT)
        axis.set_xlabel(xlabel, color=TEXT)
        axis.set_ylabel("Training loss", color=TEXT)
        axis.tick_params(colors=MUTED)
        axis.grid(color=GRID, linewidth=0.8, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color(GRID)
        shown = 0
        for arm in arms:
            stage = f"{prefix}/{arm}"
            available = {seed: curves[(seed, stage)] for seed in range(5)
                         if curves.get((seed, stage))}
            if not available:
                continue
            color = COLORS[arm]
            for rows in available.values():
                x = [tokens / 1_000_000 for tokens, _ in rows]
                y = [loss for _, loss in rows]
                axis.plot(x, smooth(y, alpha), color=color, linewidth=1.0, alpha=0.29)
            mean = shared_mean(available)
            if mean:
                x = [tokens / 1_000_000 for tokens, _ in mean]
                y = smooth([loss for _, loss in mean], alpha)
                axis.plot(x, y, color=color, linewidth=2.3,
                          label=f"{LABELS[arm]} ({len(available)}/5)")
            shown += 1
        if shown:
            legend = axis.legend(frameon=False, loc="upper right", ncol=shown, fontsize=9)
            for label in legend.get_texts():
                label.set_color(TEXT)
            axis.margins(x=0.02, y=0.1)
        else:
            axis.text(0.5, 0.5, "Waiting for training logs", ha="center", va="center",
                      transform=axis.transAxes, color=MUTED)


def replication_ended(events: Path) -> bool:
    if not events.is_file():
        return False
    final = None
    with events.open(errors="replace") as stream:
        for line in stream:
            if line.strip():
                final = json.loads(line)
    return final is not None and (final["stage"] == "failed" or
                                  (final["seed"] == 4 and final["stage"] == "complete"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=RUN / "runner.log",
                        help="Original seed-0 log")
    parser.add_argument("--replication-log", type=Path,
                        default=RUN / "replication/runner.log")
    parser.add_argument("--output", type=Path, default=RUN / "live_loss.png")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Seconds between log polls (default: 30)")
    parser.add_argument("--ema", type=float, default=0.18,
                        help="EMA weight for the main curve (default: 0.18)")
    parser.add_argument("--show", action="store_true", help="Also open a live window")
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit")
    parser.add_argument("--until-run-ends", action="store_true",
                        help="Stop after seed 4 completes or the replication fails")
    args = parser.parse_args()
    if args.interval <= 0 or not 0 < args.ema <= 1:
        parser.error("--interval must be positive and --ema must be in (0, 1]")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be a .png path")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    if args.show:
        plt.ion()
        plt.show(block=False)

    last_signature = None
    try:
        while True:
            signature = tuple((path.stat().st_mtime_ns, path.stat().st_size)
                              if path.is_file() else None
                              for path in (args.log, args.replication_log))
            if args.once or signature != last_signature:
                draw(axes, read_curves(args.log, args.replication_log), args.ema)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_name(f".{args.output.stem}.{os.getpid()}.tmp.png")
                fig.savefig(temporary, dpi=150, facecolor=BACKGROUND)
                os.replace(temporary, args.output)
                if args.show:
                    fig.canvas.draw_idle()
                print(f"Updated {args.output}", flush=True)
                last_signature = signature
            if args.once or args.until_run_ends and replication_ended(RUN / "replication/events.jsonl"):
                break
            if args.show:
                plt.pause(args.interval)
                if not plt.fignum_exists(fig.number):
                    break
            else:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
