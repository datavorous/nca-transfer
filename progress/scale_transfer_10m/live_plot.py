"""Poll the scale-transfer log and plot live training loss. Ctrl-C stops polling."""

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
COLORS = {"scratch": "#79818c", "ordinary": "#cc7654", "designed": "#318b89"}
LABELS = {"scratch": "Scratch", "ordinary": "Ordinary NCA", "designed": "Designed NCA"}


def read_curves(log: Path) -> dict[str, list[tuple[float, float]]]:
    """Keep the latest logged value for a token position after a resumed attempt."""
    points: dict[str, dict[int, float]] = {}
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            match = PATTERN.match(line)
            if match:
                stage = match["stage"]
                points.setdefault(stage, {})[int(match["tokens"])] = float(match["loss"])
    return {stage: [(tokens / 1_000_000, loss) for tokens, loss in sorted(rows.items())]
            for stage, rows in points.items()}


def smooth(values: list[float], alpha: float) -> list[float]:
    result: list[float] = []
    for value in values:
        result.append(value if not result else alpha * value + (1 - alpha) * result[-1])
    return result


def draw(axes, curves: dict[str, list[tuple[float, float]]], alpha: float) -> None:
    for axis, prefix, title, xlabel, arms in (
        (axes[0], "nca", "NCA", "Scored NCA labels (M)", ("ordinary", "designed")),
        (axes[1], "downstream", "TinyStories", "TinyStories tokens (M)",
         ("scratch", "ordinary", "designed")),
    ):
        axis.clear()
        axis.set_title(title, loc="left", fontsize=12, pad=8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Training loss")
        axis.grid(color="#dce2e6", linewidth=0.8, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        shown = 0
        for arm in arms:
            rows = curves.get(f"{prefix}/{arm}", [])
            if not rows:
                continue
            x, y = zip(*rows)
            color = COLORS[arm]
            axis.plot(x, y, color=color, linewidth=0.8, alpha=0.20)
            axis.plot(x, smooth(list(y), alpha), color=color, linewidth=2.2,
                      label=LABELS[arm])
            shown += 1
        if shown:
            axis.legend(frameon=False, loc="upper right", ncol=shown, fontsize=9)
            axis.margins(x=0.02, y=0.1)
        else:
            axis.text(0.5, 0.5, "Waiting for training logs", ha="center", va="center",
                      transform=axis.transAxes, color="#88919b")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=RUN / "runner.log")
    parser.add_argument("--output", type=Path, default=RUN / "live_loss.png")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Seconds between log polls (default: 5)")
    parser.add_argument("--ema", type=float, default=0.18,
                        help="EMA weight for the main curve (default: 0.18)")
    parser.add_argument("--show", action="store_true", help="Also open a live window")
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit")
    parser.add_argument("--until-run-ends", action="store_true",
                        help="Stop after the runner records complete or failed")
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
    fig.patch.set_facecolor("white")
    if args.show:
        plt.ion()
        plt.show(block=False)

    last_signature = None
    try:
        while True:
            signature = None
            if args.log.is_file():
                stat = args.log.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
            if args.once or signature != last_signature:
                draw(axes, read_curves(args.log), args.ema)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_name(f".{args.output.stem}.tmp.png")
                fig.savefig(temporary, dpi=150)
                os.replace(temporary, args.output)
                if args.show:
                    fig.canvas.draw_idle()
                print(f"Updated {args.output}", flush=True)
                last_signature = signature
            if args.once:
                break
            if args.until_run_ends:
                events = RUN / "events.jsonl"
                if events.is_file():
                    lines = events.read_text().splitlines()
                    if lines and json.loads(lines[-1]).get("stage") in ("complete", "failed"):
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
