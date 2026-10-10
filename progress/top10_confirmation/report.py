"""Summarize completed confirmation runs without training or selecting anything."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/nca-data-search/top10-confirmation"
SOURCE = ROOT / "runs/1m-nca-transfer-v2/seeds"
OUT = Path(__file__).with_name("RESULTS.md")


def history(path: Path) -> list[dict]:
    return json.loads(path.read_text())["history"]


def crossing(rows: list[dict], target: float) -> float | None:
    for a, b in zip(rows, rows[1:]):
        if a["validation_loss"] >= target >= b["validation_loss"]:
            fraction = (a["validation_loss"] - target) / (a["validation_loss"] - b["validation_loss"])
            return a["tokens"] + fraction * (b["tokens"] - a["tokens"])
    return None


def auc(rows: list[dict]) -> float:
    assert rows[0]["tokens"] == 0 and rows[-1]["tokens"] == 10_000_000
    return sum((b["tokens"] - a["tokens"]) *
               (a["validation_loss"] + b["validation_loss"]) / 2
               for a, b in zip(rows, rows[1:])) / 10_000_000


def fmt(value: float | None, digits: int = 5) -> str:
    return "not reached" if value is None else f"{value:.{digits}f}"


def main() -> None:
    audit = json.loads((RUN / "audit.json").read_text())
    assert audit["earlier_selected_overlap"] == audit["cross_block_or_arm_overlap"] == 0
    rows = []
    for block in "abc":
        for seed in range(5):
            scratch = history(SOURCE / f"seed_{seed}/downstream/scratch/metrics.json")
            target = scratch[-1]["validation_loss"]
            conditions = {}
            for arm in ("ordinary", "designed"):
                path = RUN / f"block-{block}/{arm}/seeds/seed_{seed}/downstream/full_10m"
                done = json.loads((path / "done.json").read_text())
                assert done["tokens_consumed"] == 10_000_000
                curve = history(path / "metrics.json")
                assert [r["tokens"] for r in curve] == [r["tokens"] for r in scratch]
                t = crossing(curve, target)
                conditions[arm] = {"loss": curve[-1]["validation_loss"],
                                   "ppl": curve[-1]["perplexity"], "auc": auc(curve),
                                   "crossing": t, "speedup": None if t is None else 10_000_000 / t,
                                   "curve": curve}
            ordinary, designed = conditions["ordinary"], conditions["designed"]
            rows.append({"block": block.upper(), "seed": seed, "scratch": target,
                         "ordinary": ordinary, "designed": designed,
                         "gain": ordinary["loss"] - designed["loss"],
                         "auc_gain": ordinary["auc"] - designed["auc"],
                         "speedup_gain": None if ordinary["speedup"] is None or designed["speedup"] is None
                         else designed["speedup"] - ordinary["speedup"]})
    lines = ["# Frozen Top-10 fresh-corpus confirmation", "",
             "Completed after freezing [PROTOCOL.md](PROTOCOL.md). Three disjoint corpus blocks are the independent generation units; five model initializations per block are paired within-block variation. Designed and Ordinary use contemporaneous, pre-feature hash-assigned rules; Scratch is the saved seed-matched deterministic control. Lower loss/AUC and higher speedup are better.", "",
             "## Every block and seed", "",
             "| Block | Seed | Scratch loss | Ordinary loss | Designed loss | Scratch−Ordinary | Scratch−Designed | Ordinary−Designed | Ordinary PPL | Designed PPL | Ordinary AUC | Designed AUC | Ordinary crossing (M) | Designed crossing (M) | Ordinary speedup | Designed speedup |",
             "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        o, d = r["ordinary"], r["designed"]
        lines.append(f"| {r['block']} | {r['seed']} | {r['scratch']:.5f} | {o['loss']:.5f} | "
                     f"{d['loss']:.5f} | {r['scratch']-o['loss']:+.5f} | "
                     f"{r['scratch']-d['loss']:+.5f} | {r['gain']:+.5f} | "
                     f"{o['ppl']:.2f} | {d['ppl']:.2f} | "
                     f"{o['auc']:.5f} | {d['auc']:.5f} | "
                     f"{fmt(None if o['crossing'] is None else o['crossing']/1e6,3)} | "
                     f"{fmt(None if d['crossing'] is None else d['crossing']/1e6,3)} | "
                     f"{fmt(o['speedup'],3)} | {fmt(d['speedup'],3)} |")
    lines += ["", "## Corpus-level analysis", "",
              "| Block | Mean Ordinary loss | Mean Designed loss | Mean paired loss gain | Within-block seed SD of gain | Designed wins | Mean AUC gain | Mean speedup gain |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    block_means = []
    for block in "ABC":
        subset = [r for r in rows if r["block"] == block]
        gains = [r["gain"] for r in subset]
        speed = [r["speedup_gain"] for r in subset]
        block_means.append({"gain": stats.mean(gains), "auc_gain": stats.mean(r["auc_gain"] for r in subset),
                            "speedup_gain": None if any(v is None for v in speed) else stats.mean(speed),
                            "ordinary_loss": stats.mean(r["ordinary"]["loss"] for r in subset),
                            "designed_loss": stats.mean(r["designed"]["loss"] for r in subset)})
        lines.append(f"| {block} | {block_means[-1]['ordinary_loss']:.5f} | "
                     f"{block_means[-1]['designed_loss']:.5f} | {stats.mean(gains):+.5f} | "
                     f"{stats.stdev(gains):.5f} | {sum(g>0 for g in gains)}/5 | "
                     f"{block_means[-1]['auc_gain']:+.5f} | {fmt(block_means[-1]['speedup_gain'])} |")
    gain_means = [b["gain"] for b in block_means]
    auc_means = [b["auc_gain"] for b in block_means]
    lines += ["", f"Across-block mean ± sample SD of paired 10M loss gain: **{stats.mean(gain_means):+.5f} ± {stats.stdev(gain_means):.5f}** (N=3 independent corpus blocks).",
              f"Across-block mean ± sample SD of paired 0–10M AUC gain: **{stats.mean(auc_means):+.5f} ± {stats.stdev(auc_means):.5f}**.",
              f"Designed wins: **{sum(r['gain']>0 for r in rows)}/15 seed pairs** and **{sum(v>0 for v in gain_means)}/3 block means**. Seed pairs do not enlarge the independent-corpus N.", ""]
    for arm in ("ordinary", "designed"):
        block_gain = [stats.mean(r["scratch"] - r[arm]["loss"] for r in rows if r["block"] == block)
                      for block in "ABC"]
        lines.append(f"{arm.title()} versus Scratch: across-block mean ± sample SD loss gain "
                     f"**{stats.mean(block_gain):+.5f} ± {stats.stdev(block_gain):.5f}**; "
                     f"wins **{sum(r[arm]['loss'] < r['scratch'] for r in rows)}/15 seed pairs**, "
                     f"**{sum(v>0 for v in block_gain)}/3 block means**.")
    if all(b["speedup_gain"] is not None for b in block_means):
        vals = [b["speedup_gain"] for b in block_means]
        lines.append(f"Across-block mean ± sample SD of Designed−Ordinary scratch-target speedup: **{stats.mean(vals):+.3f} ± {stats.stdev(vals):.3f}×**.")
    else:
        lines.append("Some scratch targets were not reached by 10M; no crossing was extrapolated or speedup mean reported for those blocks.")
    lines += ["", "## Full downstream curve", "",
              "Each condition is averaged within corpus block over five model seeds, then across the three block means. Scratch is the same saved seed-matched reference in every block, so repetition does not make it three independent scratch corpora.", "",
              "| TinyStories tokens | Scratch loss | Ordinary loss | Designed loss | Ordinary−Designed |",
              "|--:|--:|--:|--:|--:|"]
    first = rows[0]["ordinary"]["curve"]
    for i, point in enumerate(first):
        per_block = []
        for block in "ABC":
            subset = [r for r in rows if r["block"] == block]
            per_block.append({arm: stats.mean(r[arm]["curve"][i]["validation_loss"] for r in subset)
                              for arm in ("ordinary", "designed")})
        scratch_mean = stats.mean(history(SOURCE / f"seed_{seed}/downstream/scratch/metrics.json")[i]["validation_loss"]
                                  for seed in range(5))
        ordinary_mean = stats.mean(b["ordinary"] for b in per_block)
        designed_mean = stats.mean(b["designed"] for b in per_block)
        lines.append(f"| {point['tokens']:,} | {scratch_mean:.5f} | {ordinary_mean:.5f} | "
                     f"{designed_mean:.5f} | {ordinary_mean-designed_mean:+.5f} |")
    lines += ["", "The selected-corpus ~2× speedup was exploratory because Top-10 was chosen on that corpus. This report tests the frozen policy on disjoint rule populations. Horizontal recurrence can still covary with other trajectory properties; a positive result validates a selection policy, not a unique mechanism. Crossings count downstream language tokens only.", "",
              f"[Corpus audit](../../runs/nca-data-search/top10-confirmation/audit.json) verifies {audit['selected_total']:,} distinct selected rule IDs and no prior selected-rule overlap.", ""]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
