"""Summarize fixed top-10 corpus at 10M across model seeds 0–4."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "runs/1m-nca-transfer-v2/seeds"
BOOSTED = REPO / "runs/nca-data-search/boosted-nca"
CORPUS = BOOSTED / "sweep-a/top_10"
TOKENS = (0, 251_904, 501_760, 1_001_472, 2_000_896, 3_000_320,
          5_001_088, 7_501_696, 10_000_000)


def history(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())["history"]
    assert tuple(row["tokens"] for row in rows) == TOKENS, path
    return rows


def crossing(rows: list[dict], target: float) -> float | None:
    if rows[0]["validation_loss"] <= target:
        return 0.0
    for a, b in zip(rows, rows[1:]):
        y0, y1 = a["validation_loss"], b["validation_loss"]
        if y0 > target >= y1:
            return a["tokens"] + (b["tokens"]-a["tokens"]) * (y0-target)/(y0-y1)
    return None


def auc(rows: list[dict]) -> float:
    return sum((b["tokens"]-a["tokens"]) *
               (a["validation_loss"]+b["validation_loss"])/2
               for a,b in zip(rows,rows[1:])) / TOKENS[-1]


def fmt_mean_sd(values: list[float], digits: int = 3) -> str:
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def main() -> None:
    decision = json.loads((BOOSTED / "decision.json").read_text())
    assert decision["selected_tier"] == "top_10"
    receipt = json.loads((CORPUS / "done.json").read_text())
    assert receipt["train_usable_tokens"] == 19_715_325
    rows = []
    for seed in range(5):
        reference = SOURCE / f"seed_{seed}/downstream"
        scratch = history(reference / "scratch/metrics.json")
        ordinary = history(reference / "nca_40_50/metrics.json")
        if seed == 0:
            run = BOOSTED / "final-10m"
        else:
            run = CORPUS / f"seeds/seed_{seed}/downstream/full_10m"
            nca_done = json.loads((CORPUS / f"seeds/seed_{seed}/nca/boosted_top_10/done.json").read_text())
            assert nca_done["tokens_consumed"] == 19_715_325
        done = json.loads((run / "done.json").read_text())
        assert done["tokens_consumed"] == 10_000_000
        designed = history(run / "metrics.json")
        assert designed[-1]["validation_loss"] == done["final_validation_loss"]
        assert [(r["step"], r["tokens"], r["learning_rate"]) for r in designed] == [
            (r["step"], r["tokens"], r["learning_rate"]) for r in ordinary]
        target = scratch[-1]["validation_loss"]
        t_ordinary, t_designed = crossing(ordinary,target), crossing(designed,target)
        rows.append({"seed":seed, "target":target,
                     "ordinary_loss":ordinary[-1]["validation_loss"],
                     "designed_loss":designed[-1]["validation_loss"],
                     "ordinary_tokens":t_ordinary,"designed_tokens":t_designed,
                     "ordinary_speedup":None if t_ordinary in (None,0.0) else 10_000_000/t_ordinary,
                     "designed_speedup":None if t_designed in (None,0.0) else 10_000_000/t_designed,
                     "ordinary_auc":auc(ordinary),"designed_auc":auc(designed)})

    lines = ["# Frozen top-10 corpus: model-seed validation", "",
             "Completed 2026-10-08. The top-10 corpus and operating point were selected using seed 0. "
             "Seeds 1–4 use the **same audited top-10 corpus** and each seed's original base initialization. "
             "All five models received 19,715,325 NCA labels and 10M TinyStories tokens under the original schedules. "
             "Only seed 0 participated in tier selection; seeds 1–4 are held-out **model initializations**, not fresh corpus blocks.", "",
             "For each seed, the fixed target is its own scratch validation loss at 10M. "
             "Crossing tokens use the first downward linear interpolation of the nine saved evaluation points. "
             "Speedup = 10M / crossing tokens; it counts downstream TinyStories tokens only.", "",
             "| Seed | Scratch target loss | Ordinary 10M loss | Top-10 10M loss | Ordinary crossing | Top-10 crossing | Ordinary speedup | Top-10 speedup | Top-10 − ordinary speedup |",
             "|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        fmt_t = lambda x: "unreached" if x is None else f"{x/1e6:.3f}M"
        fmt_s = lambda x: "NA" if x is None else f"{x:.3f}x"
        delta = None if r["ordinary_speedup"] is None or r["designed_speedup"] is None else r["designed_speedup"]-r["ordinary_speedup"]
        lines.append(f"| {r['seed']} | {r['target']:.5f} | {r['ordinary_loss']:.5f} | "
                     f"{r['designed_loss']:.5f} | {fmt_t(r['ordinary_tokens'])} | "
                     f"{fmt_t(r['designed_tokens'])} | {fmt_s(r['ordinary_speedup'])} | "
                     f"{fmt_s(r['designed_speedup'])} | {fmt_s(delta)} |")
    lines.extend(["", "## Mean ± sample SD across model seeds", "",
                  "| Seeds | Ordinary speedup | Top-10 speedup | Paired speedup difference | Top-10 faster count |",
                  "|:--|--:|--:|--:|--:|"])
    for label, subset in (("All 5 (seed 0 selected)",rows),("Held-out seeds 1–4",rows[1:])):
        ordinary = [r["ordinary_speedup"] for r in subset]
        designed = [r["designed_speedup"] for r in subset]
        if any(x is None for x in ordinary+designed):
            lines.append(f"| {label} | censored | censored | censored | "
                         f"{sum(d is not None and o is not None and d>o for o,d in zip(ordinary,designed))}/{len(subset)} |")
        else:
            differences = [d-o for o,d in zip(ordinary,designed)]
            lines.append(f"| {label} | {fmt_mean_sd(ordinary)}x | "
                         f"{fmt_mean_sd(designed)}x | {fmt_mean_sd(differences)}x | "
                         f"{sum(x>0 for x in differences)}/{len(subset)} |")
    heldout = rows[1:]
    if all(r["ordinary_speedup"] is not None and r["designed_speedup"] is not None for r in heldout):
        lines.extend(["", "The held-out mean is **" +
                      fmt_mean_sd([r["designed_speedup"] for r in heldout]) +
                      "x** for top-10 versus **" +
                      fmt_mean_sd([r["ordinary_speedup"] for r in heldout]) +
                      "x** for ordinary NCA. This supports an approximately 2x "
                      "downstream-token convergence result across these held-out model initializations "
                      "on the selected corpus, with no claim of fresh-corpus replication."])
    lines.extend(["", "## Endpoint and curve checks", "",
                  "| Seed | Ordinary 10M PPL | Top-10 10M PPL | Ordinary 0–10M AUC | Top-10 0–10M AUC | Ordinary − top-10 10M loss |",
                  "|--:|--:|--:|--:|--:|--:|"])
    for r in rows:
        lines.append(f"| {r['seed']} | {math.exp(r['ordinary_loss']):.2f} | "
                     f"{math.exp(r['designed_loss']):.2f} | {r['ordinary_auc']:.5f} | "
                     f"{r['designed_auc']:.5f} | {r['ordinary_loss']-r['designed_loss']:+.5f} |")
    lines.extend(["", "## Interpretation limits", "",
                  "The held-out comparison tests sensitivity to model initialization on one selected corpus. "
                  "It does not independently validate the top-10 selection across fresh NCA rule populations. "
                  "Seed 0 was used to select the tier, so the all-five-seed mean is descriptive and the seeds 1–4 row is the cleaner check. "
                  "Crossings are interpolated between sparse milestones, and NCA pretraining compute is excluded. "
                  "No unique horizontal-copy mechanism is identified; correlated transition MI, lag-2 recurrence, and other rule properties remain possible.", "",
                  "Sources: [frozen decision](../../runs/nca-data-search/boosted-nca/decision.json), "
                  "[corpus audit](../../runs/nca-data-search/boosted-nca/sweep-a/audit.json), "
                  "[seed-0 result](SWEEP_RESULTS.md), "
                  "[seed-1 run](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_1/downstream/full_10m/metrics.json), "
                  "[seed-2 run](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_2/downstream/full_10m/metrics.json), "
                  "[seed-3 run](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_3/downstream/full_10m/metrics.json), "
                  "[seed-4 run](../../runs/nca-data-search/boosted-nca/sweep-a/top_10/seeds/seed_4/downstream/full_10m/metrics.json).", ""])
    output = REPO / "progress/boosted_nca/SEED_VALIDATION.md"
    output.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
