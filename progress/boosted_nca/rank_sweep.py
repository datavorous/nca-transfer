"""Read-only ranking of completed seed-0 horizontal dose probes."""

from __future__ import annotations

import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "runs/nca-data-search"
HISTORIES = {
    "top_50_anchor": BASE / "patch-replication/block-a/high/seeds/seed_0/downstream/probe_1m/metrics.json",
    "top_25": BASE / "boosted-nca/sweep-a/top_25/seeds/seed_0/downstream/probe_1m/metrics.json",
    "top_10": BASE / "boosted-nca/sweep-a/top_10/seeds/seed_0/downstream/probe_1m/metrics.json",
    "top_5": BASE / "boosted-nca/sweep-a/top_5/seeds/seed_0/downstream/probe_1m/metrics.json",
}
TARGETS = (7.0, 6.5, 6.25)
TOKENS = (0, 251_904, 501_760, 1_001_472)
STEPS = (0, 123, 245, 489)


def crossing(history: list[dict], target: float) -> float | None:
    if history[0]["validation_loss"] <= target:
        return 0.0
    for first, second in zip(history, history[1:]):
        y0, y1 = first["validation_loss"], second["validation_loss"]
        if y0 > target >= y1:
            return first["tokens"] + (second["tokens"]-first["tokens"]) * (y0-target)/(y0-y1)
    return None


def load(name: str) -> dict:
    path = HISTORIES[name]
    history = json.loads(path.read_text())["history"]
    assert [(row["step"], row["tokens"]) for row in history] == list(zip(STEPS, TOKENS)), name
    if name != "top_50_anchor":
        done = json.loads((path.parent / "done.json").read_text())
        assert done["tokens_consumed"] == TOKENS[-1]
        assert done["final_validation_loss"] == history[-1]["validation_loss"]
    crosses = [crossing(history, target) for target in TARGETS]
    reached = [x for x in crosses if x is not None]
    score = math.exp(sum(math.log(x) for x in reached)/len(reached)) if len(reached) == 3 else None
    auc = sum((b["tokens"]-a["tokens"]) *
              (a["validation_loss"]+b["validation_loss"]) / 2
              for a,b in zip(history, history[1:])) / TOKENS[-1]
    return {"name": name, "crossings": crosses, "score": score,
            "unreached": 3-len(reached), "loss_1m": history[-1]["validation_loss"],
            "auc_0_1m": auc, "history": history}


def choose(results: list[dict]) -> tuple[bool, str | None]:
    by_name = {result["name"]: result for result in results}
    top50, top25, top10 = (by_name[name] for name in ("top_50_anchor", "top_25", "top_10"))
    top5_gate = all(a is not None and b is not None and c is not None
                    and c <= .98*b and b <= .98*a
                    for a,b,c in zip(top50["crossings"], top25["crossings"], top10["crossings"]))
    if top5_gate and "top_5" not in by_name:
        return True, None
    ranked = sorted(results, key=lambda r: (r["unreached"], r["score"] if r["score"] is not None else math.inf))
    best = ranked[0]
    if best["score"] is not None:
        near = [r for r in ranked if r["score"] is not None and r["score"] <= 1.01*best["score"]]
        best = min(near, key=lambda r: ("top_50_anchor", "top_25", "top_10", "top_5").index(r["name"]))
    return top5_gate, best["name"]


def main() -> None:
    present = [name for name in HISTORIES if HISTORIES[name].is_file()]
    assert all(name in present for name in ("top_50_anchor", "top_25", "top_10")), "1M sweep incomplete"
    results = [load(name) for name in present]
    reference_lr = [row["learning_rate"] for row in results[0]["history"]]
    assert all([row["learning_rate"] for row in result["history"]] == reference_lr
               for result in results[1:])
    print("| Tier | Loss 7 tokens | Loss 6.5 tokens | Loss 6.25 tokens | Geometric mean tokens | 1M loss | 0–1M AUC |")
    print("|:--|--:|--:|--:|--:|--:|--:|")
    for result in results:
        fmt = lambda x: "NA" if x is None else f"{x/1000:.1f}k"
        print(f"| {result['name']} | " + " | ".join(fmt(x) for x in result["crossings"]) +
              f" | {fmt(result['score'])} | {result['loss_1m']:.5f} | "
              f"{result['auc_0_1m']:.5f} |")
    top5_gate, winner = choose(results)
    print(f"\nTop-5 gate (each target: 25 at least 2% faster than 50, 10 at least 2% faster than 25): "
          f"{'PASS' if top5_gate else 'FAIL'}")
    if winner is None:
        print("Decision pending one predeclared top-5 probe.")
        return
    print(f"Winner by the frozen metric and 1% conservative tie-break: {winner}")
    print("One stream and one initialization: ranking is exploratory; no significance claim.")


if __name__ == "__main__":
    main()
