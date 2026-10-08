"""Read only: summarize completed frozen scale-transfer runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "runs/scale-transfer-10m"
TARGET_TOKENS = 100_000_000


def read(stage: str) -> tuple[dict, list[dict]]:
    root = ROOT / stage
    return json.loads((root / "done.json").read_text()), json.loads((root / "metrics.json").read_text())["history"]


def cross(history: list[dict], target: float) -> float | None:
    if history[0]["validation_loss"] <= target:
        return 0.0
    for a, b in zip(history, history[1:]):
        if a["validation_loss"] > target >= b["validation_loss"]:
            return a["tokens"] + (target-a["validation_loss"]) * \
                (b["tokens"]-a["tokens"]) / (b["validation_loss"]-a["validation_loss"])
    return None


def auc(history: list[dict]) -> float:
    return sum((a["validation_loss"]+b["validation_loss"]) *
               (b["tokens"]-a["tokens"]) / 2
               for a,b in zip(history,history[1:])) / history[-1]["tokens"]


def energy_status() -> dict:
    path = ROOT / "telemetry.csv"
    if not path.is_file():
        return {"joules": None, "reason": "telemetry file absent"}
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    on_ac = sum(row["ac_connected"] == "Yes" for row in rows)
    # Battery discharge is a measurable energy proxy only for stages run wholly off AC.
    stage_battery_joules={}
    for stage in ("downstream_scratch", "downstream_ordinary", "downstream_designed"):
        samples=[r for r in rows if r["stage"] == stage]
        if len(samples) < 2 or any(r["ac_connected"] != "No" for r in samples):
            stage_battery_joules[stage]=None
            continue
        try:
            values=[(float(r["monotonic_seconds"]),
                     float(r["battery_mv"])*-float(r["battery_ma"])/1e6)
                    for r in samples]
            if any(power < 0 for _,power in values):
                stage_battery_joules[stage]=None
            else:
                stage_battery_joules[stage]=sum((b[0]-a[0])*(a[1]+b[1])/2
                                                  for a,b in zip(values,values[1:]))
        except (ValueError,KeyError):
            stage_battery_joules[stage]=None
    return {"joules": None, "energy_to_target_joules": None,
            "reason": "No privileged SoC power sensor. Battery discharge is only a partial proxy and on-AC current is not total system power.",
            "stage_battery_discharge_joules_when_off_ac": stage_battery_joules,
            "battery_samples": len(rows), "samples_on_ac": on_ac,
            "powermetrics_requires_root": True}


def main() -> None:
    runs={arm:read("downstream/"+arm) for arm in ("scratch","ordinary","designed")}
    nca={arm:read("nca/"+arm)[0] for arm in ("ordinary","designed")}
    target=runs["scratch"][1][-1]["validation_loss"]
    assert all(d["tokens_consumed"] == TARGET_TOKENS for d,h in runs.values())
    assert all(d["tokens_consumed"] == 197_153_250 for d in nca.values())
    results={}
    for arm,(done,h) in runs.items():
        t=cross(h,target)
        results[arm]={"final_validation_loss":h[-1]["validation_loss"],
                      "final_perplexity":h[-1]["perplexity"],
                      "normalized_0_100m_auc":auc(h),
                      "tokens_to_scratch_100m_target":t,
                      "downstream_token_speedup":TARGET_TOKENS/t if t else None,
                      "wall_seconds":done["runtime_seconds"],
                      "mean_training_tokens_per_second":TARGET_TOKENS/done["runtime_seconds"]}
    result={"schema":"sampleeff-10x-scale-transfer-summary-v1",
            "completed_at_utc":datetime.now(timezone.utc).isoformat(),
            "scratch_100m_target_loss":target,
            "energy":energy_status(),"downstream":results,
            "nca_stage":{k:{"scored_labels":v["tokens_consumed"],
                            "wall_seconds":v["runtime_seconds"]} for k,v in nca.items()},
            "limits":["one seed", "ten passes over each fixed 19,975-world NCA corpus",
                      "downstream token speedup excludes synthetic pretraining compute",
                      "sparse checkpoint interpolation"]}
    output=ROOT/"summary.json"
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# Frozen 10M-model scale-transfer result", "",
           f"Scratch@100M target validation loss: **{target:.5f}**.", "",
           "| Condition | 100M loss | PPL | 0–100M AUC | Tokens to target | Downstream speedup | Stage wall time |",
           "|:--|--:|--:|--:|--:|--:|--:|"]
    for arm,v in results.items():
        t=v["tokens_to_scratch_100m_target"]
        tokens_text=f"{t/1e6:.3f}M" if t is not None else "not reached"
        speed_text=f"{v['downstream_token_speedup']:.3f}x" if v['downstream_token_speedup'] is not None else "not reached"
        lines.append(f"| {arm} | {v['final_validation_loss']:.5f} | {v['final_perplexity']:.2f} | {v['normalized_0_100m_auc']:.5f} | {tokens_text} | {speed_text} | {v['wall_seconds']/3600:.2f}h |")
    lines += ["", "Each NCA arm processed 197,153,250 scored labels across ten passes over its frozen corpus.",
              "The comparison uses one model seed and counts downstream TinyStories tokens only.",
              "Energy-to-target: unavailable on AC without privileged SoC power telemetry; no value is imputed.", ""]
    (ROOT/"SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
