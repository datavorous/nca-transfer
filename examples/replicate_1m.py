"""Five-seed, six-condition NCA-to-TinyStories replication.

Run with `uv run --locked --all-packages python -u examples/replicate_1m.py`.
`--setup` prepares all immutable inputs and exits before training.
"""

from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch

import transfer_1m as v1
from sampleeff_nca_pretraining.packing import PackedDataset


ROOT = v1.REPO / "runs/1m-nca-transfer-v2"
V1 = v1.REPO / "runs/1m-nca-transfer-v1"
SEEDS = tuple(range(5))
CONDITIONS = ("scratch", "shuffled_nca", "nca_20_30", "nca_30_40",
              "nca_40_50", "nca_50_plus")
STRUCTURED = CONDITIONS[2:]
TEXT_MILESTONES = (0, 250_000, 500_000, 1_000_000, 2_000_000,
                   3_000_000, 5_000_000, 7_500_000, 10_000_000)
SHUFFLE_SOURCE = "40_50"
SHUFFLE_SALT = 0x5A17C0DE


def config() -> dict:
    v1.LANGUAGE_TOKENS = 10_000_000
    v1.LANGUAGE_MILESTONES = TEXT_MILESTONES
    previous = json.loads((V1 / "config/experiment.json").read_text())
    if (previous["nca_usable_training_tokens_per_band"] != 19_715_325 or
            previous["text_model"] != asdict(v1.TEXT_MODEL) or
            previous["nca_model"] != asdict(v1.NCA_MODEL)):
        raise ValueError("v1 experiment setup changed")
    result = v1.configuration()
    result.update({
        "name": "1m-nca-transfer-v2",
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "shuffled_source_band": SHUFFLE_SOURCE,
        "shuffle_salt": SHUFFLE_SALT,
        "shuffle_method": (
            "For each 1026-token trajectory, permute its 972 patch tokens across all "
            "27 frames with a NumPy RNG seeded from [experimental seed, shuffle salt]. "
            "Keep all 27 start/end grid markers at their original positions. "
            "This preserves each trajectory's token multiset, length, marker layout, "
            "rule split, and optimization budget while destroying patch order."
        ),
        "initialization_seed": None,
        "initialization_seeds": list(SEEDS),
        "language_tokens": 10_000_000,
        "language_milestones": list(TEXT_MILESTONES),
        "previous_experiment": str(V1),
        "previous_experiment_config_sha256": v1.digest(V1 / "config/experiment.json"),
        "previous_experiment_model_parameters": 1_081_177,
    })
    result["source_files_sha256"][str(Path(__file__).resolve().relative_to(v1.REPO))] = v1.digest(Path(__file__).resolve())
    return result


def seed_config(base: dict, seed: int) -> dict:
    return {**base, "experimental_seed": seed, "initialization_seed": seed}


def initialize(config_: dict) -> None:
    v1.ROOT = ROOT
    v1.init_root(config_)
    prior = ROOT / "config/v1_experiment.json"
    if prior.exists():
        if v1.digest(prior) != config_["previous_experiment_config_sha256"]:
            raise ValueError("copied v1 configuration differs")
    else:
        prior.parent.mkdir(parents=True, exist_ok=True)
        prior.write_bytes((V1 / "config/experiment.json").read_bytes())


def shuffle_packed(seed: int, config_: dict) -> Path:
    directory = ROOT / "data/shuffled_nca" / f"seed_{seed}"
    output = directory / "packed.pt"
    if v1.stage_done(directory, config_, (output,)):
        print(f"SKIP shuffled packed seed={seed}", flush=True)
        return output
    directory.mkdir(parents=True, exist_ok=True)
    original = PackedDataset.load(ROOT / "data/nca" / SHUFFLE_SOURCE / "packed.pt")
    if original.sequences.shape != (25_000, 1026):
        raise ValueError("unexpected source packed shape")
    sequences = original.sequences.clone()
    rng = np.random.default_rng(np.random.SeedSequence([seed, SHUFFLE_SALT]))
    started = time.monotonic()
    for row in range(len(sequences)):
        grids = sequences[row].view(27, 38)
        patches = grids[:, 1:37].reshape(-1).numpy().copy()
        rng.shuffle(patches)
        grids[:, 1:37] = torch.from_numpy(patches.reshape(27, 36))
        if row in (0, 12_499, 24_999):
            if not torch.equal(torch.sort(sequences[row]).values,
                               torch.sort(original.sequences[row]).values):
                raise ValueError(f"shuffle changed token multiset at row {row}")
            if not torch.equal(sequences[row].view(27, 38)[:, (0, 37)],
                               original.sequences[row].view(27, 38)[:, (0, 37)]):
                raise ValueError(f"shuffle changed grid markers at row {row}")
    targets = sequences.clone()
    targets[:, 0::38] = -100
    targets[:, 37::38] = -100
    metadata = {**original.metadata,
                "derived_from": str(ROOT / "data/nca" / SHUFFLE_SOURCE / "packed.pt"),
                "shuffle_seed": seed,
                "shuffle_salt": SHUFFLE_SALT,
                "shuffle_method": config_["shuffle_method"]}
    temporary = output.with_suffix(".pt.tmp")
    temporary.unlink(missing_ok=True)
    PackedDataset(sequences, targets, metadata).save(temporary)
    os.replace(temporary, output)
    v1.mark_done(directory, config_, sha256=v1.digest(output),
                 runtime_seconds=time.monotonic()-started)
    print(f"SHUFFLE DONE seed={seed} seconds={time.monotonic()-started:.1f}", flush=True)
    return output


def setup(config_: dict) -> None:
    initialize(config_)
    v1.ROOT = ROOT
    v1.prepare_tiny(config_)
    for band in v1.BANDS:
        v1.pack_band(band, config_)
    for seed in SEEDS:
        run_config = seed_config(config_, seed)
        v1.ROOT = ROOT / "seeds" / f"seed_{seed}"
        v1.init_root(run_config)
        v1.INITIALIZATION_SEED = seed
        v1.base_initialization(run_config)
        v1.ROOT = ROOT
        shuffle_packed(seed, run_config)
    print("SETUP COMPLETE", flush=True)


def train_seed(seed: int, base_config: dict) -> None:
    current = seed_config(base_config, seed)
    v1.ROOT = ROOT / "seeds" / f"seed_{seed}"
    v1.INITIALIZATION_SEED = seed
    v1.ORDER_SEED = 17  # same token-window order for all 30 downstream runs
    v1.TRAINING_SEED = 23
    base = v1.base_initialization(current)
    nca_checkpoints = {}
    for condition in CONDITIONS[1:]:
        if condition == "shuffled_nca":
            packed_path = ROOT / "data/shuffled_nca" / f"seed_{seed}" / "packed.pt"
        else:
            band = condition.removeprefix("nca_")
            packed_path = ROOT / "data/nca" / band / "packed.pt"
        packed = PackedDataset.load(packed_path)
        training = v1.NCAWindows(packed, "train", current["nca_train_worlds_per_band"])
        validation = v1.NCAWindows(packed, "validation", v1.EVAL_NCA_WORLDS)
        if int(training.valid.sum()) != current["nca_usable_training_tokens_per_band"]:
            raise ValueError(f"NCA budget differs: seed={seed} {condition}")
        nca_checkpoints[condition] = v1.train_stage(
            f"nca/{condition}", current, training, validation, v1.NCA_MODEL,
            v1.NCA_MILESTONES, v1.NCA_PHYSICAL_BATCH, base)
        del packed, training, validation
    language = v1.TextWindows(ROOT / "data/tinystories/train.bin", 10_000_000)
    validation = v1.TextWindows(ROOT / "data/tinystories/val.bin", v1.VALIDATION_TOKENS)
    for condition in CONDITIONS:
        v1.train_stage(f"downstream/{condition}", current, language, validation,
                       v1.TEXT_MODEL, TEXT_MILESTONES, v1.LANGUAGE_PHYSICAL_BATCH,
                       base, nca_checkpoints.get(condition))
    print(f"SEED COMPLETE {seed}", flush=True)


def aggregate_curves(stage: str, conditions: tuple[str, ...]):
    result = {}
    for condition in conditions:
        histories = []
        for seed in SEEDS:
            path = ROOT / "seeds" / f"seed_{seed}" / stage / condition / "metrics.json"
            rows = json.loads(path.read_text())["history"]
            histories.append(rows)
        reference = [row["tokens"] for row in histories[0]]
        if any([row["tokens"] for row in rows] != reference for rows in histories[1:]):
            raise ValueError(f"milestone token positions differ for {stage}/{condition}")
        result[condition] = {
            "tokens": reference,
            "loss_by_seed": {str(seed): [row["validation_loss"] for row in rows]
                             for seed, rows in zip(SEEDS, histories)},
            "perplexity_by_seed": {str(seed): [row["perplexity"] for row in rows]
                                   for seed, rows in zip(SEEDS, histories)},
        }
        array = np.asarray(list(result[condition]["loss_by_seed"].values()))
        result[condition]["mean_loss"] = array.mean(axis=0).tolist()
        result[condition]["std_loss"] = array.std(axis=0, ddof=1).tolist()
    return result


def crossing(tokens: list[int], values: list[float], threshold: float) -> float | None:
    if values[0] <= threshold:
        return float(tokens[0])
    for index in range(1, len(tokens)):
        a, b = values[index-1], values[index]
        if a > threshold >= b:
            if a == b:
                return float(tokens[index])
            return float(tokens[index-1] + (tokens[index]-tokens[index-1]) *
                         (a-threshold)/(a-b))
    return None


def interpolate(tokens: list[int], values: list[float], point: float) -> float:
    return float(np.interp(point, tokens, values))


def area(tokens: list[int], values: list[float], start=500_000, end=10_000_000) -> float:
    x = [start] + [point for point in tokens if start < point < end] + [end]
    y = [interpolate(tokens, values, point) for point in x]
    return float(np.trapezoid(y, x))


def summarize(config_: dict) -> None:
    v1.ROOT = ROOT
    directory = ROOT / "summary"
    outputs = (directory / "summary.json", directory / "milestones.csv",
               directory / "sample_efficiency.csv", directory / "README.md",
               ROOT / "plots/tinystories_validation_mean_std.png",
               ROOT / "plots/tinystories_validation_individual_seeds.png",
               ROOT / "plots/nca_validation_mean_std.png")
    if v1.stage_done(directory, config_, outputs):
        print("SKIP aggregate summary", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    downstream = aggregate_curves("downstream", CONDITIONS)
    nca = aggregate_curves("nca", STRUCTURED)
    scratch = downstream["scratch"]
    scratch_final = scratch["mean_loss"][-1]
    overlap_low = max(min(item["mean_loss"]) for item in downstream.values())
    overlap_high = min(max(item["mean_loss"]) for item in downstream.values())
    thresholds = [float(x) for x in np.linspace(overlap_low, overlap_high, 5)[1:-1]] if overlap_high > overlap_low else []
    threshold_rows = []
    for threshold_name, threshold in [("scratch_final_10m", scratch_final)] + [
            (f"common_{index+1}", value) for index, value in enumerate(thresholds)]:
        for condition in CONDITIONS:
            curve = downstream[condition]
            needed = crossing(curve["tokens"], curve["mean_loss"], threshold)
            threshold_rows.append({
                "threshold_name": threshold_name, "threshold_loss": threshold,
                "condition": condition, "tokens_to_threshold": needed,
                "token_savings_vs_10m": None if needed is None else 1-needed/10_000_000,
            })
    milestone_rows = []
    for condition in CONDITIONS:
        curve = downstream[condition]
        for index, target in enumerate(TEXT_MILESTONES):
            values = [curve["loss_by_seed"][str(seed)][index] for seed in SEEDS]
            milestone_rows.append({"condition": condition, "target_tokens": target,
                                   "actual_tokens": curve["tokens"][index],
                                   "mean_loss": curve["mean_loss"][index],
                                   "std_loss": curve["std_loss"][index],
                                   **{f"seed_{seed}_loss": values[seed] for seed in SEEDS}})
    paired = {}
    for condition in CONDITIONS[1:]:
        paired[condition] = {}
        for target in (5_000_000, 10_000_000):
            index = TEXT_MILESTONES.index(target)
            against_scratch = [downstream[condition]["loss_by_seed"][str(seed)][index] -
                               scratch["loss_by_seed"][str(seed)][index] for seed in SEEDS]
            paired[condition][str(target)] = {
                "versus_scratch_by_seed": against_scratch,
                "mean_difference": statistics.mean(against_scratch),
                "std_difference": statistics.stdev(against_scratch),
                "seeds_favor_condition": sum(value < 0 for value in against_scratch),
            }
    paired["nca_40_50"]["versus_shuffled_nca"] = {}
    for target in (5_000_000, 10_000_000):
        index = TEXT_MILESTONES.index(target)
        values = [downstream["nca_40_50"]["loss_by_seed"][str(seed)][index] -
                  downstream["shuffled_nca"]["loss_by_seed"][str(seed)][index]
                  for seed in SEEDS]
        paired["nca_40_50"]["versus_shuffled_nca"][str(target)] = {
            "by_seed": values, "mean_difference": statistics.mean(values),
            "std_difference": statistics.stdev(values),
            "seeds_favor_structured": sum(value < 0 for value in values),
        }
    auc = {}
    for condition, curve in downstream.items():
        individual = {str(seed): area(curve["tokens"], curve["loss_by_seed"][str(seed)])
                      for seed in SEEDS}
        auc[condition] = {"interval_tokens": [500_000, 10_000_000],
                          "by_seed": individual,
                          "mean": statistics.mean(individual.values()),
                          "std": statistics.stdev(individual.values())}
    ranks = {}
    for target in (1_000_000, 5_000_000, 10_000_000):
        index = TEXT_MILESTONES.index(target)
        sorted_regimes = sorted(STRUCTURED, key=lambda name: downstream[name]["mean_loss"][index])
        ranks[str(target)] = [
            {"regime": name, "mean_loss": downstream[name]["mean_loss"][index],
             "std_loss": downstream[name]["std_loss"][index],
             "rank": sorted_regimes.index(name)+1}
            for name in STRUCTURED]
    def make_plot(curves, names, filename, title, individual=False):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(9, 5.5))
        for name in names:
            item = curves[name]
            x = np.asarray(item["tokens"])
            mean = np.asarray(item["mean_loss"])
            std = np.asarray(item["std_loss"])
            if individual:
                for seed in SEEDS:
                    axis.plot(x, item["loss_by_seed"][str(seed)], alpha=0.16, linewidth=1)
            line, = axis.plot(x, mean, marker="o", label=name)
            if not individual:
                axis.fill_between(x, mean-std, mean+std, color=line.get_color(), alpha=0.15)
        axis.set(title=title, xlabel="Tokens consumed", ylabel="Validation cross-entropy loss")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(ROOT / "plots" / filename, dpi=150)
        plt.close(fig)
    make_plot(downstream, CONDITIONS, "tinystories_validation_mean_std.png",
              "TinyStories validation loss: mean ±1 SD across five seeds")
    make_plot(downstream, CONDITIONS, "tinystories_validation_individual_seeds.png",
              "TinyStories validation loss: individual seeds", individual=True)
    make_plot(nca, STRUCTURED, "nca_validation_mean_std.png",
              "Held-out-rule NCA validation loss: mean ±1 SD")
    nca_final = {name: {"mean": curve["mean_loss"][-1], "std": curve["std_loss"][-1],
                        "by_seed": {seed: values[-1] for seed, values in curve["loss_by_seed"].items()}}
                 for name, curve in nca.items()}
    attempts = {}
    runtimes = {}
    for seed in SEEDS:
        for stage in ("nca", "downstream"):
            names = CONDITIONS[1:] if stage == "nca" else CONDITIONS
            for name in names:
                path = ROOT / "seeds" / f"seed_{seed}" / stage / name
                key = f"seed_{seed}/{stage}/{name}"
                records = [json.loads(line) for line in (path / "attempts.jsonl").read_text().splitlines()]
                if len(records) > 1:
                    attempts[key] = records
                runtimes[key] = json.loads((path / "done.json").read_text())["runtime_seconds"]
    final_means = {name: downstream[name]["mean_loss"][-1] for name in CONDITIONS}
    structured_favor = {name: paired[name]["10000000"]["seeds_favor_condition"]
                        for name in STRUCTURED}
    vs_control = paired["nca_40_50"]["versus_shuffled_nca"]["10000000"]
    clear_structured = (vs_control["seeds_favor_structured"] >= 4 and
                        vs_control["mean_difference"] <= -0.05)
    consistent_positive = (sum(structured_favor[name] >= 4 for name in STRUCTURED) >= 3 and
                           sum(final_means[name] <= final_means["scratch"]-0.05
                               for name in STRUCTURED) >= 3)
    generic = (consistent_positive and
               paired["shuffled_nca"]["10000000"]["seeds_favor_condition"] >= 4 and
               not clear_structured)
    negative = all(final_means[name] > final_means["scratch"] for name in STRUCTURED)
    if clear_structured and consistent_positive:
        outcome = "structure-specific positive"
    elif generic:
        outcome = "generic pretraining effect"
    elif consistent_positive:
        outcome = "strong positive"
    elif negative:
        outcome = "negative transfer"
    else:
        outcome = "inconclusive"
    regime_gap = max(final_means[name] for name in STRUCTURED)-min(final_means[name] for name in STRUCTURED)
    seed_std = statistics.mean(downstream[name]["std_loss"][-1] for name in STRUCTURED)
    complexity_sensitive = regime_gap > seed_std and sorted(STRUCTURED, key=final_means.get)[0] == "nca_40_50"
    summary = {
        "one_seed_exploratory": False, "five_seed_paired_replication": True,
        "config": config_, "model_trainable_parameters": 1_081_177,
        "nca_trainable_parameters": 235_822,
        "downstream": downstream, "nca": nca,
        "scratch_final_10m_mean_loss": scratch_final,
        "sample_efficiency": threshold_rows,
        "auc_loss_vs_tokens": auc, "paired_effects": paired,
        "complexity_ranks": ranks, "nca_final": nca_final,
        "interpretation": {"outcome": outcome,
                           "complexity_sensitive_heuristic": complexity_sensitive,
                           "final_regime_mean_gap": regime_gap,
                           "mean_regime_seed_std": seed_std,
                           "criteria": ("A descriptive heuristic, not a significance test: "
                                        "consistent positive requires at least three regimes to "
                                        "beat scratch by >=0.05 mean loss and in >=4/5 paired seeds; "
                                        "structure-specific additionally requires 40_50 to beat "
                                        "shuffled_nca by >=0.05 mean and in >=4/5 seeds. "
                                        "Complexity sensitivity compares final mean regime gap "
                                        "with mean within-regime seed SD.")},
        "stage_runtimes_seconds": runtimes,
        "total_training_runtime_seconds": sum(runtimes.values()),
        "failed_or_retried_stages": attempts,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "torch": str(torch.__version__), "backend": "mps"},
        "git_commit": None,
        "source_files_sha256": config_["source_files_sha256"],
    }
    try:
        import subprocess
        summary["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=v1.REPO,
            stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        pass
    v1.atomic_json(directory / "summary.json", summary)
    with (directory / "milestones.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=milestone_rows[0].keys())
        writer.writeheader();writer.writerows(milestone_rows)
    with (directory / "sample_efficiency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=threshold_rows[0].keys())
        writer.writeheader();writer.writerows(threshold_rows)
    savings = next(row["token_savings_vs_10m"] for row in threshold_rows
                   if row["threshold_name"] == "scratch_final_10m" and row["condition"] == "nca_40_50")
    def yesno(value): return "yes" if value else "no"
    with (directory / "README.md").open("w") as stream:
        stream.write("# Five-seed paired NCA → TinyStories replication\n\n")
        stream.write("| Question | Result |\n| --- | --- |\n")
        stream.write(f"| Reliable improvement over scratch? | {yesno(consistent_positive)}; "
                     f"{sum(structured_favor[name]>=4 for name in STRUCTURED)}/4 regimes favor NCA in >=4/5 seeds |\n")
        stream.write(f"| Does `40_50` beat shuffled NCA? | {vs_control['seeds_favor_structured']}/5 paired seeds; "
                     f"mean loss difference {vs_control['mean_difference']:+.4f} at 10M |\n")
        stream.write(f"| `40_50` savings to scratch-final loss? | "
                     f"{'not reached' if savings is None else f'{savings:.1%}'} on mean curves |\n")
        stream.write(f"| Are effects consistent? | {outcome}; see paired seed values below |\n")
        stream.write(f"| Does complexity matter? | mean regime gap {regime_gap:.4f} vs mean seed SD {seed_std:.4f}; "
                     f"heuristic sensitivity: {yesno(complexity_sensitive)} |\n")
        stream.write(f"| Did `40_50` replicate as best? | rank {sorted(STRUCTURED,key=final_means.get).index('nca_40_50')+1}/4 "
                     f"at 10M mean loss |\n\n")
        stream.write("## Final 10M-token validation\n\n")
        stream.write("| Condition | Mean loss | SD | Individual seeds 0–4 |\n| --- | ---: | ---: | --- |\n")
        for name in CONDITIONS:
            curve=downstream[name]
            vals=[curve["loss_by_seed"][str(seed)][-1] for seed in SEEDS]
            stream.write(f"| {name} | {statistics.mean(vals):.4f} | {statistics.stdev(vals):.4f} | "
                         + ", ".join(f"{x:.4f}" for x in vals) + " |\n")
        stream.write("\n## Paired effects and sample efficiency\n\n")
        stream.write("Loss differences use the same seed on both sides; negative favors pretraining.\n\n")
        stream.write("| Comparison | 5M mean ± SD | 10M mean ± SD | Seeds favoring at 10M |\n"
                     "| --- | ---: | ---: | ---: |\n")
        for name in CONDITIONS[1:]:
            first, last = paired[name]["5000000"], paired[name]["10000000"]
            stream.write(f"| {name} − scratch | {first['mean_difference']:+.4f} ± {first['std_difference']:.4f} "
                         f"| {last['mean_difference']:+.4f} ± {last['std_difference']:.4f} "
                         f"| {last['seeds_favor_condition']}/5 |\n")
        first, last = (paired["nca_40_50"]["versus_shuffled_nca"][str(target)]
                       for target in (5_000_000, 10_000_000))
        stream.write(f"| nca_40_50 − shuffled_nca | {first['mean_difference']:+.4f} ± {first['std_difference']:.4f} "
                     f"| {last['mean_difference']:+.4f} ± {last['std_difference']:.4f} "
                     f"| {last['seeds_favor_structured']}/5 |\n\n")
        stream.write("| Condition | Tokens to scratch 10M loss | Token savings | AUC, 0.5M–10M |\n"
                     "| --- | ---: | ---: | ---: |\n")
        for name in CONDITIONS:
            row = next(row for row in threshold_rows
                       if row["threshold_name"] == "scratch_final_10m" and row["condition"] == name)
            needed = row["tokens_to_threshold"]
            saving = row["token_savings_vs_10m"]
            stream.write(f"| {name} | {'not reached' if needed is None else f'{needed:,.0f}'} "
                         f"| {'—' if saving is None else f'{saving:.1%}'} "
                         f"| {auc[name]['mean']:,.0f} ± {auc[name]['std']:,.0f} |\n")
        stream.write("\nTokens to threshold use linear interpolation between measured milestones. "
                     "AUC is lower when average validation loss is lower. "
                     "Other shared thresholds are in sample_efficiency.csv.\n\n")
        stream.write("## Complexity regimes\n\n")
        stream.write("| Regime | 1M loss ± SD (rank) | 5M loss ± SD (rank) | 10M loss ± SD (rank) |\n"
                     "| --- | ---: | ---: | ---: |\n")
        for name in STRUCTURED:
            cells = []
            for target in (1_000_000, 5_000_000, 10_000_000):
                row = next(row for row in ranks[str(target)] if row["regime"] == name)
                cells.append(f"{row['mean_loss']:.4f} ± {row['std_loss']:.4f} ({row['rank']}/4)")
            stream.write(f"| {name} | " + " | ".join(cells) + " |\n")
        stream.write("\n## Held-out-rule NCA validation\n\n")
        stream.write("| Regime | Final mean loss | SD |\n| --- | ---: | ---: |\n")
        for name in STRUCTURED:
            row = nca_final[name]
            stream.write(f"| {name} | {row['mean']:.4f} | {row['std']:.4f} |\n")
        stream.write("\n## Interpretation\n\n")
        stream.write(f"**{outcome.capitalize()}** under the stated descriptive criteria. "
                     "This is a five-seed exploratory replication, not a statistical proof. "
                     "The shuffled control preserves per-trajectory patch-token multisets and "
                     "grid-marker positions while scrambling patch order across frames. "
                     "See summary.json for exact paired differences, thresholds, AUC, NCA-side "
                     "results, runtimes, and retries.\n\n")
        stream.write("## Setup\n\n")
        stream.write("The model, tokenizer, optimizer, schedule, context, and batch settings "
                     "match v1. Each seed shares one random body and text head across all six "
                     "conditions; seeds 0–4 use independent initializations. Every condition "
                     "sees the same 10M TinyStories training targets in the same order and "
                     "the same validation targets. Each pre-pretraining condition consumes "
                     "19,715,325 usable NCA labels. Exact configuration and source hashes "
                     "are in summary.json.\n")
    v1.mark_done(directory, config_, summary_sha256=v1.digest(directory / "summary.json"))
    print("QUEUE COMPLETE", flush=True)


def main() -> None:
    setup_only = "--setup" in sys.argv[1:]
    if any(arg != "--setup" for arg in sys.argv[1:]):
        raise SystemExit("supported option: --setup")
    cfg = config()
    print(f"START v2 seed_matrix={list(SEEDS)} conditions={list(CONDITIONS)} "
          f"pid={os.getpid()} root={ROOT}", flush=True)
    setup(cfg)
    if setup_only:
        return
    for seed in SEEDS:
        train_seed(seed, cfg)
    summarize(cfg)


if __name__ == "__main__":
    main()
