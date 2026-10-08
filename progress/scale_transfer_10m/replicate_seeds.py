"""Run seeds 1–4 of the frozen 10M-scale comparison. No corpus generation or tuning."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from examples import transfer_1m as train
from progress.scale_transfer_10m import run as frozen
from sampleeff_nca_pretraining.model import create_model, trainable_parameters
from sampleeff_nca_pretraining.packing import PackedDataset
from sampleeff_nca_pretraining.training import TrainingConfig, load_checkpoint, prepare
from sampleeff_shared.models import create_attention_mask

ROOT = frozen.ROOT
SEEDS = (1, 2, 3, 4)
DOSE_TARGETS = (25_000_000, 50_000_000, 100_000_000, 150_000_000)
NCA_ARMS = (("ordinary", frozen.ORDINARY), ("designed", frozen.DESIGNED))
TEXT_ARMS = ("scratch", "ordinary", "designed")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def receipt_sha(path: Path) -> str:
    return frozen.sha(path)


def event(seed: int, stage: str, **fields) -> None:
    folder = ROOT / "replication"
    folder.mkdir(parents=True, exist_ok=True)
    row = {"utc": utc(), "seed": seed, "stage": stage, **fields}
    with (folder / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"SEED {seed} {stage} {fields}", flush=True)


def reference_config() -> dict:
    path = ROOT / "config.json"
    reference = json.loads(path.read_text())
    # Re-evaluate only input/code fingerprints; this does not write or retrain seed 0.
    frozen.configure_old()
    assert json.loads(json.dumps(frozen.frozen_config())) == reference, "seed-0 frozen inputs or code changed"
    text = json.loads((ROOT / "data/tinystories/metadata.json").read_text())
    assert text["tokens_saved"] == frozen.TEXT_TOKENS + 1
    assert receipt_sha(ROOT / "data/tinystories/train.bin") == text["sha256"]
    assert receipt_sha(frozen.SOURCE / "data/tinystories/val.bin") == reference["validation_sha256"]
    for stage in ("nca/ordinary", "nca/designed", "downstream/scratch",
                  "downstream/ordinary", "downstream/designed"):
        assert (ROOT / stage / "done.json").is_file(), f"seed-0 stage incomplete: {stage}"
    return reference


def seed_config(seed: int, reference: dict) -> dict:
    assert seed in SEEDS
    config = dict(reference)
    config["seed"] = seed
    config["reference_seed0_config_sha256"] = receipt_sha(ROOT / "config.json")
    config["replication_scope"] = "fresh model initialization; frozen corpora and text stream"
    config["nca_checkpoint_only_targets"] = DOSE_TARGETS
    path = ROOT / "seeds" / f"seed_{seed}" / "run_config.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(config))
    else:
        train.atomic_json(path, config)
    return config


def configure_seed(seed: int) -> Path:
    frozen.configure_old()
    root = ROOT / "seeds" / f"seed_{seed}"
    train.ROOT = root
    train.INITIALIZATION_SEED = seed
    assert train.ORDER_SEED == 17 and train.TRAINING_SEED == 23
    assert train.NCA_MODEL == frozen.NCA_MODEL and train.TEXT_MODEL == frozen.TEXT_MODEL
    return root


def initialization(seed: int, root: Path, *, record: bool = True) -> Path:
    output = root / "config/base_random_init.pth"
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        assert payload["model_config"] == asdict(frozen.TEXT_MODEL)
        assert payload["initialization_seed"] == seed
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    model = create_model(frozen.TEXT_MODEL, seed)
    assert trainable_parameters(model) == 9_943_921
    tmp = output.with_suffix(".pth.tmp")
    torch.save({"model": model.state_dict(), "model_config": asdict(frozen.TEXT_MODEL),
                "initialization_seed": seed}, tmp)
    os.replace(tmp, output)
    if record:
        event(seed, "base_initialization", sha256=receipt_sha(output))
    return output


def checkpoint_labels(seed: int, directory: Path, prefix: np.ndarray, size: int) -> None:
    """Receipt of every existing checkpoint, including extra dose saves."""
    rows=[]
    for path in sorted((directory / "checkpoints").glob("model_*_iter.pth"),
                       key=lambda p:int(p.stem.split("_")[1])):
        step=int(path.stem.split("_")[1])
        tokens=int(prefix[min(step*train.EFFECTIVE_BATCH,size)])
        rows.append({"step":step,"scored_labels":tokens,
                     "checkpoint":str(path.relative_to(directory))})
    if not rows:
        return
    targets=[]
    for target in DOSE_TARGETS:
        first=next((r for r in rows if r["scored_labels"] >= target),None)
        if first is not None:
            first_step=math.ceil(int(np.searchsorted(prefix,target,side="left"))/train.EFFECTIVE_BATCH)
            assert first["step"]==first_step, f"dose checkpoint missed first crossing of {target} labels"
        targets.append({"requested_labels":target, "saved":first})
    train.atomic_json(directory / "checkpoint_labels.json",{
        "schema":"nca-dose-checkpoints-v1", "seed":seed,
        "evaluation_milestones_unchanged":list(frozen.NCA_MILESTONES),
        "checkpoint_only_targets":targets,
        "final_scored_labels":frozen.NCA_LABELS,
        "checkpoints":rows})


def train_nca_stage(seed: int, name: str, config: dict, dataset, validation,
                    initial_checkpoint: Path) -> Path:
    """The frozen update/eval loop, with checkpoint-only saves at fixed label counts."""
    directory = train.ROOT / name
    checkpoint_dir = directory / "checkpoints"
    order,prefix = train.ordered(dataset,train.ORDER_SEED)
    budget = int(prefix[-1])
    steps = math.ceil(dataset.size/train.EFFECTIVE_BATCH)
    assert budget == frozen.NCA_LABELS and steps == 99_875
    done_path = directory / "done.json"
    if train.stage_done(directory,config,(done_path,)):
        done=json.loads(done_path.read_text())
        final=directory/done["final_checkpoint"]
        assert final.is_file() and done["tokens_consumed"] == budget
        checkpoint_labels(seed,directory,prefix,dataset.size)
        print(f"SKIP {name} seed={seed} complete",flush=True)
        return final
    directory.mkdir(parents=True,exist_ok=True)
    train_config=TrainingConfig(steps=steps,batch_size=train.EFFECTIVE_BATCH,
        learning_rate=train.LR,weight_decay=0.0,warmup_steps=train.WARMUP_STEPS,
        device="mps",seed=train.TRAINING_SEED,eval_interval=train.CHECKPOINT_STEPS)
    prior=train.latest_checkpoint(checkpoint_dir) if checkpoint_dir.exists() else None
    if prior:
        state=load_checkpoint(prior)
        assert state.model_config == frozen.NCA_MODEL and state.config == train_config
        print(f"RESUME {name} seed={seed} step={state.step}",flush=True)
    else:
        model=create_model(frozen.NCA_MODEL,seed)
        train.load_pretrained_body(model,initial_checkpoint)
        state=prepare(model,frozen.NCA_MODEL,train_config,
            dataset_metadata={"stage":name,"config":config},model_seed=seed)
    with (directory/"attempts.jsonl").open("a") as stream:
        stream.write(json.dumps({"started_at_utc":utc(),"pid":os.getpid(),
                                 "resumed_from_step":state.step})+"\n")
    mask=create_attention_mask(frozen.NCA_MODEL.context_length,additive=True).to(state.device)
    stage_started=time.monotonic()
    interval_started=stage_started
    previous_runtime=0.0
    progress_path=directory/"progress.json"
    if prior and progress_path.is_file():
        progress=json.loads(progress_path.read_text())
        if progress["step"] == state.step:
            previous_runtime=progress.get("runtime_seconds",0.0)
    previous_tokens=int(prefix[min(state.step*train.EFFECTIVE_BATCH,dataset.size)])
    next_eval=next((v for v in frozen.NCA_MILESTONES if v>previous_tokens),None)
    next_dose=next((v for v in DOSE_TARGETS if v>previous_tokens),None)
    if state.step == 0 and not state.history:
        value=train.evaluate_model(state.model,validation,train.NCA_PHYSICAL_BATCH,mask)
        state.history.append({"step":0,"tokens":0,"train_loss":None,
            "validation_loss":value,"perplexity":math.exp(value),
            "learning_rate":state.optimizer.param_groups[0]["lr"],
            "tokens_per_second":None,"elapsed_seconds":0.0})
        train.record_metrics(directory,state)
        train.checkpoint(state,checkpoint_dir)
        print(f"EVAL {name} tokens=0 loss={value:.5f}",flush=True)
    loss_sum=0.0
    loss_steps=0
    for step in range(state.step+1,steps+1):
        positions=order[(step-1)*train.EFFECTIVE_BATCH:min(step*train.EFFECTIVE_BATCH,dataset.size)]
        loss=train.update_step(state,dataset,positions,train.NCA_PHYSICAL_BATCH,mask)
        state.step=step
        loss_sum+=loss
        loss_steps+=1
        consumed=int(prefix[min(step*train.EFFECTIVE_BATCH,dataset.size)])
        need_eval=(next_eval is not None and consumed>=next_eval) or step==steps
        dose_save=next_dose is not None and consumed>=next_dose
        need_save=need_eval or dose_save or step%train.CHECKPOINT_STEPS==0
        if need_eval:
            val_loss=train.evaluate_model(state.model,validation,train.NCA_PHYSICAL_BATCH,mask)
            if not math.isfinite(val_loss):
                raise FloatingPointError(f"non-finite validation loss: {name}")
            elapsed=time.monotonic()-interval_started
            row={"step":step,"tokens":consumed,
                "train_loss":loss_sum/max(loss_steps,1),"validation_loss":val_loss,
                "perplexity":math.exp(val_loss),
                "learning_rate":state.optimizer.param_groups[0]["lr"],
                "tokens_per_second":(consumed-previous_tokens)/max(elapsed,1e-9),
                "elapsed_seconds":previous_runtime+time.monotonic()-stage_started}
            state.history.append(row)
            print(f"EVAL {name} step={step} tokens={consumed} train={row['train_loss']:.5f} "
                  f"val={val_loss:.5f} tokens_per_sec={row['tokens_per_second']:.0f}",flush=True)
            previous_tokens=consumed
            interval_started=time.monotonic()
            loss_sum=0.0
            loss_steps=0
            next_eval=next((v for v in frozen.NCA_MILESTONES if v>consumed),None)
        elif step%100==0:
            print(f"PROGRESS {name} step={step}/{steps} tokens={consumed} "
                  f"train_loss={loss:.5f} lr={state.optimizer.param_groups[0]['lr']:.8f}",flush=True)
        if dose_save:
            print(f"DOSE CHECKPOINT {name} requested={next_dose} scored_labels={consumed} step={step}",flush=True)
            next_dose=next((v for v in DOSE_TARGETS if v>consumed),None)
        if need_save:
            final=train.checkpoint(state,checkpoint_dir)
            train.record_metrics(directory,state)
            train.atomic_json(progress_path,{"step":step,"tokens":consumed,
                "checkpoint":str(final.relative_to(directory)),"updated_at_utc":utc(),
                "runtime_seconds":previous_runtime+time.monotonic()-stage_started,
                "runtime_semantics":"cumulative"})
            if dose_save:
                checkpoint_labels(seed,directory,prefix,dataset.size)
    final=train.latest_checkpoint(checkpoint_dir)
    assert final is not None
    train.mark_done(directory,config,final_checkpoint=str(final.relative_to(directory)),
        tokens_consumed=budget,final_validation_loss=state.history[-1]["validation_loss"],
        final_perplexity=state.history[-1]["perplexity"],completed_steps=steps,
        runtime_seconds=previous_runtime+time.monotonic()-stage_started)
    checkpoint_labels(seed,directory,prefix,dataset.size)
    print(f"DONE {name} seed={seed} tokens={budget} checkpoint={final}",flush=True)
    return final


def run_seed(seed: int, reference: dict) -> None:
    root=configure_seed(seed)
    config=seed_config(seed,reference)
    base=initialization(seed,root)
    nca_checkpoints={}
    for arm,path in NCA_ARMS:
        event(seed,f"nca_{arm}_start")
        packed=PackedDataset.load(path)
        train_data=frozen.RepeatedNCA(train.NCAWindows(packed,"train",19_975),frozen.PASSES)
        val_data=train.NCAWindows(packed,"validation",256)
        assert int(train_data.valid.sum())==frozen.NCA_LABELS
        nca_checkpoints[arm]=train_nca_stage(seed,"nca/"+arm,config,train_data,val_data,base)
        event(seed,f"nca_{arm}_done")
        del packed,train_data,val_data
        torch.mps.empty_cache()
    text=train.TextWindows(ROOT/"data/tinystories/train.bin",frozen.TEXT_TOKENS)
    val=train.TextWindows(frozen.SOURCE/"data/tinystories/val.bin",frozen.VAL_TOKENS)
    for arm in TEXT_ARMS:
        event(seed,f"downstream_{arm}_start")
        train.train_stage("downstream/"+arm,config,text,val,frozen.TEXT_MODEL,
                          frozen.TEXT_MILESTONES,train.LANGUAGE_PHYSICAL_BATCH,
                          base,nca_checkpoints.get(arm))
        event(seed,f"downstream_{arm}_done")
        torch.mps.empty_cache()
    event(seed,"complete")


def smoke(reference: dict) -> None:
    """Two seed-1 optimizer steps through the new NCA loop, in a temporary root."""
    assert torch.backends.mps.is_available()
    with tempfile.TemporaryDirectory(prefix="sampleeff-scale-replication-smoke-") as tmp:
        root=configure_seed(1)
        train.ROOT=Path(tmp)
        config=dict(reference)
        config["seed"]=1
        base=initialization(1,Path(tmp),record=False)
        packed=PackedDataset.load(frozen.ORDINARY)
        source=train.NCAWindows(packed,"train",19_975)
        class Tiny:
            size=16
            valid=source.valid[:16]
            def batch(self,positions):
                return source.batch(positions)
        # Validate only that the frozen model, pairing and update path can run.
        model=create_model(frozen.NCA_MODEL,1)
        train.load_pretrained_body(model,base)
        state=prepare(model,frozen.NCA_MODEL,TrainingConfig(steps=2,batch_size=8,
            learning_rate=train.LR,warmup_steps=0,device="mps",seed=23))
        mask=create_attention_mask(256,additive=True).to("mps")
        for step in (1,2):
            train.update_step(state,Tiny(),np.arange((step-1)*8,step*8),4,mask)
        assert math.isfinite(train.evaluate_model(state.model,train.NCAWindows(packed,"validation",256),
                                                   4,mask,limit=4))
        print("PASS seed-1 NCA update/transfer smoke",flush=True)
        train.ROOT=root


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",action="store_true")
    group.add_argument("--seed",type=int,choices=SEEDS)
    group.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    reference=reference_config()
    if args.smoke:
        smoke(reference)
        return
    for seed in SEEDS if args.all else (args.seed,):
        try:
            run_seed(seed,reference)
        except BaseException as exc:
            event(seed,"failed",error=repr(exc))
            raise
    if args.all:
        subprocess.run([sys.executable,str(Path(__file__).with_name("summarize_replication.py"))],
                       cwd=REPO,check=True)


if __name__ == "__main__":
    main()
