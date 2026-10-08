"""Read only: close the frozen five-seed 10M-scale replication after all runs finish."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/"runs/scale-transfer-10m"
OUT=Path(__file__).with_name("REPLICATION_RESULTS.md")
SEEDS=range(5)
ARMS=("scratch","ordinary","designed")
TARGETS=(3.0,2.5,2.25,2.0,1.9)
RANGES=((0,1_000_000),(0,10_000_000),(0,100_000_000),(10_000_000,100_000_000))
COLORS={"scratch":"#A9D0F4","ordinary":"#EFA7D0","designed":"#F3D38A"}
LABELS={"scratch":"Scratch","ordinary":"Ordinary NCA","designed":"Designed NCA"}
BACKGROUND="#111722"
PANEL="#192230"
TEXT="#E8EDF4"
MUTED="#A9B5C4"
GRID="#344255"


def root(seed:int)->Path:
    return ROOT if seed==0 else ROOT/"seeds"/f"seed_{seed}"


def read(seed:int,arm:str)->list[dict]:
    stage=root(seed)/"downstream"/arm
    done=json.loads((stage/"done.json").read_text())
    assert done["tokens_consumed"]==100_000_000
    return json.loads((stage/"metrics.json").read_text())["history"]


def mean_sd(values:list[float])->str:
    return f"{st.mean(values):.5f} ± {st.stdev(values):.5f}"


def at(history:list[dict],tokens:float)->float:
    if tokens==history[0]["tokens"]:return history[0]["validation_loss"]
    for a,b in zip(history,history[1:]):
        if a["tokens"]<=tokens<=b["tokens"]:
            weight=(tokens-a["tokens"])/(b["tokens"]-a["tokens"])
            return a["validation_loss"]+weight*(b["validation_loss"]-a["validation_loss"])
    raise ValueError(f"{tokens} outside recorded curve")


def auc(history:list[dict],left:int,right:int)->float:
    points=[(left,at(history,left))]
    points.extend((row["tokens"],row["validation_loss"]) for row in history
                  if left<row["tokens"]<right)
    points.append((right,at(history,right)))
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(points,points[1:]))/(right-left)


def target_tokens(history:list[dict],target:float)->float|None:
    if history[0]["validation_loss"]<=target:return 0.0
    for a,b in zip(history,history[1:]):
        if a["validation_loss"]>target>=b["validation_loss"]:
            return a["tokens"]+(target-a["validation_loss"]) * \
                   (b["tokens"]-a["tokens"])/(b["validation_loss"]-a["validation_loss"])
    return None


def first_crossing(a:list[dict],b:list[dict])->tuple[float|None,str]:
    """First intersection of Designed minus Scratch, with its direction."""
    assert [r["tokens"] for r in a]==[r["tokens"] for r in b]
    differences=[x["validation_loss"]-y["validation_loss"] for x,y in zip(a,b)]
    if differences[0]==0:return 0.0,"equal at zero"
    for i,(u,v) in enumerate(zip(differences,differences[1:])):
        if u==0 or u*v<=0:
            x0=a[i]["tokens"]
            x1=a[i+1]["tokens"]
            t=x0+(0-u)*(x1-x0)/(v-u) if v!=u else x0
            return t,("Designed lower → Scratch lower" if u<0 else
                      "Scratch lower → Designed lower")
    return None,"no crossing through 100M"


def number(value:float|None,unit:str="")->str:
    return "not reached" if value is None else f"{value:.3f}{unit}"


def table(lines:list[str],headers:list[str],rows:list[list[object]])->None:
    lines.append("| "+" | ".join(headers)+" |")
    lines.append("|"+"|".join([":--"]+["--:" for _ in headers[1:]])+"|")
    lines.extend("| "+" | ".join(map(str,row))+" |" for row in rows)
    lines.append("")


def plot(histories:dict[int,dict[str,list[dict]]])->None:
    fig,ax=plt.subplots(figsize=(10.5,6.2),layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    inset=ax.inset_axes([.53,.43,.42,.48])
    for axis in (ax,inset):
        axis.set_facecolor(PANEL)
        axis.grid(color=GRID,linewidth=.8,alpha=.65)
        axis.spines[["top","right"]].set_visible(False)
        axis.spines[["bottom","left"]].set_color(GRID)
        axis.tick_params(colors=MUTED)
        for arm in ARMS:
            for seed in SEEDS:
                h=histories[seed][arm]
                axis.plot([r["tokens"]/1e6 for r in h],
                          [r["validation_loss"] for r in h],
                          color=COLORS[arm],linewidth=.65,alpha=.14)
            tokens=[r["tokens"]/1e6 for r in histories[0][arm]]
            mean=[st.mean(histories[seed][arm][i]["validation_loss"] for seed in SEEDS)
                  for i in range(len(tokens))]
            axis.plot(tokens,mean,color=COLORS[arm],linewidth=1.45,alpha=.82,label=LABELS[arm])
    ax.set_xlim(0,100)
    ax.set_xlabel("TinyStories tokens (M)",color=TEXT)
    ax.set_ylabel("Validation loss",color=TEXT)
    legend=ax.legend(frameon=False,loc="upper right",ncol=3)
    for label in legend.get_texts():label.set_color(TEXT)
    inset.set_xlim(0,10)
    inset.set_ylim(2.0,6.5)
    inset.tick_params(labelsize=8)
    folder=OUT.parent/"figures"
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/"scale_transfer_five_seeds.png",dpi=180,facecolor=BACKGROUND)
    fig.savefig(folder/"scale_transfer_five_seeds.svg",facecolor=BACKGROUND)
    plt.close(fig)


def main()->None:
    histories={seed:{arm:read(seed,arm) for arm in ARMS} for seed in SEEDS}
    expected=[r["tokens"] for r in histories[0]["scratch"]]
    assert all([r["tokens"] for r in histories[seed][arm]]==expected
               for seed in SEEDS for arm in ARMS)
    assert len(expected)==15 and expected[-1]==100_000_000
    plot(histories)
    lines=["# Frozen 10M-model scale-transfer replication: seeds 0–4","",
        "Five initializations; the same Ordinary and frozen Top-10 corpora, 197,153,250 scored NCA labels per pretrained arm, and the same 100M-token TinyStories stream per condition. Values are validation measurements. PPL is `exp(loss)`; lower is better. SD is the sample SD across five model seeds. Crossings use first downward linear interpolation of saved checkpoints, with no extrapolation.","",
        "![Individual seed trajectories and mean curves](figures/scale_transfer_five_seeds.png)","",
        "## Every downstream checkpoint by seed","",
        "Positive paired differences mean the first named condition has higher loss.",""]
    rows=[]
    for seed in SEEDS:
        for i,tokens in enumerate(expected):
            h=histories[seed]
            s,o,d=(h[arm][i] for arm in ARMS)
            rows.append([seed,f"{tokens:,}",f"{s['validation_loss']:.5f}",f"{s['perplexity']:.2f}",
                f"{o['validation_loss']:.5f}",f"{o['perplexity']:.2f}",
                f"{d['validation_loss']:.5f}",f"{d['perplexity']:.2f}",
                f"{d['validation_loss']-s['validation_loss']:+.5f}",
                f"{o['validation_loss']-s['validation_loss']:+.5f}",
                f"{d['validation_loss']-o['validation_loss']:+.5f}"])
    table(lines,["Seed","Tokens","Scratch loss","Scratch PPL","Ordinary loss","Ordinary PPL",
                 "Designed loss","Designed PPL","D−S loss","O−S loss","D−O loss"],rows)
    lines += ["## Mean ± sample SD at every checkpoint","",
              "PPL columns average the five individual perplexities; they are not `exp(mean loss)`.",""]
    rows=[]
    for i,tokens in enumerate(expected):
        h=[histories[seed] for seed in SEEDS]
        losses={arm:[x[arm][i]["validation_loss"] for x in h] for arm in ARMS}
        ppls={arm:[x[arm][i]["perplexity"] for x in h] for arm in ARMS}
        ds=[a-b for a,b in zip(losses["designed"],losses["scratch"])]
        os_=[a-b for a,b in zip(losses["ordinary"],losses["scratch"])]
        do=[a-b for a,b in zip(losses["designed"],losses["ordinary"])]
        rows.append([f"{tokens:,}",*[mean_sd(losses[a]) for a in ARMS],
                     *[mean_sd(ppls[a]) for a in ARMS],
                     mean_sd(ds),mean_sd(os_),mean_sd(do),
                     sum(x<0 for x in ds),sum(x<0 for x in os_),sum(x<0 for x in do)])
    table(lines,["Tokens","Scratch loss","Ordinary loss","Designed loss",
                 "Scratch PPL","Ordinary PPL","Designed PPL",
                 "D−S loss","O−S loss","D−O loss","D<S","O<S","D<O"],rows)
    lines += ["## First Designed–Scratch curve crossing","",
              "A crossing is a sign change of Designed minus Scratch loss between adjacent logged checkpoints; it is not an exact optimizer step.",""]
    rows=[]
    for seed in SEEDS:
        t,direction=first_crossing(histories[seed]["designed"],histories[seed]["scratch"])
        rows.append([seed,number(None if t is None else t/1e6,"M"),direction])
    table(lines,["Seed","First crossing tokens","Direction"],rows)
    lines += ["## Tokens to fixed target losses","",
              "The target grid **3.0, 2.5, 2.25, 2.0, 1.9** was frozen before seeds 1–4 ran. A ratio above 1 means the denominator condition needs fewer downstream tokens. No value is supplied where a target was not reached by 100M.",""]
    rows=[]
    ratios={target:{"S/D":[],"S/O":[],"O/D":[]} for target in TARGETS}
    for target in TARGETS:
        for seed in SEEDS:
            ts={arm:target_tokens(histories[seed][arm],target) for arm in ARMS}
            pair={"S/D":("scratch","designed"),"S/O":("scratch","ordinary"),
                  "O/D":("ordinary","designed")}
            result={name:(ts[a]/ts[b] if ts[a] is not None and ts[b] not in (None,0) else None)
                    for name,(a,b) in pair.items()}
            for name,value in result.items():
                if value is not None:ratios[target][name].append(value)
            rows.append([target,seed,*[number(None if ts[a] is None else ts[a]/1e6,"M") for a in ARMS],
                         *[number(result[k],"x") for k in ("S/D","S/O","O/D")]])
    table(lines,["Target loss","Seed","Scratch tokens","Ordinary tokens","Designed tokens",
                 "S/D token ratio","S/O token ratio","O/D token ratio"],rows)
    table(lines,["Target loss","S/D mean ± SD (n)","S/O mean ± SD (n)","O/D mean ± SD (n)"],
          [[target,*[(f"{st.mean(v):.3f} ± {st.stdev(v):.3f} (n={len(v)})" if len(v)>1 else
                       f"{v[0]:.3f} (n=1)" if v else "not available")
                     for v in (ratios[target]["S/D"],ratios[target]["S/O"],ratios[target]["O/D"])]]
           for target in TARGETS])
    lines += ["## Paired scratch-at-100M target","",
              "For each seed, the target is that seed's Scratch validation loss at 100M tokens. Ratios divide 100M by the first interpolated crossing time of the named NCA arm. This is a different target for each seed.",""]
    rows=[]
    paired_speedups={arm:[] for arm in ("ordinary","designed")}
    for seed in SEEDS:
        target=histories[seed]["scratch"][-1]["validation_loss"]
        crossings={arm:target_tokens(histories[seed][arm],target) for arm in ("ordinary","designed")}
        speedups={arm:(100_000_000/crossings[arm] if crossings[arm] not in (None,0) else None)
                  for arm in crossings}
        for arm,value in speedups.items():
            if value is not None:paired_speedups[arm].append(value)
        rows.append([seed,f"{target:.5f}",
                     *[number(None if crossings[a] is None else crossings[a]/1e6,"M")
                       for a in ("ordinary","designed")],
                     *[number(speedups[a],"x") for a in ("ordinary","designed")]])
    table(lines,["Seed","Scratch @100M loss","Ordinary crossing","Designed crossing",
                 "Ordinary speedup","Designed speedup"],rows)
    table(lines,["Condition","Mean ± sample SD","Seeds reached"],
          [[arm.title(),mean_sd(paired_speedups[arm]) if len(paired_speedups[arm])>1 else
            number(paired_speedups[arm][0],"x") if paired_speedups[arm] else "not reached",
            len(paired_speedups[arm])] for arm in ("ordinary","designed")])
    lines += ["## Normalized AUC over fixed downstream ranges","",
              "AUC uses trapezoidal integration, linearly interpolating only at a range boundary between saved checkpoints.",""]
    auc_rows=[]
    for left,right in RANGES:
        for seed in SEEDS:
            values={arm:auc(histories[seed][arm],left,right) for arm in ARMS}
            auc_rows.append([f"{left/1e6:g}–{right/1e6:g}M",seed,
                *[f"{values[a]:.5f}" for a in ARMS],
                f"{values['designed']-values['scratch']:+.5f}",
                f"{values['ordinary']-values['scratch']:+.5f}",
                f"{values['designed']-values['ordinary']:+.5f}"])
    table(lines,["Range","Seed","Scratch","Ordinary","Designed","D−S","O−S","D−O"],auc_rows)
    aggregate=[]
    for left,right in RANGES:
        all_values={arm:[auc(histories[seed][arm],left,right) for seed in SEEDS] for arm in ARMS}
        aggregate.append([f"{left/1e6:g}–{right/1e6:g}M",
            *[mean_sd(all_values[a]) for a in ARMS],
            *[mean_sd([x-y for x,y in zip(all_values[a],all_values[b])])
              for a,b in (("designed","scratch"),("ordinary","scratch"),("designed","ordinary"))]])
    table(lines,["Range","Scratch mean ± SD","Ordinary mean ± SD","Designed mean ± SD",
                 "D−S mean ± SD","O−S mean ± SD","D−O mean ± SD"],aggregate)
    lines += ["## Paired win counts at 100M","",]
    last={arm:[histories[seed][arm][-1]["validation_loss"] for seed in SEEDS] for arm in ARMS}
    table(lines,["Comparison","Winning seeds","Count"],[
        ["Designed < Ordinary",", ".join(str(s) for s in SEEDS if last["designed"][s]<last["ordinary"][s]),
         sum(a<b for a,b in zip(last["designed"],last["ordinary"]))],
        ["Designed < Scratch",", ".join(str(s) for s in SEEDS if last["designed"][s]<last["scratch"][s]),
         sum(a<b for a,b in zip(last["designed"],last["scratch"]))],
        ["Ordinary < Scratch",", ".join(str(s) for s in SEEDS if last["ordinary"][s]<last["scratch"][s]),
         sum(a<b for a,b in zip(last["ordinary"],last["scratch"]))],])
    lines += ["## NCA checkpoint receipts","",
              "Seed 0 has 30 saved checkpoints per NCA arm; its exact counts are in [the frozen audit](REPLICATION_PROTOCOL.md). Seeds 1–4 retain all periodic/evaluation checkpoints and add checkpoint-only saves at the first steps reaching 25M, 50M, 100M and 150M scored labels. The final checkpoint is at 197,153,250 labels. No dose branch was run.",""]
    for seed in range(1,5):
        for arm in ("ordinary","designed"):
            path=root(seed)/"nca"/arm/"checkpoint_labels.json"
            assert path.is_file()
            receipt=json.loads(path.read_text())
            assert receipt["final_scored_labels"]==197_153_250
            lines.append(f"- Seed {seed} {arm}: "+", ".join(
                f"{x['requested_labels']/1e6:g}M→{x['saved']['scored_labels']:,}"
                for x in receipt["checkpoint_only_targets"])+"; final 197,153,250.")
    lines.append("")
    lines += ["## Source and fixed scope","",
              "Seed-0 metrics are under `runs/scale-transfer-10m/downstream/`; seed-1–4 metrics are under `runs/scale-transfer-10m/seeds/seed_N/downstream/`. The [protocol](REPLICATION_PROTOCOL.md) records corpus and TinyStories hashes, architecture and optimizer settings, and the seed-0 checkpoint inventory. All five seeds use the same two NCA corpora and the same TinyStories stream; this is model-initialization replication, not fresh-corpus replication. Downstream token ratios exclude NCA training compute.",""]
    OUT.write_text("\n".join(lines))
    print(f"WROTE {OUT}",flush=True)


if __name__=="__main__":
    main()
