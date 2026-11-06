"""Export scientific figures from already computed results; never recompute accuracy."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"outputs/gpu_agent_20261021/plot_dependencies"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(run,out):
    run,out=Path(run),Path(out)
    out.resolve().relative_to((ROOT/"outputs").resolve())
    out.mkdir(parents=True,exist_ok=True)
    result=json.loads((run/"comparison.json").read_text("utf8"))
    logs=[json.loads(x) for x in (run/"train/training_log.jsonl").read_text("utf8").splitlines()]
    logs=[r for r in logs if "loss" in r]
    methods=["fixed_uniform","forward_only","backtracking"]
    names=["Fixed 16 frames","Forward only","Backtracking"]
    colors=["#66788A","#147D92"]
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for i,label in enumerate(["base","sft"]):
        values=[100*result["groups"][label][m]["r1_at_05"] for m in methods]
        bars=axes[0].bar([j+(i-.5)*.36 for j in range(3)],values,width=.36,
                        color=colors[i],label="NF4 base" if i==0 else "QLoRA SFT")
        axes[0].bar_label(bars,fmt="%.1f",padding=3,fontsize=9)
    axes[0].set(xticks=range(3),xticklabels=names,ylabel="R1@0.5 (%)",
                title="128 held-out sources (internal split)")
    axes[0].legend(frameon=False,fontsize=9)
    for j,m in enumerate(methods):
        d=result["sft_minus_base_source_average"][m]["r1_at_05"]
        mean=100*d["mean"];lo,hi=[100*v for v in d["ci95"]]
        axes[1].errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt="o",color=colors[1],capsize=4)
    axes[1].axvline(0,color="#909090",linestyle="--",linewidth=1)
    axes[1].set(yticks=range(3),yticklabels=names,xlabel="SFT minus base (percentage points)",
                title="Paired source bootstrap, 95% CI")
    axes[1].invert_yaxis()
    axes[2].plot([r["step"] for r in logs],[r["loss"] for r in logs],color=colors[1],linewidth=1.3)
    axes[2].set(xlabel="Optimizer step",ylabel="Assistant-token training loss",
                title="Training diagnostics")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
    fig.text(.5,.01,"Same NF4 base, prompt, source-PTS frames and budgets. Agent: two starts averaged per source; 5,000 bootstrap replicates.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ["png","pdf"]:fig.savefig(out/f"sft_results.{ext}",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args();plot(a.run,a.out)
