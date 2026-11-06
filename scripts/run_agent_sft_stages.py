"""Sequential GPU worker. Confirmation requires an already frozen selection file."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.gpu_agent_experiment import sha,write,within_outputs


def checkpoint_key(metrics,epoch):
    groups=[v for k,v in metrics["groups"].items() if k.startswith("backtracking/")]
    n=sum(g["n"] for g in groups)
    if not n:raise ValueError("Missing development backtracking episodes")
    return (sum(g["n"]*g["r1_at_05"] for g in groups)/n,
            sum(g["n"]*g["top1_iou"] for g in groups)/n,-epoch)


def run(mode,package,out):
    package=Path(package).resolve();out=within_outputs(out);out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    def launch(name,script,args,timeout):
        target=out/name
        if target.exists():
            raise FileExistsError(f"Refuse to overwrite or silently resume {target}")
        write(out/"worker_status.json",{"phase":name,"pid":__import__("os").getpid(),"started":started})
        with (out/(name+".log")).open("w",encoding="utf8") as log:
            subprocess.run([sys.executable,"-u",str(ROOT/"scripts"/script),
                            "--package",str(package),"--out",str(target),*args],
                           cwd=ROOT,shell=False,timeout=timeout,check=True,stdout=log,stderr=subprocess.STDOUT)
    try:
        if mode=="train-dev":
            launch("train","train_agent_sft.py",[],30000)
            result=json.loads((out/"train"/"training_result.json").read_text("utf8"))
            if not result["completed"] or result["smoke"] or not result["adapter_weights_changed"]:
                raise ValueError("Training not completed")
            launch("dev_base","eval_agent_sft.py",["--stage","dev"],15000)
            checkpoints=sorted((out/"train").glob("checkpoint-*"),key=lambda p:int(p.name.split("-")[-1]))
            if len(checkpoints)!=2:raise ValueError("Expected both epoch checkpoints")
            candidates=[]
            for epoch,ckpt in enumerate(checkpoints,1):
                name=f"dev_epoch{epoch}"
                launch(name,"eval_agent_sft.py",["--stage","dev","--adapter",str(ckpt)],15000)
                metrics=json.loads((out/name/"metrics.json").read_text("utf8"))
                candidates.append({"epoch":epoch,"adapter":str(ckpt),
                    "adapter_sha256":sha(ckpt/"adapter_model.safetensors"),
                    "metrics_sha256":sha(out/name/"metrics.json"),
                    "selection_key":checkpoint_key(metrics,epoch)})
            selected=max(candidates,key=lambda x:x["selection_key"])
            write(out/"selection.json",{"rule":"dev backtracking R1@.5, then mean IoU, then earlier epoch",
                                       "candidates":candidates,"selected":selected,
                                       "confirmation_used_for_selection":False})
            write(out/"worker_status.json",{"phase":"dev_complete_pending_review","elapsed_sec":time.time()-started})
        else:
            selection=json.loads((out/"selection.json").read_text("utf8"))
            adapter=Path(selection["selected"]["adapter"])
            if sha(adapter/"adapter_model.safetensors")!=selection["selected"]["adapter_sha256"]:
                raise ValueError("Selected adapter changed")
            write(out/"confirmation_freeze.json",{"selection_sha256":sha(out/"selection.json"),
                  "selected_adapter_sha256":selection["selected"]["adapter_sha256"],
                  "started":started,"retuning_after_confirmation":False})
            launch("confirmation_base","eval_agent_sft.py",["--stage","confirmation"],15000)
            launch("confirmation_sft","eval_agent_sft.py",["--stage","confirmation","--adapter",str(adapter)],15000)
            write(out/"worker_status.json",{"phase":"confirmation_complete","elapsed_sec":time.time()-started})
    except Exception as e:
        write(out/"worker_error.json",{"type":type(e).__name__,"message":str(e),"elapsed_sec":time.time()-started})
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["train-dev","confirmation"],required=True)
    p.add_argument("--package",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.mode,a.package,a.out)
