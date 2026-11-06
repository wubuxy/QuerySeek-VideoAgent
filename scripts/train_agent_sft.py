"""Real QLoRA via PEFT + Transformers Trainer; only assistant actions receive loss."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.gpu_agent_experiment import TransformersBackend,sha,within_outputs,write
from queryseek.schemas import VideoFrame


def load_model(cfg,training=False,adapter=None):
    import torch
    from transformers import AutoProcessor,BitsAndBytesConfig,Qwen3VLForConditionalGeneration
    torch.manual_seed(cfg["seed"]);torch.set_num_threads(8)
    model_path=ROOT/cfg["model_path"]
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    processor.image_processor.size={"shortest_edge":cfg["min_pixels"],"longest_edge":cfg["max_pixels"]}
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["visual","lm_head"])
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_path,
        local_files_only=True,dtype=torch.bfloat16,device_map={"":"cuda:0"},
        quantization_config=quant,attn_implementation=cfg["attention"])
    if training:
        from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant":False})
        # k-bit preparation upcasts nonquantized weights; visual tower stays frozen BF16.
        model.model.visual.to(dtype=torch.bfloat16)
        target=r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
        model=get_peft_model(model,LoraConfig(r=cfg["lora_rank"],lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],target_modules=target,bias="none",task_type="CAUSAL_LM"))
        model.config.use_cache=False
        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n or "language_model" not in n for n in trainable):
            raise ValueError("Unexpected trainable parameters")
    elif adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    return model,processor


class QuantizedBackend(TransformersBackend):
    def __init__(self,cfg,adapter=None):
        import torch
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        self.torch,self.cfg=torch,cfg
        self.model,self.processor=load_model(cfg,adapter=adapter)
        self.model.eval();self.model.config.use_cache=True
        self.tokenizer_data=build_token_enforcer_tokenizer_data(self.processor.tokenizer)
        self.calls=[];self.errors=0


def user_message(prompt,frames):
    content=[{"type":"text","text":prompt}]
    for i,f in enumerate(frames):
        content.extend([{"type":"text","text":f"anchor_index={i}; id={f.frame_id}; source_time_sec={f.timestamp_sec}"},
                        {"type":"image"}])
    return {"role":"user","content":content}


def mask_prefix_labels(input_ids,prompt_ids):
    if input_ids.shape[0]!=1 or prompt_ids.shape[0]!=1:
        raise ValueError("Only microbatch one is supported")
    n=prompt_ids.shape[-1]
    if n>=input_ids.shape[-1] or not input_ids[:,:n].equal(prompt_ids):
        raise ValueError("Assistant boundary does not match inference prompt")
    labels=input_ids.clone();labels[:,:n]=-100
    return labels


def supervised_positions(labels):
    import torch
    if labels.shape[0]!=1:
        raise ValueError("Only microbatch one is supported")
    positions=torch.nonzero(labels[0]!=-100,as_tuple=False).flatten()
    if not len(positions) or positions[0].item()==0:
        raise ValueError("Missing causal assistant target")
    return positions


class ActionCollator:
    def __init__(self,processor,manifest,package,cfg):
        self.processor,self.package,self.cfg=processor,Path(package),cfg
        self.samples={s["qid"]:s for s in manifest["samples"]}
    def __call__(self,features):
        from PIL import Image
        if len(features)!=1:
            raise ValueError("Expected microbatch one")
        row=features[0];sample=self.samples[row["qid"]]
        by_id={f["frame_id"]:f for f in sample["frames"]}
        frames=[VideoFrame(frame_id=i,timestamp_sec=by_id[i]["timestamp_sec"],
                           image_path=str(self.package/by_id[i]["image_path"])) for i in row["frame_ids"]]
        images=[]
        try:
            for f in frames:
                with Image.open(f.image_path) as im:images.append(im.convert("RGB"))
            user=user_message(row["prompt"],frames)
            prefix=self.processor.apply_chat_template([user],tokenize=False,add_generation_prompt=True)
            text=self.processor.apply_chat_template([user,{"role":"assistant","content":[
                {"type":"text","text":json.dumps(row["target"],separators=(",",":"))}]}],
                tokenize=False,add_generation_prompt=False)
            full=self.processor(text=[text],images=images,return_tensors="pt")
            before=self.processor(text=[prefix],images=images,return_tensors="pt")
            if full.input_ids.shape[-1]>self.cfg["max_sequence_tokens"]:
                raise ValueError("Oversized example; never silently truncate images or labels")
            full["labels"]=mask_prefix_labels(full.input_ids,before.input_ids)
            return dict(full)
        finally:
            for im in images:im.close()


def assistant_trainer_class():
    from transformers import Trainer
    import torch.nn.functional as F
    class AssistantTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            labels=inputs.pop("labels")
            positions=supervised_positions(labels)
            # Official Qwen logits_to_keep avoids materializing vocabulary logits for masked images/prompt.
            outputs=model(**inputs,logits_to_keep=positions-1,use_cache=False)
            loss=F.cross_entropy(outputs.logits.float().reshape(-1,outputs.logits.shape[-1]),
                                 labels[:,positions].reshape(-1))
            return (loss,outputs) if return_outputs else loss
    return AssistantTrainer


def adapter_hash(model):
    import torch
    from peft import get_peft_model_state_dict
    h=hashlib.sha256()
    for name,t in sorted(get_peft_model_state_dict(model).items()):
        h.update(name.encode());h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def train(package,out,smoke=False):
    import torch
    from transformers import TrainingArguments,TrainerCallback
    package=Path(package).resolve();out=within_outputs(out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((package/"config.json").read_text("utf8"))
    manifest=json.loads((package/"manifest.json").read_text("utf8"))
    rows=[json.loads(x) for x in (package/"train.jsonl").read_text("utf8").splitlines()]
    train_qids={s["qid"] for s in manifest["samples"] if s["stage"]=="train"}
    if any(r["qid"] not in train_qids for r in rows):
        raise ValueError("Nontraining sample in optimizer data")
    for name,digest in manifest["source_hashes"].items():
        if sha(ROOT/name)!=digest:raise ValueError("Source hash changed")
    if smoke:
        rows=rows[:8]
    started=time.perf_counter()
    write(out/"status.json",{"phase":"loading_model","smoke":smoke})
    model,processor=load_model(cfg,training=True)
    initial=adapter_hash(model)
    counts={"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "all_parameter_objects":sum(p.numel() for p in model.parameters())}
    write(out/"protocol.json",{"config":cfg,"manifest_sha256":sha(package/"manifest.json"),
        "train_jsonl_sha256":sha(package/"train.jsonl"),"examples_used":len(rows),"smoke":smoke,
        "initial_adapter_sha256":initial,"parameter_counts":counts,
        "source_sha256":{p:sha(ROOT/p) for p in ["scripts/train_agent_sft.py","queryseek/agent/sft_policy.py"]},
        "packages":{p:importlib.metadata.version(p) for p in [
            "torch","transformers","peft","bitsandbytes","accelerate"]},
        "loss":"ordinary causal cross entropy only on assistant target tokens; official logits_to_keep"})
    class Progress(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            logs=logs or {}
            row={"step":state.global_step,"epoch":state.epoch,**logs,
                 "elapsed_sec":time.perf_counter()-started,
                 "peak_allocated_bytes":torch.cuda.max_memory_allocated()}
            with (out/"training_log.jsonl").open("a",encoding="utf8") as handle:
                handle.write(json.dumps(row)+"\n")
            write(out/"status.json",{"phase":"training",**row})
        def on_step_end(self,args,state,control,**kwargs):
            if time.perf_counter()-started>28800:
                raise TimeoutError("Eight-hour per-run training limit")
    args=TrainingArguments(output_dir=str(out),num_train_epochs=cfg["epochs"],
        max_steps=12 if smoke else -1,per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if smoke else cfg["gradient_accumulation"],
        learning_rate=cfg["learning_rate"],weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],lr_scheduler_type="cosine",max_grad_norm=cfg["max_grad_norm"],
        bf16=True,fp16=False,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        optim="adamw_torch",logging_steps=1 if smoke else 5,
        save_strategy="no" if smoke else "epoch",save_total_limit=2,
        remove_unused_columns=False,dataloader_num_workers=0,
        report_to=[],seed=cfg["seed"],data_seed=cfg["seed"],disable_tqdm=True)
    trainer=assistant_trainer_class()(model=model,args=args,train_dataset=rows,
        data_collator=ActionCollator(processor,manifest,package,cfg),callbacks=[Progress()])
    trainer.model_accepts_loss_kwargs=False
    torch.cuda.reset_peak_memory_stats()
    result=trainer.train()
    trainer.save_model(str(out/"final_adapter"))
    processor.save_pretrained(str(out/"final_adapter"))
    final=adapter_hash(model)
    if final==initial:raise ValueError("No adapter weight update")
    write(out/"training_result.json",{"completed":True,"smoke":smoke,**result.metrics,
        "global_steps":trainer.state.global_step,"elapsed_sec":time.perf_counter()-started,
        "initial_adapter_sha256":initial,"final_adapter_sha256":final,
        "adapter_weights_changed":True,"peak_allocated_bytes":torch.cuda.max_memory_allocated()})
    write(out/"status.json",{"phase":"complete","smoke":smoke,"steps":trainer.state.global_step})
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--package",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args();print(train(a.package,a.out,a.smoke))
