"""Freeze images, audit visual transport, and isolate resolution/grammar/precision."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from queryseek.eval.model_diagnostic import (conditions, minimal_prompt, parse_answer,
    donor_indices, assert_visual_transport, summarize)
from queryseek.retrieval import evidence_needs as needs
from queryseek.schemas import VideoFrame, Scene
from scripts import probe_query_reader as previous
from scripts import grounded_qa_input_study as study

read, write, sha = previous.read, study.write, study.sha256_file
append, digest = previous.append, previous.digest


def sources():
    return {**previous.sources(), **{n: sha(ROOT/n) for n in [
        'queryseek/eval/model_diagnostic.py','scripts/probe_model_diagnostic.py',
        'queryseek/retrieval/evidence_needs.py','queryseek/retrieval/evidence_packets.py',
        'scripts/make_synthetic_fixture.py']}}


def begin(path, config_path, mock):
    path = study.baseline.checked_output(path)
    path.mkdir(parents=True, exist_ok=False)
    write(path/'manifest.initial.json', {'config_sha256':sha(config_path),
        'source_sha256':sources(), 'mock':mock})
    return path


def finish(path, names, summary):
    write(path/'summary.json', summary)
    write(path/'manifest.json', {**read(path/'manifest.initial.json'),
        'files':{n:sha(path/n) for n in [*names,'summary.json']}})
    return summary


def verify(path, config_path, mock):
    m = read(path/'manifest.json')
    if (m['config_sha256'] != sha(config_path) or m['source_sha256'] != sources()
            or m['mock'] != mock):
        raise ValueError('Provenance/mock mismatch')
    for n,h in m['files'].items():
        if sha(path/n) != h: raise ValueError('Artifact changed')
    return m


def checked(path, expected):
    if sha(path) != expected: raise ValueError('Frozen input changed')
    return read(path)


def normalize_frames(frames):
    return [{**f,'frame_id':f'view_{i:03d}'} for i,f in enumerate(frames)]


def validate_row(row, count):
    previous.check_requests([row['request']])
    frames = [VideoFrame.model_validate(f) for f in row['frames']]
    times = [f.timestamp_sec for f in frames]
    if (len(frames)!=count or len(row['frame_sha256'])!=count
        or any(not 0 <= t < row['request']['duration_sec'] for t in times)
        or any(a >= b for a,b in zip(times,times[1:]))):
        raise ValueError('Invalid source time/count')
    for f,h in zip(frames,row['frame_sha256'],strict=True):
        study.baseline.checked_output(f.image_path)
        if sha(f.image_path)!=h: raise ValueError('Image changed')
    return frames


def prepare(args):
    cfg = read(args.config)
    reqs = previous.check_requests(checked(args.requests,cfg['requests_sha256']))
    original = checked(args.oracle_prepared,cfg['oracle_prepared_sha256'])
    if sha(args.plans) != cfg['plans_sha256']: raise ValueError('Plans changed')
    plans = {r['qid']:r['plan'] for r in previous.lines(args.plans) if r['status']=='ok'}
    oracle = {r['request']['qid']:r for r in original if r['condition']=='oracle_video16'}
    if len(reqs)!=cfg['sample_count'] or set(oracle)!=set(plans) or set(oracle)!={r['qid'] for r in reqs}:
        raise ValueError('Incomplete source matrix')
    out = begin(args.output,args.config,args.mock)
    rows = []
    for r in reqs:
        old = oracle[r['qid']]
        if old['request'] != r: raise ValueError('Request changed')
        row = {'request':r,'frames':normalize_frames(old['frames']),
            'frame_sha256':old['frame_sha256'],'plan':needs.validate_plan(plans[r['qid']])}
        validate_row(row,cfg['frames']); rows.append(row)
    donor_indices(rows)
    # Known positive controls use only the repository's offline fixture generator.
    from scripts.make_synthetic_fixture import build_video
    fixture = out/'fixture';fixture.mkdir()
    video = build_video(fixture)
    asset = study.asset_from_probe('transport_fixture',video)
    scenes = [Scene(scene_id=f'control_{i}',start_sec=t-.001,end_sec=t+.001)
              for i,t in enumerate([6.,18.])]
    frames = study.extract_scene_frames(video,out/'control_images',scenes,max_side=1024)
    controls = []
    for i,frame in enumerate(frames):
        controls.append({'request':{'qid':f'control_{i}','video_id':'transport_fixture',
            'video_path':str(video),'duration_sec':asset.duration_sec,
            'question':'Which color covers most of the image background?',
            'choices':['Blue','Red','Green','Yellow']},
            'frames':normalize_frames([frame.model_dump()]),'frame_sha256':[sha(frame.image_path)]})
        validate_row(controls[-1],1)
    if len(controls)!=2: raise ValueError('Missing fixture controls')
    write(out/'prepared.json',{'rows':rows,'controls':controls})
    return finish(out,['prepared.json'],{'scope':'oracle_diagnostic','rows':len(rows),
        'new_benchmark_frames':0,'new_synthetic_frames':2,'model_calls':0,'mock':args.mock,
        'input_hashes':{'oracle':sha(args.oracle_prepared),'requests':sha(args.requests),'plans':sha(args.plans)}})


class Backend:
    def __init__(self,cfg,model_path,precision):
        from scripts.gpu_agent_experiment import TransformersBackend
        from scripts.train_agent_sft import QuantizedBackend
        setup = {'seed':cfg['seed'],'model_path':str(model_path),'min_pixels':65536,
            'max_pixels':65536,'attention':'sdpa','max_new_tokens':cfg['max_output_tokens'],
            'max_seconds_per_generation':120,'max_consecutive_errors':3}
        self.base = (QuantizedBackend if precision=='nf4' else TransformersBackend)(setup)
        self.base.model.eval();self.base.model.config.use_cache=True

    def generate(self,prompt,frames,schema,pixels,constrained):
        from PIL import Image
        b=self.base;started=time.perf_counter();images=[]
        record={'images':len(frames),'input_tokens':0,'output_tokens':0,'status':'error',
                'generation_attempted':False,'pixels_budget':pixels}
        raw=''
        try:
            b.processor.image_processor.size={'shortest_edge':pixels,'longest_edge':pixels}
            content=[{'type':'text','text':prompt}]
            for i,f in enumerate(frames):
                content += [{'type':'text','text':f'anchor_index={i}; id={f.frame_id}; source_time_sec={f.timestamp_sec}'},
                            {'type':'image'}]
                with Image.open(f.image_path) as im: images.append(im.convert('RGB'))
            text=b.processor.apply_chat_template([{'role':'user','content':content}],
                                                  tokenize=False,add_generation_prompt=True)
            inputs=b.processor(text=[text],**({'images':images} if images else {}),return_tensors='pt')
            grids=inputs.image_grid_thw.tolist() if images else []
            tensor=inputs.get('pixel_values')
            image_id=b.model.config.image_token_id
            record.update(image_grid_thw=grids,visual_tokens=sum(t*h*w//4 for t,h,w in grids),
                image_token_count=int((inputs.input_ids==image_id).sum()),
                pixel_tensor_sha256=hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() if images else None,
                pixel_tensor_shape=list(tensor.shape) if images else [],
                pixel_tensor_std=float(tensor.float().std()) if images else 0.,
                image_dimensions=[list(im.size) for im in images],
                input_tokens=int(inputs.input_ids.shape[-1]),rendered_prompt_sha256=digest(text),
                input_ids_sha256=hashlib.sha256(inputs.input_ids.numpy().tobytes()).hexdigest())
            assert_visual_transport(record,len(frames))
            inputs=inputs.to('cuda:0')
            record['processor_sec']=time.perf_counter()-started
            b.torch.cuda.reset_peak_memory_stats()
            kwargs={'prefix_allowed_tokens_fn':b._prefix_constraint(schema)} if constrained else {}
            record['generation_attempted']=True
            with b.torch.inference_mode():
                output=b.model.generate(**inputs,do_sample=False,max_new_tokens=b.cfg['max_new_tokens'],
                    max_time=b.cfg['max_seconds_per_generation'],**kwargs)
            tokens=output[0,inputs.input_ids.shape[-1]:]
            raw=b.processor.tokenizer.decode(tokens,skip_special_tokens=True)
            record.update(output_tokens=int(tokens.shape[-1]),status='complete',
                          peak_allocated_bytes=b.torch.cuda.max_memory_allocated())
        except Exception as exc:
            record['error_type']=type(exc).__name__
        finally:
            for im in images: im.close()
            record['elapsed_sec']=time.perf_counter()-started
        return raw,record


def task_prompt(row,condition):
    if condition['prompt']=='minimal':return minimal_prompt(row['request'])
    times=[f['timestamp_sec'] for f in row['frames']]
    packet={'candidate_id':'c0','interval_sec':[times[0],min(row['request']['duration_sec'],times[-1]+.000001)],
            'frames':row['frames'],'frame_sha256':row['frame_sha256']}
    return needs.verdict_prompt(row['request'],packet,row['plan'])


def run(args):
    cfg=read(args.config);prepared=Path(args.prepared)
    verify(prepared,args.config,args.mock)
    data=read(prepared/'prepared.json');rows=data['rows'];controls=data['controls']
    donors=donor_indices(rows)
    for r in rows:validate_row(r,cfg['frames'])
    for r in controls:validate_row(r,1)
    out=begin(args.output,args.config,args.mock)
    write(out/'prepared.json',data)
    model_hash=None if args.mock else previous.verify_model_snapshot(args.model_path,cfg['model_revision'])
    if not args.mock and model_hash!=cfg['model_manifest_sha256']:raise ValueError('Model changed')
    write(out/'model.json',{'manifest_sha256':model_hash,'precision':args.precision,'mock':args.mock})
    started=time.perf_counter();backend=None;load_error=None
    try:
        if not args.mock:backend=Backend(cfg,args.model_path,args.precision)
    except Exception as exc:load_error=type(exc).__name__
    load_sec=time.perf_counter()-started
    tasks=[]
    for row in controls:
        tasks.append((row,row,{'name':args.precision+'_control','precision':args.precision,
            'pixels':65536,'constrained':True,'input':'oracle','prompt':'minimal'},True))
    for condition in cfg['conditions']:
        if condition['precision']!=args.precision:continue
        for i,row in enumerate(rows):
            donor=rows[donors[i]] if condition['input']=='wrong' else row
            tasks.append((row,donor,condition,False))
    records=[];blocked=load_error;consecutive=0
    for row,donor,condition,control in tasks:
        prompt,schema=task_prompt(row,condition)
        frames=[] if condition['input']=='text' else [VideoFrame.model_validate(f) for f in donor['frames']]
        hashes=[] if not frames else donor['frame_sha256']
        r={'qid':row['request']['qid'],'condition':condition['name'],'precision':args.precision,
            'control':control,'mock':args.mock,'prompt_sha256':digest(prompt),
            'schema_sha256':digest(json.dumps(schema,sort_keys=True)),
            'donor_qid':donor['request']['qid'] if frames else None,'image_sha256':hashes,
            'source_times_sec':[f.timestamp_sec for f in frames],'answer':'abstain','status':'error'}
        if blocked:
            r.update(error_type='circuit_open:'+blocked,raw='',model_call=None)
        else:
            if args.mock:
                raw=json.dumps({'answer':'A'});cost={'generation_attempted':False,'status':'mock'}
                if condition['prompt']=='needs':
                    raw=json.dumps({'decision':'target_unobserved','evidence':'','region':'full',
                        'target_description':'Mock target','observed_fact':'Mock only'})
            else:raw,cost=backend.generate(prompt,frames,schema,condition['pixels'],condition['constrained'])
            r.update(raw=raw,model_call=cost)
            try:
                if cost['status']=='error':raise RuntimeError(cost.get('error_type','backend_error'))
                if condition['prompt']=='needs':
                    times=[f['timestamp_sec'] for f in row['frames']]
                    packet={'candidate_id':'c0','interval_sec':[times[0],min(row['request']['duration_sec'],times[-1]+.000001)],
                        'frames':row['frames'],'frame_sha256':row['frame_sha256']}
                    verdict=needs.decode_verdict(json.loads(raw),row['request'],packet,row['plan'])
                    r['verdict']=verdict;r['answer']=verdict['answer']
                else:r['answer']=parse_answer(raw,row['request']['choices'])
                r['status']='ok';consecutive=0
            except Exception as exc:
                r['error_type']=cost.get('error_type',type(exc).__name__)
                # Contract failures count as outputs, not reasons to stop other conditions.
                if cost['status']=='error':consecutive+=1
                if cost.get('error_type')=='OutOfMemoryError' or consecutive>=3:
                    blocked=cost.get('error_type','backend_errors')
        append(out/'records.jsonl',r);records.append(r)
        print(json.dumps({'finished':len(records),'condition':r['condition'],'status':r['status']}),flush=True)
    calls=[r['model_call'] for r in records if r['model_call']]
    return finish(out,['prepared.json','model.json','records.jsonl'],{'records':len(records),'mock':args.mock,
        'model_load_sec':load_sec,'load_error':load_error,'blocked':blocked,
        'new_model_calls':sum(c['generation_attempted'] for c in calls)})


def evaluate(args):
    cfg=read(args.config);labels=checked(args.labels,cfg['labels_sha256'])
    records=[];loads={}
    for name in ['nf4','bf16']:
        folder=Path(args.runs)/name;verify(folder,args.config,args.mock)
        records+=previous.lines(folder/'records.jsonl');loads[name]=read(folder/'summary.json')
    if len(records)!=cfg['max_generations']:raise ValueError('Generation matrix incomplete')
    result=summarize(records,{r['qid']:r['right_answer'] for r in labels},
        [c['name'] for c in cfg['conditions']],{'control_0':'B','control_1':'A'})
    costs={}
    for name in sorted({r['condition'] for r in records}):
        calls=[r['model_call'] for r in records if r['condition']==name and r['model_call']]
        costs[name]={key:sum(c.get(key,0) for c in calls) for key in
            ['generation_attempted','images','input_tokens','output_tokens','visual_tokens','elapsed_sec']}
        costs[name]['peak_allocated_bytes']=max([c.get('peak_allocated_bytes',0) for c in calls],default=0)
    result.update(mock=args.mock,costs=costs,loads=loads)
    out=begin(args.output,args.config,args.mock);write(out/'evaluation.json',result)
    return finish(out,['evaluation.json'],{'mock':args.mock,'records':len(records)})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','evaluate'])
    for name in ['config','requests','oracle-prepared','plans','output','prepared','model-path','runs','labels']:
        p.add_argument('--'+name,type=Path)
    p.add_argument('--precision',choices=['nf4','bf16']);p.add_argument('--mock',action='store_true')
    args=p.parse_args();print(json.dumps(globals()[args.action](args)),flush=True)


if __name__=='__main__':main()
