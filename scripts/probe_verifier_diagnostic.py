"""Post-hoc, separately frozen follow-up to fixed-evidence model diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import probe_model_diagnostic as s


def sources():
    return {**s.sources(),'scripts/probe_verifier_diagnostic.py':s.sha(Path(__file__))}


def begin(path,config,mock):
    path=s.study.baseline.checked_output(path);path.mkdir(parents=True,exist_ok=False)
    s.write(path/'manifest.initial.json',{'config_sha256':s.sha(config),'source_sha256':sources(),'mock':mock})
    return path


def verify(path,config,mock):
    m=s.read(path/'manifest.json')
    if m['config_sha256']!=s.sha(config) or m['source_sha256']!=sources() or m['mock']!=mock:
        raise ValueError('Provenance mismatch')
    for n,h in m['files'].items():
        if s.sha(path/n)!=h:raise ValueError('Artifact changed')


def decode(raw,row):
    """Raw decision is diagnostic intent only; it never bypasses strict validation."""
    result={'status':'error','answer':'abstain','raw_decision':'invalid','raw_answer':'abstain'}
    try:
        value=json.loads(raw)
        if not isinstance(value,dict):raise ValueError('Object required')
        decision=value.get('decision')
        if isinstance(decision,str):result['raw_decision']=decision
        letters=[chr(65+i) for i in range(len(row['request']['choices']))]
        if decision in ['answer:'+x for x in letters]:result['raw_answer']=decision.split(':')[1]
        times=[f['timestamp_sec'] for f in row['frames']]
        packet={'candidate_id':'c0','interval_sec':[times[0],min(row['request']['duration_sec'],times[-1]+.000001)],
                'frames':row['frames'],'frame_sha256':row['frame_sha256']}
        verdict=s.needs.decode_verdict(value,row['request'],packet,row['plan'])
        result.update(status='ok',answer=verdict['answer'],verdict=verdict)
    except Exception as exc:result['error_type']=type(exc).__name__
    return result


def run(args):
    cfg=s.read(args.config)
    if s.sha(args.parent_config)!=cfg['parent_config_sha256']:raise ValueError('Parent config changed')
    parent=Path(args.prepared);s.verify(parent,args.parent_config,args.mock)
    data=s.read(parent/'prepared.json');rows=data['rows']
    if len(rows)!=cfg['sample_count']:raise ValueError('Incomplete sources')
    for row in rows:s.validate_row(row,cfg['frames'])
    condition=next(c for c in cfg['conditions'] if c['precision']==args.precision)
    out=begin(args.output,args.config,args.mock);s.write(out/'prepared.json',data)
    model_hash=None if args.mock else s.previous.verify_model_snapshot(args.model_path,cfg['model_revision'])
    if not args.mock and model_hash!=cfg['model_manifest_sha256']:raise ValueError('Model changed')
    s.write(out/'model.json',{'mock':args.mock,'manifest_sha256':model_hash,'precision':args.precision})
    started=time.perf_counter();backend=None;blocked=None
    try:
        if not args.mock:backend=s.Backend(cfg,args.model_path,args.precision)
    except Exception as exc:blocked=type(exc).__name__
    load_sec=time.perf_counter()-started;records=[];errors=0
    for row in rows:
        prompt,schema=s.task_prompt(row,{'prompt':'needs'})
        record={'qid':row['request']['qid'],'condition':condition['name'],'precision':args.precision,
            'mock':args.mock,'prompt_sha256':s.digest(prompt),'schema_sha256':s.digest(json.dumps(schema,sort_keys=True)),
            'image_sha256':row['frame_sha256'],'source_times_sec':[f['timestamp_sec'] for f in row['frames']],
            'status':'error','answer':'abstain','raw_answer':'abstain','raw_decision':'invalid','raw':'','model_call':None}
        if blocked:record['error_type']='circuit_open:'+blocked
        else:
            if args.mock:
                raw=json.dumps({'decision':'target_unobserved','evidence':'','region':'full',
                    'target_description':'Mock target','observed_fact':'Mock only'})
                cost={'generation_attempted':False,'status':'mock'}
            else:
                raw,cost=backend.generate(prompt,[s.VideoFrame.model_validate(f) for f in row['frames']],
                    schema,262144,condition['constrained'])
            record.update(raw=raw,model_call=cost)
            if cost['status']=='error':
                errors+=1;record['error_type']=cost.get('error_type','backend_error')
                if errors>=3 or cost.get('error_type')=='OutOfMemoryError':blocked=record['error_type']
            else:record.update(decode(raw,row));errors=0
        records.append(record);s.append(out/'records.jsonl',record)
        print(json.dumps({'finished':len(records),'condition':condition['name'],'status':record['status']}),flush=True)
    calls=[r['model_call'] for r in records if r['model_call']]
    return s.finish(out,['prepared.json','model.json','records.jsonl'],{'records':len(records),'mock':args.mock,
        'model_load_sec':load_sec,'blocked':blocked,'new_model_calls':sum(c['generation_attempted'] for c in calls)})


def metrics(records,labels):
    if len(records)!=len(labels) or {r['qid'] for r in records}!=set(labels):raise ValueError('Full denominator required')
    return {'n':len(labels),'valid':sum(r['status']=='ok' for r in records),
        'answered':sum(r['status']=='ok' and r['answer']!='abstain' for r in records),
        'correct':sum(r['status']=='ok' and r['answer']==labels[r['qid']] for r in records),
        'raw_decision_answered':sum(r['raw_answer']!='abstain' for r in records),
        'raw_decision_correct':sum(r['raw_answer']==labels[r['qid']] for r in records)}


def evaluate(args):
    cfg=s.read(args.config);labels={r['qid']:r['right_answer'] for r in s.checked(args.labels,cfg['labels_sha256'])}
    parent=Path(args.parent_run);s.verify(parent,args.parent_config,args.mock)
    prepared=s.read(parent/'prepared.json');rows={r['request']['qid']:r for r in prepared['rows']}
    base=[r for r in s.previous.lines(parent/'records.jsonl') if r['condition']=='nf4_high_needs']
    base=[{**r,**decode(r['raw'],rows[r['qid']])} for r in base]
    result={'metrics':{'nf4_high_needs':metrics(base,labels)},'costs':{},'mock':args.mock,'loads':{}}
    all_records=[];matched=0;baseline={r['qid']:r for r in base}
    for precision in ['nf4','bf16']:
        folder=Path(args.runs)/precision;verify(folder,args.config,args.mock)
        records=s.previous.lines(folder/'records.jsonl');all_records+=records
        name=next(c['name'] for c in cfg['conditions'] if c['precision']==precision)
        result['metrics'][name]=metrics(records,labels);calls=[r['model_call'] for r in records if r['model_call']]
        result['costs'][name]={k:sum(c.get(k,0) for c in calls) for k in ['generation_attempted','images','input_tokens','output_tokens','visual_tokens','elapsed_sec']}
        result['costs'][name]['peak_allocated_bytes']=max([c.get('peak_allocated_bytes',0) for c in calls],default=0)
        result['loads'][precision]=s.read(folder/'summary.json')
        for record in records:
            old=baseline[record['qid']]
            for k in ['prompt_sha256','schema_sha256','image_sha256','source_times_sec']:
                if record[k]!=old[k]:raise ValueError('Paired request changed')
            a,b=record['model_call'],old['model_call']
            if a and b and a.get('generation_attempted') and b.get('generation_attempted'):
                for k in ['input_ids_sha256','pixel_tensor_sha256','image_grid_thw','rendered_prompt_sha256']:
                    if a[k]!=b[k]:raise ValueError('Paired processor input changed')
                matched+=1
    if len(all_records)!=cfg['max_generations']:raise ValueError('Incomplete generation matrix')
    result.update(matched_input_pairs=matched,new_model_calls=sum(c['generation_attempted'] for c in result['costs'].values()),
        scope='Post-hoc development diagnosis, separately frozen; raw decision intent is not a validated answer.')
    out=begin(args.output,args.config,args.mock);s.write(out/'evaluation.json',result)
    return s.finish(out,['evaluation.json'],{'mock':args.mock,'records':len(all_records)})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','evaluate'])
    for n in ['config','parent-config','prepared','output','model-path','runs','labels','parent-run']:p.add_argument('--'+n,type=Path)
    p.add_argument('--precision',choices=['nf4','bf16']);p.add_argument('--mock',action='store_true')
    args=p.parse_args();print(json.dumps(globals()[args.action](args)),flush=True)


if __name__=='__main__':main()
