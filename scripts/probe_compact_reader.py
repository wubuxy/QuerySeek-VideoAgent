"""Frozen development/new-source reader experiment; labels stay outside inference."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from queryseek.eval import compact_reader as c
from scripts import probe_model_diagnostic as s
from scripts import probe_verifier_diagnostic as legacy


def sources():
    return {**legacy.sources(),**{n:s.sha(ROOT/n) for n in ['scripts/probe_compact_reader.py','queryseek/eval/compact_reader.py']}}


def begin(path,config,mock):
    path=s.study.baseline.checked_output(path);path.mkdir(parents=True,exist_ok=False)
    s.write(path/'manifest.initial.json',{'config_sha256':s.sha(config),'source_sha256':sources(),'mock':mock})
    return path


def verify(path,config,mock):
    m=s.read(path/'manifest.json')
    if m['config_sha256']!=s.sha(config) or m['source_sha256']!=sources() or m['mock']!=mock:raise ValueError('Provenance mismatch')
    for n,h in m['files'].items():
        if s.sha(path/n)!=h:raise ValueError('Artifact changed')


def prepare(args):
    cfg=s.read(args.config)
    for key,path in [('parent_config_sha256',args.parent_config),('legacy_config_sha256',args.legacy_config)]:
        if s.sha(path)!=cfg[key]:raise ValueError('Historical config changed')
    s.verify(Path(args.parent_prepared),args.parent_config,args.mock)
    s.verify(Path(args.parent_run),args.parent_config,args.mock)
    legacy.verify(Path(args.legacy_run),args.legacy_config,args.mock)
    dev=s.read(Path(args.parent_prepared)/'prepared.json')['rows']
    requests=s.previous.check_requests(s.checked(args.requests,cfg['fresh_requests_sha256']))
    labels={r['qid']:r for r in s.checked(args.labels,cfg['fresh_labels_sha256'])}
    if (len(dev)!=cfg['sample_count'] or len(requests)!=cfg['sample_count']
        or {r['video_id'] for r in requests}&{r['request']['video_id'] for r in dev}
        or set(labels)!={r['qid'] for r in requests}):raise ValueError('Invalid disjoint source groups')
    for row in dev:s.validate_row(row,cfg['frames'])
    s.donor_indices([{'request':r} for r in requests])
    out=begin(args.output,args.config,args.mock);fresh=[];audit=[]
    for request in requests:
        row={'request':dict(request),'frames':[],'frame_sha256':[]}
        try:
            asset=s.study.asset_from_probe(request['video_id'],request['video_path'])
            row['request']['duration_sec']=asset.duration_sec
            frames,cost=s.study.oracle_frames(row['request'],labels[request['qid']]['gt_intervals_sec'],cfg['frames'])
            row.update(frames=s.normalize_frames([f.model_dump() for f in frames]),
                       frame_sha256=[s.sha(f.image_path) for f in frames])
            s.validate_row(row,cfg['frames'])
            audit.append({'qid':request['qid'],'annotation_duration_sec':request['duration_sec'],
                          'actual_duration_sec':asset.duration_sec,'preprocessing':cost,'status':'ok'})
        except Exception as exc:
            row['preparation_error']=type(exc).__name__
            audit.append({'qid':request['qid'],'status':'error','error_type':type(exc).__name__})
        fresh.append(row)
    minimal=[r for r in s.previous.lines(Path(args.parent_run)/'records.jsonl') if r['condition']=='bf16_high_json']
    old=s.previous.lines(Path(args.legacy_run)/'records.jsonl')
    ids={r['request']['qid'] for r in dev}
    if (len(minimal)!=len(ids) or len(old)!=len(ids) or {r['qid'] for r in minimal}!=ids
        or {r['qid'] for r in old}!=ids):raise ValueError('Incomplete historical comparisons')
    for row in dev:
        r=next(r for r in minimal if r['qid']==row['request']['qid']);prompt,schema=s.minimal_prompt(row['request'])
        if (r['prompt_sha256']!=s.digest(prompt) or r['image_sha256']!=row['frame_sha256']
            or r['source_times_sec']!=[f['timestamp_sec'] for f in row['frames']]
            or r['schema_sha256']!=s.digest(json.dumps(schema,sort_keys=True)) or r['mock']!=args.mock):
            raise ValueError('Historical baseline input changed')
    s.write(out/'prepared.json',{'development':dev,'new_sources':fresh})
    s.write(out/'reused_minimal.json',minimal);s.write(out/'reused_legacy.json',old);s.write(out/'media_audit.json',audit)
    return s.finish(out,['prepared.json','reused_minimal.json','reused_legacy.json','media_audit.json'],
        {'mock':args.mock,'sources_per_group':cfg['sample_count'],'fresh_preparation_errors':sum('preparation_error' in r for r in fresh),
         'new_frames':sum(r.get('preprocessing',{}).get('new_frames_extracted',0) for r in audit)})


class Engine:
    def __init__(self,cfg,backend,output,mock,blocked=None):
        self.cfg,self.backend,self.output,self.mock=cfg,backend,output,mock
        self.calls=[];self.blocked=blocked;self.errors=0

    def call(self,prompt,schema,frames,step,identity):
        record={'step':step,'identity':identity,'prompt_sha256':s.digest(prompt),
            'schema_sha256':s.digest(json.dumps(schema,sort_keys=True)),
            'image_sha256':[s.sha(f.image_path) for f in frames],
            'source_times_sec':[f.timestamp_sec for f in frames],'reused':False}
        if self.blocked:
            raw='';cost={'status':'error','generation_attempted':False,'error_type':'circuit_open:'+self.blocked}
        elif self.mock:
            raw=json.dumps({'fact':'A visible fixture background.','evidence':[0]}) if step=='fact' else json.dumps({'answer':'A'})
            if step=='compact':raw=json.dumps({'answer':'A','evidence':[0]})
            if step=='legacy':raw=json.dumps({'decision':'target_unobserved','evidence':'','region':'full',
                'target_description':'Mock target','observed_fact':'Mock only'})
            cost={'status':'mock','generation_attempted':False}
        else:raw,cost=self.backend.generate(prompt,frames,schema,self.cfg['pixels'],True)
        record.update(raw=raw,model_call=cost);self.calls.append(record);s.append(self.output/'calls.jsonl',record)
        if cost['status']=='error':
            self.errors+=1
            if self.errors>=3 or cost.get('error_type')=='OutOfMemoryError':self.blocked=cost.get('error_type','backend_errors')
        else:self.errors=0
        return record


def task(row,donor,method,kind,group,engine):
    identity={'group':group,'qid':row['request']['qid'],'method':method,'input':kind}
    record={**identity,'donor_qid':donor['request']['qid'],'status':'error','answer':'abstain','evidence':[],
            'calls':[],'mock':engine.mock}
    if 'preparation_error' in row or 'preparation_error' in donor:
        return {**record,'error_type':'preparation_error'}
    frames=[s.VideoFrame.model_validate(f) for f in donor['frames']];request=row['request']
    try:
        if method=='minimal':prompt,schema=s.minimal_prompt(request);step='minimal'
        elif method=='compact':prompt,schema=c.compact_prompt(request,len(frames));step='compact'
        elif method=='fact_map':prompt,schema=c.fact_prompt(request,len(frames));step='fact'
        elif method=='legacy_explicit':
            prompt,schema=c.explicit_legacy(*s.task_prompt(row,{'prompt':'needs'}));step='legacy'
        else:raise ValueError('Unknown method')
        call=engine.call(prompt,schema,frames,step,identity);record['calls'].append(call)
        if call['model_call']['status']=='error':raise RuntimeError('backend_error')
        if method=='minimal':answer=s.parse_answer(call['raw'],request['choices']);decoded={'answer':answer,'evidence':[]}
        elif method=='compact':decoded=c.decode_compact(call['raw'],request,len(frames))
        elif method=='legacy_explicit':
            parsed=legacy.decode(call['raw'],row)
            if parsed['status']!='ok':raise ValueError('Invalid legacy contract')
            decoded={'answer':parsed['answer'],'evidence':parsed['verdict']['evidence_frames']};record['verdict']=parsed['verdict']
        else:
            observation=c.decode_fact(call['raw'],len(frames));record['observation']=observation
            prompt,schema=c.map_prompt(request,observation)
            call=engine.call(prompt,schema,[],'map',identity);record['calls'].append(call)
            if call['model_call']['status']=='error':raise RuntimeError('backend_error')
            decoded=c.decode_map(call['raw'],request,observation)
        record.update(decoded,status='ok',evidence_source_times_sec=[frames[i].timestamp_sec for i in decoded['evidence']])
    except Exception as exc:record['error_type']=type(exc).__name__
    return record


def run(args):
    cfg=s.read(args.config);prepared=Path(args.prepared);verify(prepared,args.config,args.mock)
    groups=s.read(prepared/'prepared.json');old=s.read(prepared/'reused_minimal.json')
    for rows in groups.values():
        if len(rows)!=cfg['sample_count']:raise ValueError('Incomplete source group')
        s.donor_indices(rows)
        for row in rows:
            if 'preparation_error' not in row:s.validate_row(row,cfg['frames'])
    out=begin(args.output,args.config,args.mock)
    for name in ['prepared.json','reused_minimal.json','reused_legacy.json']:
        s.write(out/name,s.read(prepared/name))
    model_hash=None if args.mock else s.previous.verify_model_snapshot(args.model_path,cfg['model_revision'])
    if not args.mock and model_hash!=cfg['model_manifest_sha256']:raise ValueError('Model changed')
    s.write(out/'model.json',{'manifest_sha256':model_hash,'mock':args.mock,'precision':'bf16'})
    started=time.perf_counter();backend=None;blocked=None
    try:
        if not args.mock:backend=s.Backend(cfg,args.model_path,'bf16')
    except Exception as exc:blocked=type(exc).__name__
    load_sec=time.perf_counter()-started;engine=Engine(cfg,backend,out,args.mock,blocked);records=[]
    for group,rows in groups.items():
        donors=s.donor_indices(rows)
        for kind in ['matched','mismatched']:
            for method in ['minimal','compact','fact_map']:
                for i,row in enumerate(rows):
                    donor=rows[donors[i]] if kind=='mismatched' else row
                    if group=='development' and kind=='matched' and method=='minimal':
                        r=next(r for r in old if r['qid']==row['request']['qid'])
                        call={'step':'minimal','raw':r['raw'],'model_call':r['model_call'],'reused':True,
                            'prompt_sha256':r['prompt_sha256'],'schema_sha256':r['schema_sha256'],
                            'image_sha256':r['image_sha256'],'source_times_sec':r['source_times_sec']}
                        result={'group':group,'qid':r['qid'],'method':method,'input':kind,'donor_qid':r['qid'],
                            'status':r['status'],'answer':r['answer'],'evidence':[],'calls':[call],'mock':args.mock}
                    else:result=task(row,donor,method,kind,group,engine)
                    records.append(result);s.append(out/'records.jsonl',result)
                    print(json.dumps({'records':len(records),'group':group,'input':kind,'method':method,'status':result['status']}),flush=True)
        if group=='development':
            for row in rows:
                result=task(row,row,'legacy_explicit','matched',group,engine)
                records.append(result);s.append(out/'records.jsonl',result)
                print(json.dumps({'records':len(records),'group':group,'method':'legacy_explicit','status':result['status']}),flush=True)
    count=sum(r['model_call']['generation_attempted'] for r in engine.calls)
    if count>cfg['max_generations']:raise ValueError('Generation budget exceeded')
    return s.finish(out,['prepared.json','reused_minimal.json','reused_legacy.json','model.json','calls.jsonl','records.jsonl'],
        {'mock':args.mock,'records':len(records),'new_model_calls':count,'historical_comparison_rows':2*cfg['sample_count'],
         'model_load_sec':load_sec,'backend_blocked':engine.blocked})


def cost(calls):
    result={k:sum(r['model_call'].get(k,0) for r in calls) for k in
            ['generation_attempted','images','input_tokens','output_tokens','visual_tokens','elapsed_sec']}
    result['peak_allocated_bytes']=max([r['model_call'].get('peak_allocated_bytes',0) for r in calls],default=0)
    return result


def evaluate(args):
    cfg=s.read(args.config);folder=Path(args.runs);verify(folder,args.config,args.mock)
    label_groups={group:{r['qid']:r['right_answer'] for r in s.checked(path,cfg[key])} for group,path,key in [
        ('development',args.dev_labels,'dev_labels_sha256'),('new_sources',args.labels,'fresh_labels_sha256')]}
    groups=s.read(folder/'prepared.json');records=s.previous.lines(folder/'records.jsonl');physical=s.previous.lines(folder/'calls.jsonl')
    expected={(group,r['request']['qid'],method,kind) for group,rows in groups.items() for r in rows
              for method in ['minimal','compact','fact_map'] for kind in ['matched','mismatched']}
    expected|={('development',r['request']['qid'],'legacy_explicit','matched') for r in groups['development']}
    if len(records)!=len(expected) or {(r['group'],r['qid'],r['method'],r['input']) for r in records}!=expected:
        raise ValueError('Incomplete experiment matrix')
    result={'mock':args.mock,'groups':{},'new_physical_cost':cost(physical),
            'run_summary':s.read(folder/'summary.json'),'scope':'Privileged temporal evidence; source-disjoint small confirmation.'}
    derived=[]
    for group,rows in groups.items():
        ids=[r['request']['qid'] for r in rows];labels=label_groups[group]
        if set(labels)!=set(ids):raise ValueError('Label denominator mismatch')
        group_result={}
        for kind in ['matched','mismatched']:
            selected=[r for r in records if r['group']==group and r['input']==kind]
            for q in ids:
                a=next(r for r in selected if r['qid']==q and r['method']=='compact')
                b=next(r for r in selected if r['qid']==q and r['method']=='fact_map')
                r={**c.agreement(a,b),'qid':q,'method':'agreement','group':group,'input':kind,'calls':a['calls']+b['calls']}
                selected.append(r);derived.append(r)
            group_result[kind]={}
            for method in c.METHODS:
                values=[r for r in selected if r['method']==method]
                group_result[kind][method]={**c.metrics(values,ids,labels if kind=='matched' else None),
                    'logical_cost_including_reused':cost([x for r in values for x in r['calls']])}
        group_result['matched_vs_mismatched_answer_changes']={method:sum(
            next(r for r in records+derived if r['group']==group and r['qid']==q and r['method']==method and r['input']=='matched')['answer']!=
            next(r for r in records+derived if r['group']==group and r['qid']==q and r['method']==method and r['input']=='mismatched')['answer'] for q in ids)
            for method in c.METHODS}
        result['groups'][group]=group_result
    ids=list(label_groups['development']);old=s.read(folder/'reused_legacy.json')
    result['legacy_comparison']={'reused':c.metrics(old,ids,label_groups['development']),
        'explicit':c.metrics([r for r in records if r['method']=='legacy_explicit'],ids,label_groups['development'])}
    out=begin(args.output,args.config,args.mock);s.write(out/'evaluation.json',result);s.write(out/'derived.private.json',derived)
    return s.finish(out,['evaluation.json','derived.private.json'],{'mock':args.mock,'new_model_calls':result['new_physical_cost']['generation_attempted']})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','evaluate'])
    for name in ['config','parent-config','legacy-config','parent-prepared','parent-run','legacy-run','requests','labels',
                 'dev-labels','output','prepared','model-path','runs']:p.add_argument('--'+name,type=Path)
    p.add_argument('--mock',action='store_true');args=p.parse_args();print(json.dumps(globals()[args.action](args)),flush=True)


if __name__=='__main__':main()
