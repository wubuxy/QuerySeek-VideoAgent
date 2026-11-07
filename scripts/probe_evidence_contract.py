"""Strict-contract repair: reuse joint controls, rerun candidate verifications once."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import probe_evidence_packets as p
from queryseek.retrieval.evidence_contract import strict_prompt,decode_verdict
from queryseek.retrieval.evidence_packets import aggregate,next_action,packet_frames_valid
from queryseek.schemas import VideoFrame


def strict_sources():
    return {n:p.sha256_file(ROOT/n) for n in ['queryseek/retrieval/evidence_contract.py','scripts/probe_evidence_contract.py']}


def call(backend,request,packet,cfg,mock):
    prompt,schema=strict_prompt(request,packet);before=len(backend.calls) if backend else 0
    r={'candidate_id':packet['candidate_id'],'status':'error','mode':'strict_verify','mock':mock,
       'prompt_sha256':p.previous.digest(prompt),'schema_sha256':p.previous.digest(json.dumps(schema,sort_keys=True))}
    try:
        value=({'decision':'insufficient:visual_detail','evidence':'','observed_fact':'Mock only.'} if mock else
            backend(prompt,[VideoFrame.model_validate(f) for f in packet['frames']],schema,
                    {'input':'images','pixels':cfg['pixels']},request['duration_sec']))
        r['atomic_prediction']=value
        canonical=decode_verdict(value,packet,request['choices'],request['duration_sec'])
        r['raw_prediction']={k:v for k,v in canonical.items() if k!='evidence_source_times_sec'}
        r.update(canonical,status='ok')
    except Exception as exc:r['error_type']=type(exc).__name__
    r['model_calls']=backend.calls[before:] if backend else []
    return r


def run(old_prepared,old_run,base_config,config_path,output,model_path=None,mock=False):
    cfg=p.read(config_path)
    if p.sha256_file(base_config)!=cfg['base_config_sha256']:raise ValueError('Base protocol changed')
    p.verify(old_prepared,base_config,mock);p.verify(old_run,base_config,mock)
    if p.sha256_file(old_prepared/'prepared.json')!=cfg['prepared_sha256']:raise ValueError('Prepared input changed')
    if p.sha256_file(old_run/'records.jsonl')!=cfg['old_records_sha256']:raise ValueError('Joint controls changed')
    rows=p.read(old_prepared/'prepared.json');old={r['qid']:r for r in p.lines(old_run/'records.jsonl')}
    count=sum(len(r['packets']) for r in rows)
    if len(rows)!=cfg['sample_count'] or count+len(rows)!=cfg['max_new_verifications']:raise ValueError('Frozen new-call budget mismatch')
    output=p.output_path(output);output.mkdir(parents=True,exist_ok=False)
    prepared=p.begin(output/'prepared',config_path,mock,{'reused_manifest_sha256':p.sha256_file(old_prepared/'manifest.json')})
    (prepared/'prepared.json').write_bytes((old_prepared/'prepared.json').read_bytes())
    p.finish(prepared,['prepared.json'],{'mock':mock,'queries':len(rows),'candidates':count,
        'new_frames_extracted':0,'reused_frames':count*cfg['frames_per_candidate'],'new_clip_encoding':0})
    run_dir=p.begin(output/'run',config_path,mock,{'prepared_manifest_sha256':p.sha256_file(prepared/'manifest.json'),
        'reused_joint_records_sha256':cfg['old_records_sha256']})
    initial=p.read(run_dir/'manifest.initial.json');initial['strict_source_sha256']=strict_sources()
    p.write_json(run_dir/'manifest.initial.json',initial)
    for row in rows:
        p.previous.check_requests([row['request']])
        for packet in row['packets']:p.check_images(packet,row['request']['duration_sec'])
    backend,load_sec=p.previous.load_backend(cfg,model_path,run_dir,mock);results=[]
    for row in rows:
        req=row['request'];packets=row['packets'];q=req['qid'];result={'qid':q,'mock':mock}
        for mode in ['joint_forward','joint_reverse']:
            result[mode]={**old[q][mode],'model_calls':[], 'reused_control':True,
                          'reused_model_call_count':len(old[q][mode]['model_calls'])}
        result['verdicts']=[call(backend,req,packet,cfg,mock) for packet in packets]
        result['packet_verify']=aggregate(packets,result['verdicts'])
        result['action']=next_action(packets,result['verdicts'],1);result['refinement']=None
        updated=result['verdicts'][:]
        if result['action']['action']=='observe_more':
            packet=next(x for x in packets if x['candidate_id']==result['action']['candidate_id'])
            extra={'candidate_id':packet['candidate_id'],'status':'error'}
            try:
                asset=p.asset_from_probe(req['video_id'],req['video_path'])
                if asset.sha256!=row['video_sha256']:raise ValueError('Video changed')
                observed,cost=p.observe(asset,packet,cfg['extra_frames'],cfg);extra['cost']=cost
                pairs=sorted(zip(packet['frames']+observed['frames'],packet['frame_sha256']+observed['frame_sha256']),
                             key=lambda pair:pair[0]['timestamp_sec'])
                combined={**packet,'frames':[f for f,h in pairs],'frame_sha256':[h for f,h in pairs]}
                packet_frames_valid(combined,req['duration_sec']);extra['packet']=combined
                extra['verdict']=call(backend,req,combined,cfg,mock);extra['status']='ok'
                updated=[extra['verdict'] if x['candidate_id']==packet['candidate_id'] else x for x in updated]
            except Exception as exc:extra['error_type']=type(exc).__name__
            result['refinement']=extra
        result['packet_observe']=aggregate(packets,updated);result['final_action']=next_action(packets,updated,0)
        results.append(result);p.append(run_dir/'records.jsonl',result)
        if backend and len(backend.calls)>cfg['max_new_verifications']:raise ValueError('New generation budget exceeded')
        print(json.dumps({'finished':len(results),'action':result['action']['action'],
                         'calls':len(backend.calls) if backend else 0}),flush=True)
    summary={'mock':mock,'queries':len(rows),'backend_calls':len(backend.calls) if backend else 0,
        'model_load_sec':load_sec,'reused_joint_calls':sum(len(old[q][m]['model_calls']) for q in old for m in ['joint_forward','joint_reverse']),
        'refinement_attempts':sum(r['refinement'] is not None for r in results)}
    return p.finish(run_dir,['records.jsonl','model.json'],summary)


def evaluate(requests,labels,folder,config,output):
    if p.read(folder/'run/manifest.json')['strict_source_sha256']!=strict_sources():raise ValueError('Strict code changed')
    return p.evaluate(requests,labels,folder/'prepared',folder/'run',config,output)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=['run','evaluate'])
    for n in ['old-prepared','old-run','base-config','config','output','folder','requests','labels','model-path']:
        a.add_argument('--'+n,type=Path,required=n in ['config','output'])
    a.add_argument('--mock',action='store_true');v=a.parse_args()
    if v.action=='run':
        if v.labels is not None:a.error('Inference rejects labels')
        r=run(v.old_prepared,v.old_run,v.base_config,v.config,v.output,v.model_path,v.mock)
    else:r=evaluate(v.requests,v.labels,v.folder,v.config,v.output)
    print(json.dumps({k:x for k,x in r.items() if k not in ['details','observations']}))
