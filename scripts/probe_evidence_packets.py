"""Candidate-local verification and one bounded re-observation; labels only in evaluate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from queryseek import PIPELINE_VERSION
from queryseek.agent.tree_observations import output_path,write_json
from queryseek.agent import tree_observations as cache_paths
from queryseek.ingest.ffprobe import asset_from_probe,sha256_file
from queryseek.ingest.frames import extract_scene_frames,load_frames_index,save_frames_index
from queryseek.schemas import Scene,VideoFrame
from queryseek.eval.grounded_qa import checked_intervals,interval_set_iou
from queryseek.retrieval.query_reader import reader_prompt,bind_answer
from queryseek.retrieval.evidence_packets import (propose_packets,packet_frames_valid,joint_view,
    verification_prompt,validate_verdict,aggregate,next_action)
from scripts import probe_query_reader as previous

read,lines,append=previous.read,previous.lines,previous.append


def sources():
    return {**previous.sources(),**{n:sha256_file(ROOT/n) for n in [
        'queryseek/retrieval/evidence_packets.py','scripts/probe_evidence_packets.py']}}


def requests_for(path,cfg):
    rows=previous.check_requests(previous.checked(path,cfg['requests_sha256']))
    if len(rows)!=cfg['sample_count'] or cfg['max_generations']!=len(rows)*(8+cfg['max_seed_windows']):
        raise ValueError('Frozen call budget mismatch')
    if cfg['max_observe_actions']!=1:raise ValueError('Exactly one observation budget required')
    return rows


def begin(output,config_path,mock,dependencies=None):
    output=output_path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'manifest.initial.json',{'config_sha256':sha256_file(config_path),
        'source_sha256':sources(),'mock':mock,'dependencies':dependencies or {}})
    return output


def finish(output,names,summary):
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{**read(output/'manifest.initial.json'),
        'files':{n:sha256_file(output/n) for n in [*names,'summary.json']}})
    return summary


def verify(folder,config,mock):
    m=read(folder/'manifest.json')
    if m['config_sha256']!=sha256_file(config) or m['source_sha256']!=sources() or m['mock']!=mock:
        raise ValueError('Frozen provenance/mock mismatch')
    for n,d in m['files'].items():
        if sha256_file(folder/n)!=d:raise ValueError('Artifact hash mismatch')
    return m


def observe(asset,packet,count,cfg):
    started=time.perf_counter();(a,b),=checked_intervals([packet['interval_sec']],asset.duration_sec)
    times=[a+(i+.5)*(b-a)/count for i in range(count)]
    signature={'video_sha256':asset.sha256,'pipeline_version':PIPELINE_VERSION,'scope':'candidate-hypothesis',
        'interval_sec':[a,b],'requested_times_sec':times,'max_side':cfg['max_side'],'jpeg_quality':3,
        'extractor_sha256':sha256_file(ROOT/'queryseek/ingest/frames.py'),'sampler_version':'packet-bin-centers-v1'}
    key=previous.digest(json.dumps(signature,sort_keys=True))
    cache=output_path(cache_paths.ROOT/'outputs/evidence_packet_frame_cache'/key)
    hit=(cache/'manifest.json').exists()
    if hit:
        m=read(cache/'manifest.json')
        if m['signature']!=signature or sha256_file(cache/'frames.jsonl')!=m['index_sha256']:
            raise ValueError('Observation cache identity changed')
        frames=load_frames_index(cache/'frames.jsonl')
        if [sha256_file(f.image_path) for f in frames]!=m['image_sha256']:
            raise ValueError('Observation image changed')
    else:
        scenes=[Scene(scene_id=f'packet_{i:03d}',start_sec=t-.0001,end_sec=t+.0001) for i,t in enumerate(times)]
        frames=extract_scene_frames(asset.path,cache,scenes,max_side=cfg['max_side'])
        save_frames_index(frames,cache/'frames.jsonl')
        write_json(cache/'manifest.json',{'signature':signature,'index_sha256':sha256_file(cache/'frames.jsonl'),
                                        'image_sha256':[sha256_file(f.image_path) for f in frames]})
    row={**packet,'frames':[f.model_copy(update={'frame_id':f"{packet['candidate_id']}_{count}_{i}"}).model_dump()
                           for i,f in enumerate(frames)],'frame_sha256':[sha256_file(f.image_path) for f in frames]}
    packet_frames_valid(row,asset.duration_sec)
    if len(frames)!=count:raise ValueError('Incomplete packet observation')
    return row,{'cache_hit':hit,'new_frames_extracted':0 if hit else len(frames),
        'elapsed_sec':time.perf_counter()-started,'manifest_path':str(cache/'manifest.json'),
        'manifest_sha256':sha256_file(cache/'manifest.json')}


def prepare(requests_path,retrieval_path,config_path,output,mock=False):
    cfg=read(config_path);requests=requests_for(requests_path,cfg)
    if sha256_file(retrieval_path)!=cfg['retrieval_sha256']:raise ValueError('Frozen retrieval mismatch')
    old=[r for r in lines(retrieval_path) if r['method']=='dense_uniform']
    lookup=previous.matrix(old,[r['qid'] for r in requests],'method',['dense_uniform'])
    output=begin(output,config_path,mock);records=[];costs=[]
    for request in requests:
        q=request['qid'];asset=asset_from_probe(request['video_id'],request['video_path'])
        if abs(asset.duration_sec-request['duration_sec'])>.001:raise ValueError('Duration changed')
        proposed,uncovered=propose_packets(lookup[q,'dense_uniform'],asset.duration_sec,
                                          cfg['max_seed_windows'],cfg['seed_window_seconds'])
        packets=[]
        for packet in proposed:
            observed,cost=observe(asset,packet,cfg['frames_per_candidate'],cfg)
            packets.append(observed);costs.append({'qid':q,'candidate_id':packet['candidate_id'],**cost})
        records.append({'request':request,'video_sha256':asset.sha256,'packets':packets,
                        'old_review_times_not_in_new_candidates':uncovered})
        print(json.dumps({'stage':'prepare','finished':len(records),'candidates':len(packets)}),flush=True)
    write_json(output/'prepared.json',records)
    return finish(output,['prepared.json'],{'mock':mock,'queries':len(records),
        'candidates':sum(len(r['packets']) for r in records),'costs':costs,
        'new_frames_extracted':sum(c['new_frames_extracted'] for c in costs),'new_clip_encoding':0})


def check_images(packet,duration):
    packet_frames_valid(packet,duration)
    for frame,digest in zip(packet['frames'],packet['frame_sha256'],strict=True):
        if sha256_file(output_path(frame['image_path']))!=digest:raise ValueError('Prepared image changed')


def call(backend,request,packets,cfg,mode,mock):
    if mode=='verify':
        packet,=packets;prompt,schema=verification_prompt(request,packet);rawframes=packet['frames']
    else:
        rawframes,candidates=joint_view(packets,request['duration_sec'],mode=='joint_reverse')
        prompt,schema,_=reader_prompt(request,'raw',candidates)
    frames=[VideoFrame.model_validate(f) for f in rawframes]
    result={'status':'error','mode':mode,'prompt_sha256':previous.digest(prompt),'mock':mock}
    before=len(backend.calls) if backend else 0
    try:
        if mock:
            value=({'state':'insufficient','answer':'abstain','evidence_frames':[],
                'missing_reason':'visual_detail','observed_fact':'Mock observation only.'} if mode=='verify'
                else {'answer':'A','candidate_id':packets[0]['candidate_id']})
        else:
            value=backend(prompt,frames,schema,{'input':'images','pixels':cfg['pixels']},request['duration_sec'])
        result['raw_prediction']=value
        value=(validate_verdict(value,packet,request['choices'],request['duration_sec']) if mode=='verify'
               else bind_answer(value,candidates,request['choices'],request['duration_sec']))
        result.update(value,status='ok')
    except Exception as exc:
        result['error_type']=type(exc).__name__
    result['model_calls']=backend.calls[before:] if backend else []
    if mode=='verify':result['candidate_id']=packet['candidate_id']
    return result


def run(prepared_dir,config_path,output,model_path=None,mock=False):
    cfg=read(config_path);verify(prepared_dir,config_path,mock)
    rows=read(prepared_dir/'prepared.json');output=begin(output,config_path,mock,
        {'prepared_manifest_sha256':sha256_file(prepared_dir/'manifest.json')})
    for row in rows:
        previous.check_requests([row['request']])
        for packet in row['packets']:check_images(packet,row['request']['duration_sec'])
    backend,load_sec=previous.load_backend(cfg,model_path,output,mock)
    results=[]
    for row in rows:
        request=row['request'];packets=row['packets']
        result={'qid':request['qid'],'mock':mock}
        for mode in ['joint_forward','joint_reverse']:
            result[mode]=call(backend,request,packets,cfg,mode,mock)
        result['verdicts']=[call(backend,request,[packet],cfg,'verify',mock) for packet in packets]
        result['packet_verify']=aggregate(packets,result['verdicts'])
        action=next_action(packets,result['verdicts'],cfg['max_observe_actions']);result['action']=action
        result['refinement']=None;updated=result['verdicts'][:]
        if action['action']=='observe_more':
            packet=next(p for p in packets if p['candidate_id']==action['candidate_id'])
            refinement={'candidate_id':packet['candidate_id'],'status':'error'}
            try:
                asset=asset_from_probe(request['video_id'],request['video_path'])
                if asset.sha256!=row['video_sha256']:raise ValueError('Video bytes changed')
                extra,cost=observe(asset,packet,cfg['extra_frames'],cfg)
                refinement['cost']=cost
                pairs=sorted(zip(packet['frames']+extra['frames'],packet['frame_sha256']+extra['frame_sha256']),
                             key=lambda p:p[0]['timestamp_sec'])
                combined={**packet,'frames':[f for f,h in pairs],'frame_sha256':[h for f,h in pairs]}
                packet_frames_valid(combined,request['duration_sec'])
                refinement['packet']=combined;refinement['verdict']=call(backend,request,[combined],cfg,'verify',mock)
                refinement['status']='ok'
                updated=[refinement['verdict'] if v['candidate_id']==packet['candidate_id'] else v for v in updated]
            except Exception as exc:refinement['error_type']=type(exc).__name__
            result['refinement']=refinement
        result['packet_observe']=aggregate(packets,updated)
        result['final_action']=next_action(packets,updated,0)
        results.append(result);append(output/'records.jsonl',result)
        if backend and len(backend.calls)>cfg['max_generations']:raise ValueError('Call budget exceeded')
        print(json.dumps({'stage':'run','finished':len(results),'action':action['action'],
                         'calls':len(backend.calls) if backend else 0}),flush=True)
    return finish(output,['records.jsonl','model.json'],{'mock':mock,'queries':len(results),
        'backend_calls':len(backend.calls) if backend else 0,'model_load_sec':load_sec,
        'refinement_attempts':sum(r['refinement'] is not None for r in results)})


def all_calls(rows):
    for row in rows:
        predictions=[row['joint_forward'],row['joint_reverse'],*row['verdicts']]
        if row['refinement'] and 'verdict' in row['refinement']:predictions.append(row['refinement']['verdict'])
        for p in predictions:yield from p['model_calls']


def evaluate(requests_path,labels_path,prepared_dir,run_dir,config_path,output):
    cfg=read(config_path);requests=requests_for(requests_path,cfg)
    verify(prepared_dir,config_path,False);verify(run_dir,config_path,False)
    labels=previous.checked(labels_path,cfg['labels_sha256']);lookup={r['qid']:r for r in labels}
    if len(lookup)!=len(labels) or set(lookup)!={r['qid'] for r in requests}:raise ValueError('Label identities mismatch')
    rows=lines(run_dir/'records.jsonl');index={r['qid']:r for r in rows}
    prepared=read(prepared_dir/'prepared.json');packets={r['request']['qid']:r['packets'] for r in prepared}
    if len(rows)!=len(requests) or set(index)!=set(lookup):raise ValueError('Incomplete result denominator')
    metrics={};details={}
    for mode in ['joint_forward','joint_reverse','packet_verify','packet_observe']:
        scored=[{'qid':q['qid'],**previous.score_bound(index[q['qid']][mode],lookup[q['qid']],q['duration_sec'])}
                for q in requests]
        details[mode]=scored;metrics[mode]={'n':len(scored),**{k:sum(r[k] for r in scored)/len(scored)
                         for k in scored[0] if k!='qid'}}
        accepted=sum(r['accepted'] for r in scored)
        metrics[mode]['accepted_accuracy']=sum(r['correct'] for r in scored)/accepted if accepted else None
    observations=[]
    for request in requests:
        q=request['qid'];gt=lookup[q]['gt_intervals_sec'];duration=request['duration_sec']
        ps=packets[q];ious=[interval_set_iou([p['interval_sec']],gt,duration) for p in ps]
        observations.append({'qid':q,'candidate_count':len(ps),'best_candidate_iou':max(ious,default=0.),
            'candidate_r05':int(max(ious,default=0.)>=.5),
            'displayed_gt_hit':int(any(a<=f['timestamp_sec']<b for p in ps for f in p['frames'] for a,b in gt))})
    calls=list(all_calls(rows));refinements=[r['refinement'] for r in rows if r['refinement']]
    result={'scope':'Eight reused development sources; changed candidate pool/sampling/input vs prior round.',
        'metrics':metrics,'details':details,'observations':observations,
        'candidate_r05':sum(r['candidate_r05'] for r in observations)/len(requests),
        'displayed_gt_hit':sum(r['displayed_gt_hit'] for r in observations)/len(requests),
        'order_changed_answer':sum(r['joint_forward'].get('answer')!=r['joint_reverse'].get('answer') for r in rows),
        'order_changed_candidate':sum(r['joint_forward'].get('candidate_id')!=r['joint_reverse'].get('candidate_id') for r in rows),
        'verifier_errors':sum(v['status']!='ok' for r in rows for v in r['verdicts']),
        'state_counts':{s:sum(v['status']=='ok' and v['state']==s for r in rows for v in r['verdicts'])
                        for s in ['supported','contradicted','insufficient']},
        'refinement_attempts':len(refinements),'refinement_errors':sum(r['status']!='ok' for r in refinements),
        'costs':{'backend_calls':len(calls),**{k:sum(c.get(k,0) for c in calls)
             for k in ['input_tokens','output_tokens','images','elapsed_sec']},
             'peak_allocated_bytes':max((c.get('peak_allocated_bytes',0) for c in calls),default=0),
             'model_load_sec':read(run_dir/'summary.json')['model_load_sec'],
             'base_new_frames':read(prepared_dir/'summary.json')['new_frames_extracted'],
             'extra_new_frames':sum(r.get('cost',{}).get('new_frames_extracted',0) for r in refinements)}}
    output=begin(output,config_path,False);write_json(output/'evaluation.json',result)
    finish(output,['evaluation.json'],{'completed':True,'new_model_calls':0})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','evaluate'])
    for name in ['requests','retrieval','config','output','prepared-dir','run-dir','model-path','labels']:
        p.add_argument('--'+name,type=Path,required=name in ['config','output'])
    p.add_argument('--mock',action='store_true');a=p.parse_args()
    if a.action!='evaluate' and a.labels is not None:p.error('Inference/preparation reject labels')
    if a.action=='prepare':r=prepare(a.requests,a.retrieval,a.config,a.output,a.mock)
    elif a.action=='run':r=run(a.prepared_dir,a.config,a.output,a.model_path,a.mock)
    else:r=evaluate(a.requests,a.labels,a.prepared_dir,a.run_dir,a.config,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ['costs','details','observations']}))


if __name__=='__main__':main()
