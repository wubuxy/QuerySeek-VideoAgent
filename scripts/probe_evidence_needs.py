"""Frozen shared-state comparison of rank-first and evidence-directed observation."""
from pathlib import Path
from copy import deepcopy
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import probe_evidence_packets as p
from queryseek import PIPELINE_VERSION
from queryseek.schemas import VideoFrame
from queryseek.retrieval.evidence_packets import aggregate
from queryseek.retrieval.evidence_needs import (REGIONS, plan_prompt, validate_plan, verdict_prompt,
    decode_verdict, views, select_action, context_interval)


def sources():
    return {**p.sources(),**{n:p.sha256_file(ROOT/n) for n in
        ['queryseek/retrieval/evidence_needs.py','scripts/probe_evidence_needs.py']}}


def begin(output,config,mock,dependencies=None):
    output=p.output_path(output);output.mkdir(parents=True,exist_ok=False)
    p.write_json(output/'manifest.initial.json',{'source_sha256':sources(),
        'config_sha256':p.sha256_file(config),'mock':mock,'dependencies':dependencies or {}})
    return output


def verify(folder,config,mock):
    m=p.read(folder/'manifest.json')
    if m['source_sha256']!=sources() or m['config_sha256']!=p.sha256_file(config) or m['mock']!=mock:
        raise ValueError('Study provenance/mock mismatch')
    for n,d in m['files'].items():
        if p.sha256_file(folder/n)!=d:raise ValueError('Study artifact changed')
    return m


def check_views(packet,duration):
    p.check_images(packet,duration);views(packet,duration)
    for v in packet.get('detail_views',[]):
        if p.sha256_file(p.output_path(v['frame']['image_path']))!=v['sha256']:
            raise ValueError('Detail pixels changed')


def call(backend,request,packet,plan,cfg,mock=False):
    prompt,schema=(plan_prompt(request['question']) if packet is None else verdict_prompt(request,packet,plan))
    before=len(backend.calls) if backend else 0
    if backend and before>=cfg['max_generations']:raise ValueError('Generation budget exhausted')
    result={'status':'error','mock':mock,'prompt_sha256':p.previous.digest(prompt),
            'schema_sha256':p.previous.digest(json.dumps(schema,sort_keys=True))}
    if packet:result['candidate_id']=packet['candidate_id']
    try:
        if mock:
            value=({'target':'Requested object','relation':'Requested context','scope':'single_view'} if packet is None
                else {'decision':'target_unobserved','evidence':'','region':'full',
                      'target_description':'Not observed.','observed_fact':'Mock observation only.'})
        else:
            value=backend(prompt,[] if packet is None else [VideoFrame.model_validate(f)
                          for f in views(packet,request['duration_sec'])],schema,
                          {'input':'text' if packet is None else 'images','pixels':cfg['pixels']},request['duration_sec'])
        result['raw_prediction']=value
        if packet is None:result['plan']=validate_plan(value)
        else:result.update(decode_verdict(value,request,packet,plan))
        result['status']='ok'
    except Exception as exc:result['error_type']=type(exc).__name__
    result['model_calls']=backend.calls[before:] if backend else []
    return result


def observe_action(request,packet,action,video_sha256,cfg):
    """Reuse source-PTS extraction or lossless crops of existing observed images."""
    started=time.perf_counter();check_views(packet,request['duration_sec'])
    if action['candidate_id']!=packet['candidate_id']:raise ValueError('Action/candidate mismatch')
    if action['action']=='detail':
        from PIL import Image,__version__ as pillow_version
        i=action['anchor_index'];region=action['region']
        if type(i) is not int or not 0<=i<len(packet['frames']) or region not in REGIONS or region=='full':
            raise ValueError('Invalid detail action')
        parent=packet['frames'][i];digest=packet['frame_sha256'][i]
        signature={'video_sha256':video_sha256,'parent_sha256':digest,'source_time_sec':parent['timestamp_sec'],
            'pipeline_version':PIPELINE_VERSION,'region':region,'box_xyxy':list(REGIONS[region]),
            'pillow_version':pillow_version,'encoding':'PNG','source_sha256':sources(),
            'version':'evidence-needs-detail-v1'}
        cache=p.output_path(p.cache_paths.ROOT/'outputs/evidence_detail_cache'/p.previous.digest(json.dumps(signature,sort_keys=True)))
        hit=(cache/'manifest.json').is_file();dest=cache/'detail.png'
        if hit:
            m=p.read(cache/'manifest.json')
            if m['signature']!=signature or p.sha256_file(dest)!=m['image_sha256']:raise ValueError('Detail cache mismatch')
        else:
            cache.mkdir(parents=True,exist_ok=False)
            with Image.open(p.output_path(parent['image_path'])) as im:
                rgb=im.convert('RGB');w,h=rgb.size
                a,b,c,d=REGIONS[region];box=(int(a*w),int(b*h),int(c*w),int(d*h))
                rgb.crop(box).save(dest,format='PNG')
            m={'signature':signature,'pixel_box_xyxy':box,'image_sha256':p.sha256_file(dest)}
            p.write_json(cache/'manifest.json',m)
        extra={'frame':{**parent,'frame_id':packet['candidate_id']+'_detail','image_path':str(dest)},
               'parent_index':i,'parent_sha256':digest,'region':region,'sha256':m['image_sha256']}
        observed={**deepcopy(packet),'detail_views':[extra]};check_views(observed,request['duration_sec'])
        return observed,{'cache_hit':hit,'new_frames_extracted':0,'new_crops_created':0 if hit else 1,
            'elapsed_sec':time.perf_counter()-started,'manifest_path':str(cache/'manifest.json'),
            'manifest_sha256':p.sha256_file(cache/'manifest.json')}
    if action['action'] not in ('uniform','context'):raise ValueError('Unsupported observation')
    asset=p.asset_from_probe(request['video_id'],request['video_path'])
    if asset.sha256!=video_sha256:raise ValueError('Video changed')
    window=packet['interval_sec']
    if action['action']=='context':
        i=action['anchor_index']
        if type(i) is not int or not 0<=i<len(packet['frames']):raise ValueError('Invalid context anchor')
        window=context_interval(packet['frames'][i]['timestamp_sec'],request['duration_sec'],
                                cfg['context_before_sec'],cfg['context_after_sec'])
    observed,cost=p.observe(asset,{**packet,'interval_sec':window},cfg['extra_frames'],cfg)
    by_time={f['timestamp_sec']:(f,h) for f,h in zip(packet['frames'],packet['frame_sha256'],strict=True)}
    for f,h in zip(observed['frames'],observed['frame_sha256'],strict=True):by_time.setdefault(f['timestamp_sec'],(f,h))
    pairs=[by_time[t] for t in sorted(by_time)]
    merged={**deepcopy(packet),'original_interval_sec':packet['interval_sec'],
        'interval_sec':[min(window[0],packet['interval_sec'][0]),max(window[1],packet['interval_sec'][1])],
        'frames':[f for f,h in pairs],'frame_sha256':[h for f,h in pairs]}
    check_views(merged,request['duration_sec']);return merged,{**cost,'new_crops_created':0}


def run(prepared_path,requests_path,config,output,model_path=None,mock=False):
    cfg=p.read(config);rows=p.previous.checked(prepared_path,cfg['prepared_sha256'])
    requests=p.previous.check_requests(p.previous.checked(requests_path,cfg['requests_sha256']))
    if ([r['request'] for r in rows]!=requests or len(rows)!=cfg['sample_count']
        or sum(len(r['packets']) for r in rows)!=cfg['candidate_count']
        or cfg['max_generations']!=cfg['candidate_count']+3*len(rows)):
        raise ValueError('Incomplete frozen inputs or call budget')
    for row in rows:
        for packet in row['packets']:check_views(packet,row['request']['duration_sec'])
    output=begin(output,config,mock,{'prepared_sha256':cfg['prepared_sha256'],'requests_sha256':cfg['requests_sha256']})
    p.write_json(output/'prepared.json',rows)
    backend,load_sec=p.previous.load_backend(cfg,model_path,output,mock)
    plans={}
    for request in requests:
        plan=call(backend,request,None,None,cfg,mock);plans[request['qid']]=plan
        p.append(output/'plans.jsonl',{'qid':request['qid'],**plan})
    results=[]
    for row in rows:
        req=row['request'];packets=row['packets'];q=req['qid'];plan=plans[q]
        result={'qid':q,'mock':mock,'verdicts':[],'branches':{}}
        if plan['status']=='ok':
            result['verdicts']=[call(backend,req,packet,plan['plan'],cfg,mock) for packet in packets]
        else:
            result['verdicts']=[{'candidate_id':x['candidate_id'],'status':'error','error_type':'PlanFailed','model_calls':[]} for x in packets]
        result['base']=aggregate(packets,result['verdicts']);seen={}
        for policy in ['rank_uniform','evidence_routed']:
            action=select_action(packets,result['verdicts'],1,policy);branch={'action':action,'observation':None}
            updated=result['verdicts'][:];updated_packets=packets[:]
            if action['action'] in ('uniform','context','detail'):
                key=json.dumps(action,sort_keys=True)
                if key in seen:
                    observation=deepcopy(seen[key]);observation['verdict']['model_calls']=[]
                    observation['reused_from']='rank_uniform';observation['cost']={
                        **observation['cost'],'cache_hit':True,'new_frames_extracted':0,'new_crops_created':0,'elapsed_sec':0.}
                else:
                    packet=next(x for x in packets if x['candidate_id']==action['candidate_id'])
                    observation={'status':'error','candidate_id':packet['candidate_id']}
                    try:
                        observed,cost=observe_action(req,packet,action,row['video_sha256'],cfg)
                        observation.update(packet=observed,cost=cost)
                        observation['verdict']=call(backend,req,observed,plan['plan'],cfg,mock)
                        observation['status']='ok';seen[key]=deepcopy(observation)
                    except Exception as exc:observation['error_type']=type(exc).__name__
                branch['observation']=observation
                if observation['status']=='ok':
                    updated=[observation['verdict'] if v['candidate_id']==action['candidate_id'] else v for v in updated]
                    updated_packets=[observation['packet'] if x['candidate_id']==action['candidate_id'] else x for x in packets]
            branch['result']=aggregate(updated_packets,updated)
            branch['final_action']=select_action(updated_packets,updated,0,policy)
            result['branches'][policy]=branch
        results.append(result);p.append(output/'records.jsonl',result)
        print(json.dumps({'finished':len(results),'new_calls':len(backend.calls) if backend else 0,
                         'actions':{k:v['action']['action'] for k,v in result['branches'].items()}}),flush=True)
    return p.finish(output,['plans.jsonl','records.jsonl','prepared.json','model.json'],
        {'mock':mock,'queries':len(rows),'backend_calls':len(backend.calls) if backend else 0,'model_load_sec':load_sec})


def evaluate(folder,requests_path,labels_path,config,output):
    cfg=p.read(config);verify(folder,config,False)
    requests=p.previous.check_requests(p.previous.checked(requests_path,cfg['requests_sha256']))
    labels=p.previous.checked(labels_path,cfg['labels_sha256']);lookup={r['qid']:r for r in labels}
    rows=p.lines(folder/'records.jsonl');index={r['qid']:r for r in rows};plans=p.lines(folder/'plans.jsonl')
    if (len(index)!=len(rows) or len(lookup)!=len(labels) or len(rows)!=cfg['sample_count']
        or set(index)!=set(lookup) or set(index)!={r['qid'] for r in requests}):
        raise ValueError('Full frozen denominator required')
    metrics={};details={}
    for mode in ['base','rank_uniform','evidence_routed']:
        scores=[{'qid':q['qid'],**p.previous.score_bound(index[q['qid']]['base'] if mode=='base' else
                  index[q['qid']]['branches'][mode]['result'],lookup[q['qid']],q['duration_sec'])} for q in requests]
        metrics[mode]={'n':len(scores),**{k:sum(s[k] for s in scores)/len(scores) for k in scores[0] if k!='qid'}}
        accepted=sum(s['accepted'] for s in scores)
        metrics[mode]['accepted_accuracy']=sum(s['correct'] for s in scores)/accepted if accepted else None
        details[mode]=scores
    calls=[c for x in plans for c in x['model_calls']];costs=[];actions={};errors={}
    for row in rows:
        calls.extend(c for v in row['verdicts'] for c in v['model_calls'])
        for policy,branch in row['branches'].items():
            actions.setdefault(policy,[]).append(branch['action']);o=branch['observation']
            if o:
                costs.append(o.get('cost',{}));calls.extend(o.get('verdict',{}).get('model_calls',[]))
                errors[policy]=errors.get(policy,0)+int(o['status']!='ok' or o.get('verdict',{}).get('status')!='ok')
    if len(calls)!=p.read(folder/'summary.json')['backend_calls']:raise ValueError('Call accounting mismatch')
    result={'scope':'Eight previously examined development sources; human feedback informed design, no human answers in inference.',
        'metrics':metrics,'details':details,'plan_errors':sum(x['status']!='ok' for x in plans),
        'initial_verdict_errors':sum(v['status']!='ok' for r in rows for v in r['verdicts']),
        'initial_states':{k:sum(v.get('decision')==k for r in rows for v in r['verdicts']) for k in
            ['target_unobserved','target_ambiguous','relation_unverified','attribute_unreadable',
             'temporal_unverified','global_unverified','contradicted']},
        'action_counts':{policy:{kind:sum(a['action']==kind for a in acts) for kind in
            ['answer','abstain','uniform','detail','context']} for policy,acts in actions.items()},
        'action_selection_changed':sum(r['branches']['rank_uniform']['action']!=r['branches']['evidence_routed']['action'] for r in rows),
        'branch_errors':errors,'costs':{'backend_calls':len(calls),**{k:sum(c.get(k,0) for c in calls) for k in
            ['input_tokens','output_tokens','visual_tokens','images','elapsed_sec']},
            'peak_allocated_bytes':max((c.get('peak_allocated_bytes',0) for c in calls),default=0),
            'model_load_sec':p.read(folder/'summary.json')['model_load_sec'],
            'new_frames_extracted':sum(c.get('new_frames_extracted',0) for c in costs),
            'new_crops_created':sum(c.get('new_crops_created',0) for c in costs)}}
    output=begin(output,config,False);p.write_json(output/'evaluation.json',result)
    p.finish(output,['evaluation.json'],{'completed':True,'new_model_calls':0});return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['run','evaluate'])
    for n in ['prepared','requests','config','output','folder','labels','model-path']:
        parser.add_argument('--'+n,type=Path,required=n in ['requests','config','output'])
    parser.add_argument('--mock',action='store_true');a=parser.parse_args()
    if a.action=='run':
        if a.labels is not None:parser.error('Inference rejects evaluator labels')
        r=run(a.prepared,a.requests,a.config,a.output,a.model_path,a.mock)
    else:r=evaluate(a.folder,a.requests,a.labels,a.config,a.output)
    print(json.dumps({k:v for k,v in r.items() if k!='details'}))
