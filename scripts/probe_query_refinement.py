"""Budget-matched query-guided retrieval, followed by a separate oracle diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from queryseek.agent.tree_observations import MockBoundaryEncoder, SamplingConfig, WindowObservationStore, output_path, write_json
from queryseek.embeddings.base import MockEmbedding
from queryseek.eval.grounded_qa import GroundedQASample, checked_intervals, evaluate_predictions, interval_set_iou, merged_intervals
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from queryseek.retrieval.query_refinement import METHODS, retrieve
from scripts import grounded_qa_input_study as study
from scripts.run_grounded_qa_baseline import check_requests


class MockQueryEncoder(MockBoundaryEncoder):
    def encode_texts(self,texts):
        return MockEmbedding().encode_texts(texts)


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def load_config(path):
    config=read(path); r=config['retrieval']
    if (r['dense_frames']!=r['root_frames']+r['selected_regions']*r['local_frames']
            or r['selected_regions']!=2 or r['review_frames']%2 or r['review_frames']<4
            or [c['name'] for c in config['conditions']]!=[*METHODS,'oracle_context16']
            or any(c['frames']!=r['review_frames'] for c in config['conditions'])):
        raise ValueError('Unmatched observation/review budgets')
    return config


def checked_requests(path,config):
    if sha256_file(path)!=config['requests_sha256']:
        raise ValueError('Frozen request hash mismatch')
    requests=check_requests(read(path))
    if len(requests)*len(config['conditions'])!=config['max_generations']:
        raise ValueError('Frozen generation budget mismatch')
    return requests


def encoder_for(config,clip_manifest,weights,mock):
    if mock:
        return MockQueryEncoder(),{'mock':True,'version':'query-refinement-mock-v1'},None
    if sha256_file(clip_manifest)!=config['clip_manifest_sha256']:
        raise ValueError('Frozen CLIP manifest mismatch')
    signature=read(clip_manifest)['encoder']
    if signature['mock'] or sha256_file(weights)!=signature['weights_sha256']:
        raise ValueError('Frozen real CLIP weights required')
    for name,version in signature['packages'].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('CLIP environment differs from frozen cache identity')
    if sha256_file(ROOT/'queryseek/embeddings/openclip_encoder.py')!=signature['adapter_sha256']:
        raise ValueError('CLIP adapter changed')
    import torch
    from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
    torch.manual_seed(config['seed']);torch.set_num_threads(8)
    if torch.cuda.get_device_name()!=signature['gpu']:
        raise ValueError('GPU identity changed')
    torch.cuda.reset_peak_memory_stats()
    encoder=OpenCLIPEmbedding(model_name=signature['model'],pretrained=str(weights),device=signature['device'],
        max_side=signature['image_max_side'],force_quick_gelu=signature['force_quick_gelu'])
    return encoder,signature,torch


def source_hashes():
    names=['scripts/probe_query_refinement.py','queryseek/retrieval/query_refinement.py',
        'queryseek/retrieval/curve_spans.py','queryseek/agent/tree_observations.py',
        'queryseek/agent/temporal_tree.py','queryseek/ingest/frames.py',
        'scripts/grounded_qa_input_study.py','scripts/train_agent_sft.py',
        'scripts/gpu_agent_experiment.py','queryseek/embeddings/openclip_encoder.py',
        'queryseek/eval/grounded_qa.py']
    return {n:sha256_file(ROOT/n) for n in names}


def prepare(requests_path,config_path,output,clip_manifest=None,weights=None,mock=False):
    config=load_config(config_path); requests=checked_requests(requests_path,config)
    output=output_path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'manifest.initial.json',{'config_sha256':sha256_file(config_path),
        'requests_sha256':sha256_file(requests_path),'source_sha256':source_hashes(),'mock':mock})
    began=time.perf_counter()
    encoder,signature,torch=encoder_for(config,clip_manifest,weights,mock)
    load_sec=time.perf_counter()-began
    text_start=time.perf_counter()
    text_vectors=encoder.encode_texts([r['question'] for r in requests])
    if len(text_vectors)!=len(requests):raise ValueError('Incomplete text encoding')
    text_sec=time.perf_counter()-text_start
    prepared,records,physical_costs=[],[],[]
    for request,query in zip(requests,text_vectors):
        started=time.perf_counter();rows=[]
        try:
            asset=asset_from_probe(request['video_id'],request['video_path'])
            if abs(asset.duration_sec-request['duration_sec'])>.001:
                raise ValueError('Probed duration changed')
            def observe(a,b,budget):
                provider=WindowObservationStore(asset.path,asset.duration_sec,asset.sha256,encoder,signature,
                    replace(SamplingConfig(**config['sampling']),max_frames=budget))
                observation=provider(a,b)
                physical_costs.append({'qid':request['qid'],'window_sec':[a,b],**observation.cost})
                return observation
            root=observe(0.,asset.duration_sec,config['retrieval']['root_frames'])
            token_audit={'raw_tokens':None,'context_length':None,'truncated':None}
            if not mock:
                tokens=len(encoder._tokenizer.encode(request['question']))+2
                context=encoder._tokenizer.context_length
                token_audit={'raw_tokens':tokens,'context_length':context,'truncated':tokens>context}
            for method in METHODS:
                record,frames=retrieve(method,asset.duration_sec,query,observe,config['retrieval'],root)
                record.update(qid=request['qid'],status='ok',mock=mock,token_audit=token_audit,
                              coarse_source_times_sec=[f.timestamp_sec for f in root.frames])
                rows.append(record)
                prepared.append({'request':request,'condition':method,'frames':[f.model_dump() for f in frames],
                    'frame_sha256':[sha256_file(f.image_path) for f in frames],
                    'preprocessing':{'charged_image_observations':record['charged_image_observations'],
                                     'unique_observations':record['unique_observations'],'vlm_calls':0}})
                print(json.dumps({'qid':request['qid'],'method':method,'status':'ok',
                    'unique_observations':record['unique_observations']}),flush=True)
        except Exception as exc:
            completed={r['method'] for r in rows}
            for method in METHODS:
                if method not in completed:
                    rows.append({'qid':request['qid'],'method':method,'status':'error','mock':mock,
                        'error_type':type(exc).__name__,
                        'error_locations':[{'file':Path(f.filename).name,'line':f.lineno}
                                           for f in traceback.extract_tb(exc.__traceback__)]})
        records.extend(rows)
        with (output/'retrieval.jsonl').open('a',encoding='utf-8') as handle:
            for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps({'qid':request['qid'],'elapsed_sec':time.perf_counter()-started,
                          'completed':sum(r['status']=='ok' for r in rows)}),flush=True)
    write_json(output/'prepared.json',prepared)
    summary={'scheduled':len(requests)*len(METHODS),'completed':sum(r['status']=='ok' for r in records),
        'failed':sum(r['status']!='ok' for r in records),'mock':mock,'model_load_sec':load_sec,
        'text_encoding_sec':text_sec,'encoded_texts':len(requests),'new_vlm_calls':0,
        'physical_completed_observations':physical_costs,
        'new_encoded_images':sum(c['encoded_images'] for c in physical_costs),
        'new_frames_extracted':sum(c['new_frames_extracted'] for c in physical_costs),
        'cost_accounting_complete':all(r['status']=='ok' for r in records),
        'elapsed_sec':time.perf_counter()-began,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch else None}
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{**read(output/'manifest.initial.json'),
        'prepared_sha256':sha256_file(output/'prepared.json'),
        'retrieval_sha256':sha256_file(output/'retrieval.jsonl'),'encoder':signature})
    return summary


def prepare_context(prepared_dir,labels_path,config_path,output):
    config=load_config(config_path);manifest=read(prepared_dir/'manifest.json')
    if (manifest['config_sha256']!=sha256_file(config_path)
            or manifest['prepared_sha256']!=sha256_file(prepared_dir/'prepared.json')
            or manifest['retrieval_sha256']!=sha256_file(prepared_dir/'retrieval.jsonl')
            or read(prepared_dir/'summary.json')['failed']):
        raise ValueError('Label-free retrieval must finish and freeze before oracle preparation')
    if sha256_file(labels_path)!=config['labels_sha256']:
        raise ValueError('Frozen labels hash mismatch')
    labels=read(labels_path)
    if len({r['qid'] for r in labels})!=len(labels):raise ValueError('Duplicate labels')
    lookup={r['qid']:r for r in labels}
    rows=read(prepared_dir/'prepared.json')
    requests={r['request']['qid']:r['request'] for r in rows}
    if set(requests)!=set(lookup):raise ValueError('Label/request identity mismatch')
    output=output_path(output);output.mkdir(parents=True,exist_ok=False)
    costs=[]
    for request in requests.values():
        pad=config['oracle_context_padding_sec'];duration=request['duration_sec']
        truth=checked_intervals(lookup[request['qid']]['gt_intervals_sec'],duration)
        intervals=merged_intervals([(max(0.,a-pad),min(duration,b+pad)) for a,b in truth])
        frames,cost=study.oracle_frames(request,intervals,config['retrieval']['review_frames'])
        costs.append(cost)
        rows.append({'request':request,'condition':'oracle_context16','frames':[f.model_dump() for f in frames],
                     'frame_sha256':[sha256_file(f.image_path) for f in frames],'preprocessing':cost})
    study.validate_prepared(rows,config)
    write_json(output/'prepared.json',rows)
    write_json(output/'manifest.json',{'config_sha256':sha256_file(config_path),
        'prepared_sha256':sha256_file(output/'prepared.json'),'retrieval_manifest_sha256':sha256_file(prepared_dir/'manifest.json'),
        'labels_sha256':sha256_file(labels_path),'rows':len(rows),'mock':manifest['mock'],
        'oracle_preprocessing_costs':costs,'source_sha256':source_hashes()})
    return {'rows':len(rows),'new_vlm_calls':0,'mock':manifest['mock']}


def run(prepared_dir,config_path,output,model_path=None,mock=False):
    manifest=read(prepared_dir/'manifest.json')
    if (manifest['mock']!=mock or manifest['config_sha256']!=sha256_file(config_path)
            or manifest['prepared_sha256']!=sha256_file(prepared_dir/'prepared.json')
            or manifest['source_sha256']!=source_hashes()):
        raise ValueError('Prepared provenance/mock mode changed')
    return study.run(prepared_dir/'prepared.json',config_path,output,model_path,mock)


def evaluate(requests_path,labels_path,config_path,retrieval_dir,run_dir,output):
    config=load_config(config_path);requests=checked_requests(requests_path,config)
    if sha256_file(labels_path)!=config['labels_sha256']:raise ValueError('Frozen labels mismatch')
    labels=read(labels_path)
    if len({r['qid'] for r in labels})!=len(labels):raise ValueError('Duplicate labels')
    lookup={r['qid']:r for r in labels}
    if set(lookup)!={r['qid'] for r in requests}:raise ValueError('Mismatched label ids')
    manifest=read(retrieval_dir/'manifest.json')
    if (manifest['mock'] or read(run_dir/'run_summary.json')['mock']
            or manifest['config_sha256']!=sha256_file(config_path)
            or manifest['retrieval_sha256']!=sha256_file(retrieval_dir/'retrieval.jsonl')):
        raise ValueError('Real frozen records required')
    rows=[json.loads(s) for s in (retrieval_dir/'retrieval.jsonl').read_text('utf-8').splitlines()]
    expected={(r['qid'],m) for r in requests for m in METHODS}
    if len(rows)!=len(expected) or {(r['qid'],r['method']) for r in rows}!=expected:
        raise ValueError('Incomplete retrieval denominator')
    index={(r['qid'],r['method']):r for r in rows}
    predictions=[json.loads(s) for s in (run_dir/'predictions.jsonl').read_text('utf-8').splitlines()]
    expected={(r['qid'],c['name']) for r in requests for c in config['conditions']}
    if len(predictions)!=len(expected) or {(p['qid'],p['condition']) for p in predictions}!=expected:
        raise ValueError('Incomplete generation denominator')
    samples=[GroundedQASample(r['qid'],r['video_id'],r['question'],tuple(r['choices']),r['duration_sec'],
        lookup[r['qid']]['right_answer'],tuple(map(tuple,lookup[r['qid']]['gt_intervals_sec']))) for r in requests]
    result={'n':len(requests),'methods':{},'oracle':{},'rows':{},'scope':'Development diagnostics, no independent confirmation or new training'}
    for method in METHODS:
        scored=[]
        for request in requests:
            row=index[(request['qid'],method)];gt=lookup[request['qid']]['gt_intervals_sec'];duration=request['duration_sec']
            item={'qid':request['qid'],'status':row['status'],'coarse_hit':0,'observed_hit':0,'review_hit':0,'rank1_iou':0.,'top5_best_iou':0.,'top5_r05':0}
            if row['status']=='ok':
                for source,target in [('coarse_source_times_sec','coarse_hit'),('observed_source_times_sec','observed_hit'),('review_source_times_sec','review_hit')]:
                    item[target]=int(any(a<=t<b for t in row[source] for a,b in gt))
                ious=[interval_set_iou([[w['start_sec'],w['end_sec']]],gt,duration) for w in row['windows']]
                item.update(rank1_iou=ious[0] if ious else 0.,top5_best_iou=max(ious,default=0.),top5_r05=int(max(ious,default=0.)>=.5))
            scored.append(item)
        qa=evaluate_predictions(samples,[p for p in predictions if p['condition']==method])
        result['methods'][method]={k:sum(r[k] for r in scored)/len(scored)
                                  for k in ['coarse_hit','observed_hit','review_hit','rank1_iou','top5_best_iou','top5_r05']}
        result['methods'][method]['qa']=qa['metrics']
        result['rows'][method]={'retrieval':scored,'qa':qa['records']}
    oracle=[p for p in predictions if p['condition']=='oracle_context16']
    result['oracle']={'n':len(oracle),'qa_correct':sum(p['status']=='ok' and p.get('answer')==lookup[p['qid']]['right_answer'] for p in oracle)/len(requests),
                      'correct_ids':[p['qid'] for p in oracle if p['status']=='ok' and p.get('answer')==lookup[p['qid']]['right_answer']]}
    calls=[c for p in predictions for c in p['model_calls']]
    result['costs']={'backend_calls':len(calls),
        **{k:sum(c.get(k,0) for c in calls) for k in ['input_tokens','output_tokens','images','elapsed_sec']},
        'peak_allocated_bytes':max((c.get('peak_allocated_bytes',0) for c in calls),default=0),
        'model_load_sec':read(run_dir/'run_summary.json')['model_load_sec']}
    result['source_sha256']={str(p.name):sha256_file(p) for p in [requests_path,labels_path,config_path,retrieval_dir/'retrieval.jsonl',run_dir/'predictions.jsonl']}
    output=output_path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'evaluation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','context','run','evaluate'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for name in ['requests','labels','clip-manifest','weights','prepared-dir','retrieval-dir','run-dir','model-path']:
        p.add_argument('--'+name,type=Path)
    p.add_argument('--mock',action='store_true');a=p.parse_args()
    if a.action=='prepare':
        if a.labels is not None:p.error('Label-free preparation rejects labels')
        result=prepare(a.requests,a.config,a.output,a.clip_manifest,a.weights,a.mock)
        print(json.dumps({k:v for k,v in result.items() if k!='physical_completed_observations'}))
        if result['failed']:raise SystemExit(2)
    elif a.action=='context':print(json.dumps(prepare_context(a.prepared_dir,a.labels,a.config,a.output)))
    elif a.action=='run':
        if a.labels is not None or a.requests is not None:p.error('Inference accepts prepared observations only')
        result=run(a.prepared_dir,a.config,a.output,a.model_path,a.mock)
        print(json.dumps(result))
        if result['failed']:raise SystemExit(2)
    else:
        result=evaluate(a.requests,a.labels,a.config,a.retrieval_dir,a.run_dir,a.output)
        print(json.dumps({k:v for k,v in result.items() if k not in ['rows','source_sha256']}))


if __name__=='__main__':main()
