"""Label-free preparation-only repair; preserves original experiment and does not invoke models."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import probe_evidence_packets as p
from queryseek.retrieval.anchor_observations import retain_seed


def repair(prepared_path,retrieval_path,config_path,cache_root,output):
    cfg=p.read(config_path)
    if p.sha256_file(prepared_path)!=cfg['prepared_sha256']:raise ValueError('Prepared input changed')
    if p.sha256_file(retrieval_path)!=cfg['retrieval_sha256']:raise ValueError('Frozen retrieval changed')
    rows=p.read(prepared_path);lookup={r['qid']:r for r in p.lines(retrieval_path) if r['method']=='dense_uniform'}
    caches={p.sha256_file(m):m for m in Path(cache_root).glob('*/manifest.json')}
    output=p.output_path(output);output.mkdir(parents=True,exist_ok=False)
    result=[];audit=[]
    for row in rows:
        q=row['request']['qid'];request=p.previous.check_requests([row['request']])[0]
        manifest=caches[lookup[q]['observation_costs'][0]['frame_manifest_sha256']]
        m=p.read(manifest);index=manifest.parent/'frames.jsonl'
        if m['signature']['video_sha256']!=row['video_sha256'] or p.sha256_file(index)!=m['index_sha256']:
            raise ValueError('Seed source identity mismatch')
        observations=list(zip(p.lines(index),m['image_sha256'],strict=True));packets=[]
        for packet in row['packets']:
            p.check_images(packet,request['duration_sec'])
            if packet['origin']!='seed_window':packets.append(packet);continue
            matches=[(f,h) for f,h in observations if abs(f['timestamp_sec']-packet['seed_time_sec'])<=1e-6]
            if len(matches)!=1:raise ValueError('Ambiguous/missing seed identity')
            frame,digest=matches[0]
            if p.sha256_file(p.output_path(frame['image_path']))!=digest:raise ValueError('Seed image changed')
            repaired,record=retain_seed(packet,frame,digest,request['duration_sec']);packets.append(repaired)
            audit.append({'qid':q,'candidate_id':packet['candidate_id'],'original_seed_gap_sec':
                min(abs(f['timestamp_sec']-packet['seed_time_sec']) for f in packet['frames']),
                'seed_manifest_sha256':p.sha256_file(manifest),**record})
        result.append({**row,'packets':packets})
    p.write_json(output/'prepared.json',result);p.write_json(output/'audit.private.json',audit)
    summary={'scope':'Posthoc engineering repair; no new model inference or accuracy measurement.',
        'seed_windows':len(audit),'anchors_preserved':sum(any(f['timestamp_sec']==packet['seed_time_sec']
            for f in packet['frames']) for r in result for packet in r['packets'] if packet['origin']=='seed_window'),
        'original_seed_gap_sec_min':min((r['original_seed_gap_sec'] for r in audit),default=None),
        'original_seed_gap_sec_max':max((r['original_seed_gap_sec'] for r in audit),default=None),
        'prepared_frames':sum(len(packet['frames']) for r in result for packet in r['packets']),
        'new_frames_extracted':0,'new_model_calls':0,'qa_accuracy':None}
    p.write_json(output/'summary.json',summary)
    p.write_json(output/'manifest.json',{'scope':summary['scope'],'input_sha256':p.sha256_file(prepared_path),
        'config_sha256':p.sha256_file(config_path),'retrieval_sha256':p.sha256_file(retrieval_path),
        'source_sha256':{n:p.sha256_file(ROOT/n) for n in ['scripts/repair_packet_anchors.py','queryseek/retrieval/anchor_observations.py']},
        'files':{n:p.sha256_file(output/n) for n in ['prepared.json','audit.private.json','summary.json']}})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ['prepared','retrieval','config','cache-root','output']:parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args();print(json.dumps(repair(a.prepared,a.retrieval,a.config,a.cache_root,a.output)))
