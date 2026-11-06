"""One-shot post-hoc wording control, no VLM calls or new image encodings."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.event_composition_probe import phase_diagnosis
from scripts.mechanism_probe import snapshot
from scripts.sampling_probe import checked_output
from scripts.state_fact_bridge import locate_source_frames
from queryseek.embeddings.cache import load_embeddings
from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.event_composition import ordered_event_margin
from queryseek.tools.export import new_job_id


def diagnose_vectors(vectors, event_vectors, times, chain):
    if (vectors.ndim != 2 or event_vectors.shape != (2, vectors.shape[1])
            or len(vectors) != len(times) or not np.isfinite(vectors).all() or not np.isfinite(event_vectors).all()):
        raise ValueError('Finite aligned two-state vectors required')
    scores = vectors.astype(np.float64) @ event_vectors.astype(np.float64).T
    return {'order': ordered_event_margin(scores), 'phases': phase_diagnosis(scores, times, chain)}


def main():
    import torch
    torch.set_num_threads(4)
    manifest = json.loads((ROOT/'configs/state_minimal_pairs_v1.json').read_text(encoding='utf-8'))
    image_run, fact_run = ROOT/manifest['image_source'], ROOT/manifest['fact_source']
    image_rows = {r['qid']: r for r in (json.loads(line) for line in (image_run/'records.jsonl').read_text().splitlines())}
    fact_rows = [json.loads(line) for line in (fact_run/'records.jsonl').read_text().splitlines()]
    if len(fact_rows) != 8 or json.loads((fact_run/'metrics.json').read_text())['coverage'] != 1.:
        raise ValueError('Completed frozen eight-group fact experiment required')
    out = checked_output(ROOT/'outputs/state_minimal_pairs'/new_job_id())
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'manifest': manifest, 'oracle_only': True, 'independent': False,
                'fresh_selection_sha256': sha256_file(ROOT/'data/qvhighlights_fresh80/selection.json'),
                'source_hashes': {f'{directory.name}/{name}': sha256_file(directory/name)
                                  for directory in (image_run, fact_run) for name in ('protocol.json','records.jsonl','metrics.json')}}
    protocol['source_sha256'] = snapshot(out, [ROOT/'scripts/state_minimal_pair_probe.py',
        ROOT/'configs/state_minimal_pairs_v1.json', ROOT/'queryseek/retrieval/event_composition.py',
        ROOT/'scripts/event_composition_probe.py', ROOT/'queryseek/embeddings/openclip_encoder.py'])
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    encoder = OpenCLIPEmbedding()
    records = []
    for pair in manifest['pairs']:
        row = image_rows[pair['qid']]
        event_vectors = encoder.encode_texts([pair['a'], pair['b']])
        frames = locate_source_frames(row)
        loaded = load_embeddings(Path(frames[0].image_path).parent.parent)
        if loaded is None or loaded[1]['model_id'] != encoder.model_id or loaded[1]['frame_ids'] != [f.frame_id for f in frames]:
            raise ValueError('Source embeddings must match model and source frame identities')
        image_diagnosis = diagnose_vectors(loaded[0], event_vectors, row['actual_times_sec'], row['chain'])
        records.append({'qid': pair['qid'], 'representation': 'image', 'group': 'original', **image_diagnosis})
        for facts in [r for r in fact_rows if r['qid'] == pair['qid']]:
            if facts['actual_times_sec'] != row['actual_times_sec']:
                raise ValueError('Facts/source observations mismatch')
            diagnosis = diagnose_vectors(encoder.encode_texts(facts['facts_source_order']), event_vectors,
                                         row['actual_times_sec'], row['chain'])
            records.append({'qid': pair['qid'], 'representation': 'facts', 'group': facts['group'], **diagnosis})
    (out/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    report = {'completed': len(records), 'expected': 10, 'new_vlm_calls': 0, 'new_image_encodings': 0,
              'post_hoc': True, 'independent': False, 'counts': {}}
    for label in ('image','facts'):
        rows = [r for r in records if r['representation'] == label]
        report['counts'][label] = {'n': len(rows), 'order_positive': sum(r['order']['margin'] > 1e-12 for r in rows),
                                  'both_phase_positive': sum(r['phases']['a_vs_b_in_a_phase'] > 1e-12
                                                             and r['phases']['b_vs_a_in_b_phase'] > 1e-12 for r in rows)}
    (out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if sha256_file(ROOT/'data/qvhighlights_fresh80/selection.json') != protocol['fresh_selection_sha256']:
        raise RuntimeError('Independent selection changed')
    print('OUTPUT='+str(out), flush=True)
    for r in records:
        print(r['qid'],r['representation'],r['group'],r['order']['margin'],r['phases'], flush=True)


if __name__ == '__main__':
    main()
