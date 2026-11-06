"""Read-only pilot replay: no API/image encoding/training, private analysis output."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.eval.qvhighlights import eval_mr_brief, temporal_iou
from queryseek.retrieval.localizer import LocalizationHead, localization_loss
from scripts.localizer_pilot import example_tensors, prediction, read, rows
from scripts.sampling_probe import checked_output


def proposal_diagnostic(predictions, records):
    """Full planned record denominator, including empty predictions."""
    top1 = top5 = 0
    for record in records:
        qid = record['entry']['qid']
        windows = predictions.get(qid, [])
        gt = record['entry']['relevant_windows']
        matches = [max((temporal_iou(a, b, g[0], g[1]) for g in gt), default=0.)
                   for a, b, _ in windows]
        top1 += bool(matches and matches[0] >= .5)
        top5 += bool(matches and max(matches) >= .5)
    return {'queries': len(records), 'top1_iou05_hits': top1, 'top5_iou05_oracle_hits': top5}


def load_records(run):
    result = []
    for record in rows(run/'feature_records.jsonl'):
        folder = checked_output(ROOT/'outputs/localizer_feature_cache'/record['cache_key'])
        meta = read(folder/'feature_meta.json')
        if sha256_file(folder/'features.npz') != meta['feature_sha256']:
            raise ValueError('Feature replay SHA mismatch')
        signature = meta['signature']
        if signature['video_sha256'] != record['video_sha256']:
            raise ValueError('Feature replay identity mismatch')
        text_signature = {'query': record['entry']['query'], 'model_id': signature['model_id'],
                          'weights_sha256': signature['weights_sha256'],
                          'open_clip_version': signature['open_clip_version']}
        text_key = hashlib.sha256(json.dumps(text_signature, sort_keys=True).encode()).hexdigest()
        text_path = checked_output(ROOT/'outputs/localizer_text_cache'/(text_key+'.npy'))
        tm = read(text_path.with_suffix('.json'))
        if tm['signature'] != text_signature or sha256_file(text_path) != tm['sha256']:
            raise ValueError('Text replay SHA mismatch')
        with np.load(folder/'features.npz', allow_pickle=False) as data:
            result.append({**record, 'images': data['images'], 'times': data['times'],
                           'query_embedding': np.load(text_path, allow_pickle=False)})
    return result


def analyze(run):
    run = checked_output(Path(run)); protocol = read(run/'protocol.json')
    config = protocol['config']; metrics = read(run/'metrics.json')
    if metrics['mock']:
        raise ValueError('Real feature replay cannot treat mock metrics as real')
    torch.set_num_threads(config['torch_threads'])
    records = load_records(run)
    if len(records) != 32:
        raise ValueError('Incomplete planned feature denominator')
    original_hashes = {f.name: sha256_file(f) for f in run.iterdir() if f.is_file()}
    output = run/'analysis'; output.mkdir(exist_ok=True)
    report = {'training_run': str(run), 'new_vlm_calls': 0, 'new_image_encodings': 0,
              'training_updates': 0, 'source_hashes': original_hashes, 'heads': {}}
    for architecture in config['architectures']:
        path = run/(architecture+'.pt')
        if sha256_file(path) != metrics['heads'][architecture]['checkpoint_sha256']:
            raise ValueError('Checkpoint replay SHA mismatch')
        checkpoint = torch.load(path, weights_only=True)
        model = LocalizationHead(checkpoint['embedding_dim'], config['hidden'], architecture)
        model.load_state_dict(checkpoint['state_dict']); model.eval()
        results = {}; dense = {}; archived = read(run/(architecture+'_predictions.json'))['original']
        for split in ['train', 'validation']:
            subset = [r for r in records if r['split'] == split]
            predictions = {}; components = []; losses = []
            for r in subset:
                with torch.inference_mode():
                    x, y, d = example_tensors(r)
                    logits, distances = model(x)
                    total, terms = localization_loss(logits, distances, y, d, config['loss_weights'])
                    components.append(terms); losses.append(float(total))
                qid = r['entry']['qid']; spans, _ = prediction(model, r, config)
                predictions[qid] = spans
                if split == 'validation' and not np.allclose(np.asarray(spans), np.asarray(archived[str(qid)]), atol=1e-7):
                    raise ValueError('Archived prediction differs from checkpoint replay')
                dense[str(qid)+'_times'] = r['times']
                dense[str(qid)+'_logits'] = logits.cpu().numpy()
                dense[str(qid)+'_distances'] = distances.cpu().numpy()
            gt = {r['entry']['qid']: r['entry'] for r in subset}
            results[split] = {'loss': float(np.mean(losses)),
                 'loss_components': {name: float(np.mean([c[name] for c in components])) for name in components[0]},
                 'metrics': dict(eval_mr_brief(predictions, gt)), **proposal_diagnostic(predictions, subset)}
        np.savez(output/(architecture+'_dense_predictions.npz'), **dense)
        report['heads'][architecture] = results
    for name, checksum in original_hashes.items():
        if sha256_file(run/name) != checksum:
            raise ValueError('Original run changed during read-only analysis')
    report['interpretation'] = '32-source mechanical pilot; train fitting and 8-source internal validation are descriptive, not independent gains'
    write_json_atomic(output/'diagnostic.json', report)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run'); args = ap.parse_args()
    result = analyze(args.run)
    print(json.dumps(result['heads'], indent=2))
