"""Build bounded CLIP-change trees, then separately evaluate their geometry.

No VLM, search-policy training, or GT-guided node visits. Build accepts the
existing label-free request interface but passes only media identity to the tree.
"""
from __future__ import annotations

import argparse
from collections import deque
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from queryseek.agent.temporal_tree import SplitConfig, TemporalTree, TreeNavigator, uniform_partition
from queryseek.agent.tree_observations import MockBoundaryEncoder, SamplingConfig, WindowObservationStore, output_path, write_json
from queryseek.eval.grounded_qa import checked_intervals, interval_set_iou
from queryseek.ingest.ffprobe import asset_from_probe, sha256_file
from scripts.run_grounded_qa_baseline import check_requests


def build(requests_path, config_path, output, *, weights=None, device='cpu', mock=False):
    config = json.loads(Path(config_path).read_text('utf-8'))
    if sha256_file(requests_path) != config['requests_sha256']:
        raise ValueError('Frozen requests hash mismatch')
    requests = check_requests(json.loads(Path(requests_path).read_text('utf-8')))
    sampling, split = SamplingConfig(**config['sampling']), SplitConfig(**config['split'])
    if sampling.max_frames < split.min_children:
        raise ValueError('Sampling budget cannot support required child count')
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    packages = {}
    for name in ['numpy', 'open-clip-torch', 'timm', 'ftfy', 'wcwidth', 'torch', 'torchvision', 'pillow', 'safetensors']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    encoder_signature = {'model': config['model_name'], 'weights_sha256': config['weights_sha256'],
        'force_quick_gelu': config['force_quick_gelu'], 'mock': mock, 'device': device,
        'packages': packages, 'precision': 'float32', 'image_max_side': sampling.max_side,
        'adapter_sha256': sha256_file(ROOT/'queryseek/embeddings/openclip_encoder.py')}
    source_names = ['scripts/probe_scene_tree.py', 'queryseek/agent/temporal_tree.py',
        'queryseek/agent/tree_observations.py', 'queryseek/vendor/vts/boundaries.py',
        'queryseek/embeddings/openclip_encoder.py', 'queryseek/ingest/frames.py']
    manifest = {'config': config, 'config_sha256': sha256_file(config_path),
        'requests_sha256': sha256_file(requests_path), 'encoder': encoder_signature,
        'source_sha256': {p: sha256_file(ROOT/p) for p in source_names},
        'new_vlm_calls': 0, 'new_training': False, 'mock': mock}
    write_json(output/'run_manifest.json', manifest)
    load_start = time.perf_counter()
    torch = None
    if mock:
        encoder = MockBoundaryEncoder()
    else:
        if weights is None or sha256_file(weights) != config['weights_sha256']:
            raise ValueError('Frozen CLIP weight hash mismatch')
        import torch
        from queryseek.embeddings.openclip_encoder import OpenCLIPEmbedding
        torch.manual_seed(12)
        torch.set_num_threads(8)
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable')
            torch.cuda.reset_peak_memory_stats()
            encoder_signature['gpu'] = torch.cuda.get_device_name()
        encoder = OpenCLIPEmbedding(model_name=config['model_name'], pretrained=str(Path(weights).resolve()),
            device=device, max_side=sampling.max_side, force_quick_gelu=config['force_quick_gelu'])
        write_json(output/'run_manifest.json', manifest)
    load_sec = time.perf_counter()-load_start
    records = []
    for request in requests:
        began = time.perf_counter()
        row = {'qid': request['qid'], 'video_id': request['video_id'], 'status': 'error', 'mock': mock}
        tree, stage, node_id = None, 'media_probe', None
        try:
            asset = asset_from_probe(request['video_id'], request['video_path'])
            if abs(asset.duration_sec-request['duration_sec']) > .001:
                raise ValueError('Frozen probed duration changed')
            provider = WindowObservationStore(asset.path, asset.duration_sec, asset.sha256,
                                              encoder, encoder_signature, sampling)
            tree = TemporalTree(asset.duration_sec, provider, split)
            queue = deque(['r'])
            stage = 'tree_expansion'
            while queue and len(tree.observations) < split.max_observed_nodes:
                node_id = queue.popleft()
                children = tree.expand(node_id)
                queue.extend(children)
                print(json.dumps({'qid': request['qid'], 'node': node_id, 'children': len(children),
                                  'observed_nodes': len(tree.observations)}), flush=True)
            stage = 'navigation_tool_check'
            navigator = TreeNavigator(tree)
            children = tree.nodes['r'].children
            if children:
                navigator.observe(4)
                navigator.move('zoom_in', children[0])
                navigator.move('shift', children[-1])
                navigator.move('zoom_out')
            row.update(status='ok', video_sha256=asset.sha256, tree=tree.to_dict(),
                       navigation_tool_check=navigator.state())
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            row['error_stage'] = stage
            row['failed_node_id'] = node_id
            # Code locations aid diagnosis without serializing exception text,
            # command arguments, credentials, query text or image bytes.
            row['error_locations'] = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno}
                                     for f in traceback.extract_tb(exc.__traceback__)]
            if tree is not None:
                row['tree'] = tree.to_dict()
        row['elapsed_sec'] = time.perf_counter()-began
        records.append(row)
        with (output/'trees.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps({'qid': request['qid'], 'status': row['status'], 'elapsed_sec': row['elapsed_sec']}), flush=True)
    costs = [c for r in records for c in r.get('tree', {}).get('observation_costs', {}).values()]
    summary = {'scheduled': len(requests), 'completed': sum(r['status'] == 'ok' for r in records),
        'failed': sum(r['status'] != 'ok' for r in records), 'mock': mock, 'model_load_sec': load_sec,
        'probe_elapsed_sec': sum(r['elapsed_sec'] for r in records), 'observed_nodes': len(costs),
        'encoded_images': sum(c['encoded_images'] for c in costs),
        'new_frames_extracted': sum(c['new_frames_extracted'] for c in costs),
        'baseline_reused_nodes': sum(c['reused_baseline_frames'] for c in costs),
        'feature_cache_hits': sum(c['feature_cache_hit'] for c in costs),
        'observation_costs_complete': all(r['status'] == 'ok' for r in records),
        'observation_cost_scope': 'Completed observations, including partial trees. Failed attempts may incur additional unmeasured work.',
        'peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch and device.startswith('cuda') else None,
        'new_vlm_calls': 0, 'new_training': False, 'qa_accuracy': None}
    write_json(output/'run_summary.json', summary)
    return summary


def geometry(spans, labels, duration):
    gt = checked_intervals(labels, duration)
    return {'best_child_union_iou': max((interval_set_iou([s], gt, duration) for s in spans), default=0.),
            'gt_segment_containment': sum(any(s <= a and b <= e for s, e in spans) for a, b in gt)/len(gt),
            'mean_overlapping_children_per_gt': sum(sum(max(s,a) < min(e,b) for s,e in spans) for a,b in gt)/len(gt)}


def evaluate(run_dir, labels_path, config_path, output):
    config = json.loads(Path(config_path).read_text('utf-8'))
    if sha256_file(labels_path) != config['labels_sha256']:
        raise ValueError('Frozen labels hash mismatch')
    manifest = json.loads((run_dir/'run_manifest.json').read_text('utf-8'))
    if manifest['config_sha256'] != sha256_file(config_path):
        raise ValueError('Config changed after tree generation')
    if manifest['mock']:
        raise ValueError('Mock trees cannot be evaluated as real model results')
    labels = json.loads(Path(labels_path).read_text('utf-8'))
    if len({r['qid'] for r in labels}) != len(labels):
        raise ValueError('Duplicate label id')
    rows = [json.loads(line) for line in (run_dir/'trees.jsonl').read_text('utf-8').splitlines()]
    if len({r['qid'] for r in rows}) != len(rows) or {r['qid'] for r in rows} != {r['qid'] for r in labels}:
        raise ValueError('All scheduled queries must remain in denominator')
    lookup = {r['qid']: r for r in labels}
    methods = config['root_comparisons']
    scored = []
    for row in rows:
        item = {'qid': row['qid'], 'status': row['status'], 'methods': {}}
        if row['status'] == 'ok':
            tree = row['tree']
            nodes = {n['node_id']: n for n in tree['nodes']}
            root = nodes['r']
            children = root['children'] or ()
            if not children:
                raise ValueError('This root comparison requires a splittable video')
            duration = tree['duration_sec']
            spans = {'binary_uniform': uniform_partition(0., duration, 2),
                     'same_arity_uniform': uniform_partition(0., duration, len(children)),
                     'clip_change': [(nodes[c]['start_sec'], nodes[c]['end_sec']) for c in children]}
            gt = lookup[row['qid']]['gt_intervals_sec']
            item['methods'] = {name: geometry(spans[name], gt, duration) for name in methods}
            times = root['split_info']['source_times_sec']
            item.update(root_children=len(children), root_pts_hit=any(a<=t<b for t in times for a,b in gt),
                        boundary_bracket_widths_sec=[b-a for a,b in root['split_info']['boundary_brackets_sec']],
                        forced_minimum=root['split_info']['forced_minimum'])
        else:
            item['methods'] = {name: {'best_child_union_iou': 0., 'gt_segment_containment': 0.,
                'mean_overlapping_children_per_gt': None} for name in methods}
        scored.append(item)
    aggregates = {name: {key: sum(r['methods'][name][key] for r in scored)/len(scored)
        for key in ['best_child_union_iou', 'gt_segment_containment']} for name in methods}
    result = {'scope': 'Offline root-partition geometry; best-child selection uses GT only in evaluation. Not online QA or policy accuracy.',
        'n': len(scored), 'completed': sum(r['status'] == 'ok' for r in scored), 'methods': aggregates,
        'rows': scored, 'source_sha256': {'trees.jsonl': sha256_file(run_dir/'trees.jsonl'),
             'config': sha256_file(config_path), 'labels': sha256_file(labels_path)},
        'new_vlm_calls': 0, 'qa_accuracy': None}
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'evaluation.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build')
    b.add_argument('--requests', type=Path, required=True)
    b.add_argument('--weights', type=Path)
    b.add_argument('--device', default='cpu')
    b.add_argument('--mock', action='store_true')
    e = sub.add_parser('evaluate')
    e.add_argument('--run-dir', type=Path, required=True)
    e.add_argument('--labels', type=Path, required=True)
    for p in [b, e]:
        p.add_argument('--config', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.action == 'build':
        result = build(args.requests, args.config, args.output, weights=args.weights, device=args.device, mock=args.mock)
        print(json.dumps(result))
        if result['failed']:
            raise SystemExit(2)
    else:
        print(json.dumps(evaluate(args.run_dir, args.labels, args.config, args.output)['methods']))


if __name__ == '__main__':
    main()
