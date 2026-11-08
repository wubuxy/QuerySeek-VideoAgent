"""Repair the text-only bottleneck; regress on frozen, previously used sources."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from queryseek.retrieval import grounded_reader as fix
from scripts import probe_compact_reader as parent
s, c = parent.s, parent.c
METHODS = ('minimal', 'compact', 'fact_map', 'visual_remap', 'citation_checked')


def sources():
    return {**parent.sources(), **{n: s.sha(ROOT/n) for n in
        ['scripts/probe_grounded_reader_fix.py', 'queryseek/retrieval/grounded_reader.py']}}


def begin(path, config, mock):
    path = s.study.baseline.checked_output(path); path.mkdir(parents=True, exist_ok=False)
    s.write(path/'manifest.initial.json', {'config_sha256': s.sha(config), 'source_sha256': sources(), 'mock': mock})
    return path


def verify(path, config, mock):
    m = s.read(path/'manifest.json')
    if m['config_sha256'] != s.sha(config) or m['source_sha256'] != sources() or m['mock'] != mock:
        raise ValueError('Provenance mismatch')
    for n, h in m['files'].items():
        if s.sha(path/n) != h: raise ValueError('Artifact changed')


class Engine(parent.Engine):
    def call(self, *args, **kwargs):
        if len(self.calls) >= self.cfg['max_generations']:
            raise RuntimeError('Generation budget exhausted')
        return super().call(*args, **kwargs)


def repair(row, donor, previous, engine):
    identity = {k: previous[k] for k in ('group', 'qid', 'input', 'donor_qid')}
    visual = {**identity, 'method': 'visual_remap', 'status': 'error', 'answer': 'abstain',
              'evidence': [], 'calls': [], 'mock': engine.mock}
    final = {**visual, 'method': 'citation_checked', 'calls': []}
    try:
        if 'preparation_error' in row or 'preparation_error' in donor:
            raise ValueError('Preparation error')
        frames = [s.VideoFrame.model_validate(f) for f in donor['frames']]
        request = row['request']
        fact_call = previous['calls'][0]
        if fact_call['step'] != 'fact' or fact_call['model_call']['status'] == 'error':
            raise ValueError('Invalid first-stage observation')
        observation = c.decode_fact(fact_call['raw'], len(frames))
        if observation != previous['observation']:
            raise ValueError('Observation changed')
        if fact_call['image_sha256'] != donor['frame_sha256']:
            raise ValueError('Observation source changed')
        reused = {**fact_call, 'reused': True}
        visual['calls'] = [reused]
        prompt, schema = fix.visual_remap_prompt(request, observation, len(frames))
        call = engine.call(prompt, schema, frames, 'compact', {**identity, 'stage': 'visual_remap'})
        visual['calls'].append(call)
        if call['model_call']['status'] == 'error': raise RuntimeError('Visual backend error')
        proposal = c.decode_compact(call['raw'], request, len(frames))
        visual.update(proposal, status='ok', evidence_source_times_sec=[frames[i].timestamp_sec for i in proposal['evidence']])
        final = {**visual, 'method': 'citation_checked', 'calls': list(visual['calls'])}
        if proposal['answer'] == 'abstain':
            final['citation_status'] = 'upstream_abstain'
            return visual, final
        subset, mapping = fix.cited_views(frames, proposal)
        final['local_to_source'] = mapping
        prompt, schema = fix.cited_prompt(request, len(subset))
        call = engine.call(prompt, schema, subset, 'compact', {**identity, 'stage': 'citation_only'})
        final['calls'].append(call)
        if call['model_call']['status'] == 'error': raise RuntimeError('Citation backend error')
        check = c.decode_compact(call['raw'], request, len(subset))
        final['citation_read'] = check
        final.update(fix.bind_citation_result(proposal, check, mapping))
        final['evidence_source_times_sec'] = [frames[i].timestamp_sec for i in final['evidence']]
    except Exception as exc:
        if visual['status'] != 'ok':
            visual['error_type'] = type(exc).__name__
            final['calls'] = list(visual['calls'])
        final.update(status='error', answer='abstain', evidence=[], evidence_source_times_sec=[], error_type=type(exc).__name__)
    return visual, final


def run(args):
    cfg = s.read(args.config); old = Path(args.parent_run)
    if s.sha(args.parent_config) != cfg['parent_config_sha256']: raise ValueError('Parent configuration changed')
    parent.verify(old, args.parent_config, args.mock)
    groups = s.read(old/'prepared.json'); history = s.previous.lines(old/'records.jsonl')
    for rows in groups.values():
        if len(rows) != cfg['sample_count']: raise ValueError('Missing sources')
        s.donor_indices(rows)
        for row in rows:
            if 'preparation_error' not in row: s.validate_row(row, cfg['frames'])
    out = begin(args.output, args.config, args.mock)
    s.write(out/'prepared.json', groups); s.write(out/'history.json', history)
    model_hash = None if args.mock else s.previous.verify_model_snapshot(args.model_path, cfg['model_revision'])
    if not args.mock and model_hash != cfg['model_manifest_sha256']: raise ValueError('Model changed')
    s.write(out/'model.json', {'manifest_sha256': model_hash, 'mock': args.mock, 'precision': 'bf16'})
    start = time.perf_counter(); backend = None; blocked = None
    try:
        if not args.mock: backend = s.Backend(cfg, args.model_path, 'bf16')
    except Exception as exc: blocked = type(exc).__name__
    load_sec = time.perf_counter()-start
    engine = Engine(cfg, backend, out, args.mock, blocked); records = []
    for group, rows in groups.items():
        donors = s.donor_indices(rows)
        for kind in ('matched', 'mismatched'):
            for i, row in enumerate(rows):
                donor = rows[donors[i]] if kind == 'mismatched' else row
                prior = [r for r in history if r['group'] == group and r['input'] == kind and r['qid'] == row['request']['qid'] and r['method'] == 'fact_map']
                if len(prior) != 1 or prior[0]['donor_qid'] != donor['request']['qid']: raise ValueError('Historical identity mismatch')
                visual, final = repair(row, donor, prior[0], engine)
                for r in (visual, final): records.append(r); s.append(out/'records.jsonl', r)
                print(json.dumps({'records': len(records), 'new_calls': len(engine.calls), 'group': group, 'input': kind,
                                  'visual_status': visual['status'], 'final_status': final['status']}), flush=True)
    return s.finish(out, ['prepared.json', 'history.json', 'model.json', 'records.jsonl', 'calls.jsonl'],
        {'mock': args.mock, 'records': len(records), 'new_physical_cost': parent.cost(engine.calls),
         'model_load_sec': load_sec, 'backend_blocked': engine.blocked})


def evaluate(args):
    cfg = s.read(args.config); folder = Path(args.runs); verify(folder, args.config, args.mock)
    groups = s.read(folder/'prepared.json'); records = s.previous.lines(folder/'records.jsonl')
    history = s.read(folder/'history.json'); physical = s.previous.lines(folder/'calls.jsonl')
    labels = {g: {r['qid']: r['right_answer'] for r in s.checked(p, cfg[k])} for g, p, k in
        [('development', args.dev_labels, 'dev_labels_sha256'), ('new_sources', args.labels, 'fresh_labels_sha256')]}
    expected = {(g, row['request']['qid'], kind, method) for g, rows in groups.items() for row in rows
                for kind in ('matched', 'mismatched') for method in ('visual_remap', 'citation_checked')}
    if len(records) != len(expected) or {(r['group'], r['qid'], r['input'], r['method']) for r in records} != expected:
        raise ValueError('Incomplete denominator')
    result = {'mock': args.mock, 'scope': 'Previously used sources, oracle evidence: repair regression only.',
              'groups': {}, 'new_physical_cost': parent.cost(physical), 'run_summary': s.read(folder/'summary.json')}
    for group, rows in groups.items():
        ids = [r['request']['qid'] for r in rows]
        if set(labels[group]) != set(ids): raise ValueError('Label denominator mismatch')
        result['groups'][group] = {}
        for kind in ('matched', 'mismatched'):
            selected = [r for r in records+history if r['group'] == group and r['input'] == kind]
            result['groups'][group][kind] = {method: {**c.metrics([r for r in selected if r['method'] == method], ids,
                labels[group] if kind == 'matched' else None), 'logical_cost_including_reused': parent.cost(
                    [x for r in selected if r['method'] == method for x in r['calls']])} for method in METHODS}
    out = begin(args.output, args.config, args.mock); s.write(out/'evaluation.json', result)
    return s.finish(out, ['evaluation.json'], {'mock': args.mock, 'new_calls': len(physical)})


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['run', 'evaluate'])
    for n in ['config', 'parent-config', 'parent-run', 'model-path', 'output', 'runs', 'dev-labels', 'labels']:
        p.add_argument('--'+n, type=Path)
    p.add_argument('--mock', action='store_true'); args = p.parse_args()
    print(json.dumps(globals()[args.action](args)), flush=True)


if __name__ == '__main__': main()
