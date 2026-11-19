"""Budget matrix over one explicitly supplied video; Mock is engineering-only.

No benchmark labels or historical confirmation sets are read by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from queryseek import PIPELINE_VERSION
from queryseek.config import Config
from queryseek.ingest.pipeline import ingest
from queryseek.ingest.cache import write_json_atomic
from queryseek.tools.export import new_job_id
from queryseek.workspace import Workspace


def integers(text):
    values = [int(v.strip()) for v in text.split(',')]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError('Expected positive comma-separated integers')
    return list(dict.fromkeys(values))


def run_matrix(video, srt, query, *, mock, methods, budgets, frames_per_call, max_calls,
               output_tokens=None, config=None):
    cfg = (config or Config.load()).resolved()
    output_tokens = [cfg.vlm.max_output_tokens] if output_tokens is None else output_tokens
    if not methods or any(m not in {'uniform', 'adaptive', 'curve', 'subtitle'} for m in methods):
        raise ValueError('Unknown probe method')
    if not query.strip() or not budgets or not frames_per_call or max_calls <= 0:
        raise ValueError('Invalid probe grid')
    if any(type(v) is not int or not 1 <= v <= 256 for v in budgets):
        raise ValueError('Frame budget must be between 1 and 256')
    if any(type(v) is not int or not 1 <= v <= 64 for v in frames_per_call):
        raise ValueError('Invalid frames-per-call values')
    if not output_tokens or any(type(v) is not int or v <= 0 for v in output_tokens):
        raise ValueError('Invalid output-token limits')
    if mock:
        cfg.embedding.backend = 'mock'
    # An explicit video/SRT input has its own fresh registry ID. Rebuild to avoid
    # the legacy ingest cache treating a changed SRT as the same input.
    vid = 'probe_' + uuid.uuid4().hex[:16]
    index = ingest(video, video_id=vid, srt_path=srt, config=cfg, use_cache=False)
    workspace = Workspace(cfg)
    folder = workspace.path('observation_probes', new_job_id())
    folder.mkdir(parents=True, exist_ok=False)
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    versions = {}
    for package in ('pydantic', 'httpx', 'numpy', 'langgraph', 'scenedetect', 'rank-bm25'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    hints = Path(video).with_suffix('.labels.json')
    protocol = {'version': 'observation-probe-v1', 'pipeline_version': PIPELINE_VERSION,
                'video_sha256': index.asset.sha256, 'duration_sec': index.asset.duration_sec,
                'subtitle_sha256': sha(srt) if srt else None,
                'mock_hint_sha256': sha(hints) if mock and hints.is_file() else None,
                'query': query, 'mock': mock, 'methods': methods, 'budgets': budgets,
                'frames_per_call': frames_per_call, 'max_model_calls': max_calls,
                'max_output_tokens_per_call': output_tokens,
                'model_id': 'mock-vlm-1' if mock else os.environ.get('VLM_MODEL'),
                'model_revision': 'deterministic-mock' if mock else None,
                'config': cfg.model_dump(mode='json'), 'package_versions': versions,
                'python_version': sys.version.split()[0],
                'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p)
                                  for p in sorted((ROOT / 'queryseek').rglob('*.py'))},
                'runner_sha256': sha(__file__),
                'accuracy': 'TODO / 未测', 'scope': 'single supplied video; no confirmation labels'}
    write_json_atomic(folder / 'protocol.json', protocol)
    rows = []
    for method, budget, chunk, token_cap in itertools.product(methods, budgets, frames_per_call, output_tokens):
        run_cfg = cfg.model_copy(deep=True)
        run_cfg.vlm.max_frames_per_call = chunk
        run_cfg.vlm.max_output_tokens = token_cap
        task = Workspace(run_cfg).execute(vid, query, strategy=method, execution='检索与剪辑',
                                           budget=budget, mock=mock,
                                           budget_options={'max_model_calls': max_calls,
                                                           'max_http_attempts': max_calls * (cfg.vlm.max_retries + 1)})
        row = {'method': method, 'budget': budget, 'frames_per_call': chunk,
               'max_output_tokens_per_call': token_cap,
               'run_id': task['id'], 'status': task['status'], 'outcome': task.get('outcome'),
               'metrics': task.get('metrics'), 'error': task.get('error')}
        rows.append(row)
        write_json_atomic(folder / 'results.json', {'protocol': protocol, 'runs': rows})
    return folder, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--srt', type=Path)
    parser.add_argument('--query', required=True)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--methods', default='uniform,adaptive')
    parser.add_argument('--budgets', type=integers, default=[8, 16, 32, 64])
    parser.add_argument('--frames-per-call', type=integers, default=[4, 8])
    parser.add_argument('--max-calls', type=int, default=8)
    parser.add_argument('--output-tokens', type=integers,
                        help='Comma-separated output caps; defaults to the current config')
    args = parser.parse_args()
    folder, rows = run_matrix(args.video, args.srt, args.query, mock=args.mock,
                               methods=args.methods.split(','), budgets=args.budgets,
                               frames_per_call=args.frames_per_call, max_calls=args.max_calls,
                               output_tokens=args.output_tokens)
    print(json.dumps({'output': str(folder), 'tasks': len(rows),
                      'failed': sum(r['status'] == 'failed' for r in rows),
                      'mock': args.mock, 'accuracy': 'TODO / 未测'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
