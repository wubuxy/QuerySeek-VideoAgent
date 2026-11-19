"""Local single-user video library and durable runs, shared by the web UI.

Runtime records and uploaded media stay below output_root. SQLite transactions,
atomic JSON and a process lock keep reruns from executing a request twice.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

from .config import Config
from .ingest import pipeline
from .ingest.cache import read_json, write_json_atomic
from .ingest.ffprobe import asset_from_probe
from .ingest.registry import load_registry, sanitize_video_id
from .ingest.subtitles import load_srt
from .logging_utils import TraceWriter
from .tools.export import export_clip, new_job_id, sanitize_job_id
from .tools.paths import resolved_path_is_within


def _now():
    return datetime.now(timezone.utc).isoformat()


class WorkspaceBusy(RuntimeError):
    pass


class Workspace:
    def __init__(self, config: Config):
        self.cfg = config.resolved()
        self.root = self.cfg.output_root.resolve()
        self.library = self.path('library')
        self.library.mkdir(parents=True, exist_ok=True)
        self.db = self.path('library', 'workspace.sqlite')
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
                    payload TEXT NOT NULL);
            ''')
        # A free process lock proves no workspace task is currently executing.
        try:
            with self.execution_lock():
                self._recover()
        except WorkspaceBusy:
            pass

    def path(self, *parts) -> Path:
        candidate = self.root.joinpath(*parts).resolve()
        if not resolved_path_is_within(candidate, self.root):
            raise ValueError('路径超出工作目录')
        return candidate

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def execution_lock(self):
        lock_path = self.path('library', 'execution.lock')
        with open(lock_path, 'a+b') as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise WorkspaceBusy('已有任务正在处理，请稍后刷新') from exc
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _put(self, table, item):
        if table not in {'videos', 'runs'}:
            raise ValueError('Invalid record type')
        payload = json.dumps(item, ensure_ascii=False)
        with self.connection() as db:
            if table == 'runs':
                db.execute('INSERT INTO runs VALUES (?, ?, ?) ON CONFLICT(id) '
                           'DO UPDATE SET payload=excluded.payload',
                           (item['id'], item['request_id'], payload))
            else:
                db.execute('INSERT INTO videos VALUES (?, ?) ON CONFLICT(id) '
                           'DO UPDATE SET payload=excluded.payload', (item['id'], payload))

    def _records(self, table):
        if table not in {'videos', 'runs'}:
            raise ValueError('Invalid record type')
        with self.connection() as db:
            return [json.loads(row[0]) for row in db.execute(f'SELECT payload FROM {table}')]

    def _recover(self):
        for run in self._records('runs'):
            if run['status'] in {'queued', 'running'}:
                run.update(status='interrupted', finished_at=_now(),
                           error='上次任务中断；可保留记录并新建任务重跑')
                self._put('runs', run)
            if run.get('export_status') == 'running':
                run.update(export_status='interrupted')
                self._put('runs', run)
        for video in self._records('videos'):
            if video['status'] == 'indexing':
                video.update(status='interrupted', error='索引处理中断，请重新建立索引')
                self._put('videos', video)

    def videos(self):
        entries = {v['id']: v for v in self._records('videos')}
        for vid, asset in load_registry(self.cfg.data_root).get('videos', {}).items():
            if vid not in entries:
                entries[vid] = {'id': vid, 'title': vid, 'status': 'registered',
                                'asset': asset, 'created_at': '', 'managed': False}
        return sorted(entries.values(), key=lambda v: v['created_at'], reverse=True)

    def video(self, video_id):
        sanitize_video_id(video_id)
        return next((v for v in self.videos() if v['id'] == video_id), None)

    def runs(self, video_id=None):
        return sorted((r for r in self._records('runs')
                       if video_id is None or r['video_id'] == video_id),
                      key=lambda r: r['created_at'], reverse=True)

    def run(self, run_id):
        sanitize_job_id(run_id)
        return next((r for r in self._records('runs') if r['id'] == run_id), None)

    def save_video(self, name: str, content: bytes, *, subtitle: bytes | None = None):
        """Validate before publishing a library entry; never overwrite another upload."""
        suffix = Path(name.replace('\\', '/')).suffix.lower()
        if suffix not in {'.mp4', '.mkv', '.mov'}:
            raise ValueError('仅支持 MP4、MKV、MOV 视频')
        if not content or len(content) > self.cfg.video.max_file_size_mb * 1024 * 1024:
            raise ValueError('视频为空或超过文件大小上限')
        if subtitle is not None and len(subtitle) > 10 * 1024 * 1024:
            raise ValueError('字幕超过 10MB 上限')
        with self.execution_lock():
            digest = hashlib.sha256(content).hexdigest()
            existing = next((v for v in self._records('videos')
                             if v['asset']['sha256'] == digest), None)
            vid = existing['id'] if existing else f'video_{uuid.uuid4().hex[:16]}'
            folder = self.path('library', 'media', vid)
            folder.mkdir(parents=True, exist_ok=True)
            source = self.path('library', 'media', vid, 'source' + suffix)
            if existing:
                source = self.path(existing['asset']['path'])
                asset = existing['asset']
            else:
                source.write_bytes(content)
                try:
                    probed = asset_from_probe(vid, source)
                    if not 0 < probed.duration_sec <= self.cfg.video.max_duration_sec:
                        raise ValueError('视频时长无效或超过上限')
                    asset = probed.model_dump()
                except Exception:
                    source.unlink(missing_ok=True)
                    raise ValueError('无法读取有效视频，请检查格式和时长') from None
            sub_path = None
            sub_hash = None
            if subtitle is not None:
                sub_hash = hashlib.sha256(subtitle).hexdigest()
                sub_path = self.path('library', 'media', vid, f'subtitles_{sub_hash}.srt')
                sub_path.write_bytes(subtitle)
                try:
                    if not load_srt(sub_path, duration_sec=asset['duration_sec']):
                        raise ValueError('空字幕')
                except Exception:
                    sub_path.unlink(missing_ok=True)
                    if not existing:
                        source.unlink(missing_ok=True)
                    raise ValueError('无法读取有效 SRT 字幕') from None
            video = existing or {'id': vid, 'title': Path(name.replace('\\', '/')).name[:200],
                                  'asset': asset, 'created_at': _now(), 'managed': True,
                                  'status': 'saved', 'subtitle_path': None,
                                  'subtitle_sha256': None, 'index_signature': None}
            if sub_path and sub_hash != video.get('subtitle_sha256'):
                video.update(subtitle_path=str(sub_path), subtitle_sha256=sub_hash, status='saved')
            self._put('videos', video)
            return video, existing is not None

    def prepare_video(self, video_id, mock=True):
        with self.execution_lock():
            return self._prepare(video_id, mock)

    def _prepare(self, video_id, mock):
        video = self.video(video_id)
        if not video:
            raise ValueError('视频未就绪：请先上传并保存视频')
        if not video.get('managed'):
            return pipeline.load_index(video_id, self.cfg)
        cfg = self.cfg.model_copy(deep=True)
        if mock:
            cfg.embedding.backend = 'mock'
        signature = hashlib.sha256(json.dumps({
            'config': cfg.snapshot_for_cache(), 'subtitle': video.get('subtitle_sha256'),
        }, sort_keys=True).encode()).hexdigest()
        from .ingest.cache import cache_dir_for
        cache_meta = read_json(cache_dir_for(cfg.cache_root, video['asset']['sha256']) / 'index_meta.json') or {}
        use_cache = cache_meta.get('workspace_signature') == signature
        video.update(status='indexing', error=None)
        self._put('videos', video)
        try:
            source = self.path(video['asset']['path'])
            subtitle = self.path(video['subtitle_path']) if video.get('subtitle_path') else None
            index = pipeline.ingest(source, video_id=video_id, srt_path=subtitle,
                                    config=cfg, use_cache=use_cache)
            meta_path = index.cache_dir / 'index_meta.json'
            meta = read_json(meta_path) or {}
            meta['workspace_signature'] = signature
            write_json_atomic(meta_path, meta)
            video.update(status='ready', index_signature=signature,
                         thumbnail=index.frames[0].image_path if index.frames else None,
                         index_stats={'frames': len(index.frames), 'scenes': len(index.scenes),
                                      'subtitles': len(index.subtitles)})
            self._put('videos', video)
            return index
        except Exception as exc:
            video.update(status='failed', error=f'索引失败（{type(exc).__name__}），可重试')
            self._put('videos', video)
            raise

    def _locate(self, index, video_id, query, strategy, execution, budget, mock, folder, trace, session):
        from .retrieval.dispatch import search_index
        from .agent.state import default_budget, initial_state
        from .agent.graph import run_agent

        with session.activate():
            if execution == 'Agent':
                from langgraph.checkpoint.sqlite import SqliteSaver
                limits = default_budget({'max_iterations': self.cfg.agent.max_iterations,
                                         'max_vlm_calls': session.ledger.limits.max_model_calls,
                                         'max_frames': budget,
                                         'wall_time_limit_sec': session.ledger.limits.query_deadline_sec})
                state = initial_state(video_id, query, strategy=strategy, budget=limits)
                state['started_at'] = time.time()
                with SqliteSaver.from_conn_string(str(folder / 'checkpoint.sqlite')) as saver:
                    return run_agent(state,
                                     {'thread_id': folder.name, 'cfg': self.cfg.model_dump(),
                                      'mock': mock, 'trace_path': str(folder / 'trace.jsonl'),
                                      'observation_session': session}, checkpointer=saver)
            found = search_index(self.cfg, index, video_id, query, strategy,
                                 budget if strategy in {'uniform', 'adaptive'} else None, mock, trace)
            return {'status': 'ok' if found.windows else 'no_result',
                    'selected_windows': [w.model_dump() for w in found.windows],
                    'evidence': [e.model_dump() for e in found.evidence],
                    'used_frame_ids': found.used_frame_ids, 'vlm_calls': found.vlm_calls,
                    'indexed_frames': found.indexed_frames,
                    'online_encoded_frames': found.online_encoded_frames,
                    'embedding_cache_hit': found.embedding_cache_hit,
                    'warnings': found.warnings, 'budget_exhausted': found.budget_exhausted}

    def execute(self, video_id, query, *, strategy='adaptive', execution='Agent',
                budget=32, mock=True, request_id=None, parent_id=None, on_progress=None,
                budget_options=None):
        from .retrieval.dispatch import AGENT_STRATEGIES
        from .observation_budget import ObservationBudget
        from .models.metered import ObservationSession

        options = {'max_unique_frames': budget, 'max_model_calls': self.cfg.agent.max_vlm_calls,
                   'max_http_attempts': self.cfg.agent.max_vlm_calls * (self.cfg.vlm.max_retries + 1),
                   'max_output_tokens_per_call': self.cfg.vlm.max_output_tokens,
                   'query_deadline_sec': self.cfg.agent.wall_time_limit_sec}
        options.update(budget_options or {})
        limits = ObservationBudget.model_validate(options)
        if limits.max_unique_frames != budget:
            raise ValueError('帧预算参数冲突')

        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError('请输入 1–2000 字的事件描述')
        if execution not in {'Agent', '检索与剪辑'}:
            raise ValueError('未知执行方式')
        allowed = AGENT_STRATEGIES if execution == 'Agent' else ('curve', *AGENT_STRATEGIES)
        if strategy not in allowed or type(budget) is not int or not 1 <= budget <= 256:
            raise ValueError('策略或帧预算无效')
        if not mock and not os.environ.get('VLM_API_KEY') and (execution == 'Agent' or strategy in {'uniform', 'adaptive'}):
            raise ValueError('真实模式未配置 VLM 密钥；请配置环境变量或选择 Mock')
        with self.execution_lock():
            self._recover()
            token = request_id or uuid.uuid4().hex
            existing = next((r for r in self._records('runs') if r['request_id'] == token), None)
            if existing:
                current_video = self.video(video_id)
                expected = {'video_id': video_id, 'query': query, 'strategy': strategy,
                            'execution': execution, 'budget_frames': budget,
                            'mock': mock, 'parent_id': parent_id,
                            'config': self.cfg.model_dump(mode='json'),
                            'budget_limits': limits.model_dump(),
                            'subtitle_sha256': (current_video or {}).get('subtitle_sha256')}
                if any(existing.get(k) != v for k, v in expected.items()):
                    raise ValueError('请求标识已用于其他参数，请新建任务')
                return existing
            parent = self.run(parent_id) if parent_id else None
            if parent_id and (not parent or parent['video_id'] != video_id):
                raise ValueError('追问必须关联同一视频的已有任务')
            video = self.video(video_id)
            if not video:
                raise ValueError('视频未就绪：请先上传并保存视频')
            run_id = new_job_id()
            folder = self.path(run_id)
            folder.mkdir(parents=True, exist_ok=False)
            run = {'id': run_id, 'request_id': token, 'video_id': video_id,
                   'query': query, 'parent_id': parent_id, 'strategy': strategy,
                   'execution': execution, 'budget_frames': budget, 'mock': mock,
                   'model': 'mock-vlm-1' if mock else os.environ.get('VLM_MODEL', '未指定'),
                   'subtitle_sha256': video.get('subtitle_sha256'),
                   'created_at': _now(), 'status': 'queued', 'export_status': 'not_requested',
                   'budget_limits': limits.model_dump(),
                   'config': self.cfg.model_dump(mode='json'), 'error': None, 'progress': []}
            self._put('runs', run)

            def advance(stage):
                run['progress'].append({'stage': stage, 'at': _now()})
                self._put('runs', run)
                if on_progress is not None:
                    # UI callbacks receive a snapshot, never the mutable task state.
                    try:
                        on_progress(json.loads(json.dumps(run)))
                    except Exception:
                        # A disconnected display must not change model execution.
                        pass

            started = time.perf_counter()
            session = None
            trace = TraceWriter(folder / 'trace.jsonl')
            try:
                run.update(status='running')
                advance('indexing')
                index = self._prepare(video_id, mock)
                run['source_asset'] = index.asset.model_dump()
                preparation_sec = time.perf_counter() - started
                session = ObservationSession(limits, index.asset)
                advance('searching')
                result = self._locate(index, video_id, query, strategy, execution, budget, mock, folder, trace, session)
                for evidence in result.get('evidence', []):
                    if evidence.get('type') == 'asr':
                        session.state.add(source='subtitle', kind='observation', status='observed',
                                          content=evidence.get('content') or '', turn=len(session.ledger.calls))
                    elif evidence.get('type') == 'frame' and (
                            strategy in {'curve', 'semantic'} or 'embedding' in (evidence.get('model_id') or '')
                            or 'openclip' in (evidence.get('model_id') or '')):
                        session.state.add(source='retrieval', kind='retrieval_hint', status='tentative',
                                          content=f"候选帧 {evidence['evidence_id']} @ {evidence.get('timestamp_sec')}s: "
                                                  + (evidence.get('content') or ''), turn=len(session.ledger.calls))
                if session.ledger.rejections:
                    result['budget_exhausted'] = True
                    result.setdefault('warnings', []).append('观察预算限制：' + ', '.join(session.ledger.rejections))
                result['observation_records'] = session.state.snapshot()
                advance('saving')
                # Cached frames may be rebuilt by later index configurations.
                # Freeze cited pictures with the run so old evidence stays readable.
                for item in result.get('evidence', []):
                    source = Path(item.get('source_file') or '').resolve()
                    if (item.get('type') == 'frame' and source.is_file()
                            and source.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                            and any(resolved_path_is_within(source, root.resolve())
                                    for root in (self.cfg.cache_root, self.root))):
                        from .ingest.ffprobe import sha256_file
                        dest = self.path(run_id, 'evidence', sha256_file(source) + source.suffix.lower())
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not dest.exists():
                            shutil.copyfile(source, dest)
                        item['source_file'] = str(dest)
                write_json_atomic(folder / 'result.json', result)
                metrics = {'status': result.get('status'), 'vlm_calls': result.get('vlm_calls'),
                           'used_frames': len(set(result.get('used_frame_ids', []))),
                           'indexed_frames': result.get('indexed_frames'),
                           'online_encoded_frames': result.get('online_encoded_frames'),
                           'embedding_cache_hit': result.get('embedding_cache_hit'),
                           'elapsed_sec': round(time.perf_counter() - started, 3),
                           'input_tokens': None, 'output_tokens': None}
                metrics.update(session.ledger.summary(), preparation_sec=round(preparation_sec, 6))
                write_json_atomic(folder / 'metrics.json', metrics)
                run.update(status='failed' if result.get('status') == 'failed' else 'completed',
                           outcome=result.get('status'), metrics=metrics)
            except Exception as exc:
                run.update(status='failed', error=f'运行失败（{type(exc).__name__}）；请检查视频索引与后端配置')
                run['metrics'] = {'status': 'failed', 'vlm_calls': None, 'used_frames': None,
                                  'input_tokens': None, 'output_tokens': None,
                                  'elapsed_sec': round(time.perf_counter() - started, 3)}
                write_json_atomic(folder / 'metrics.json', run['metrics'])
                trace.event('workspace_error', error_type=type(exc).__name__)
            finally:
                if session is not None:
                    write_json_atomic(folder / 'costs.json', session.ledger.snapshot())
                    write_json_atomic(folder / 'observations.json', session.state.snapshot())
                    if run['status'] == 'failed':
                        run.setdefault('metrics', {}).update(session.ledger.summary())
                        write_json_atomic(folder / 'metrics.json', run['metrics'])
                trace.close()
                run['finished_at'] = _now()
                advance('completed' if run['status'] == 'completed' else 'failed')
            return run

    def result(self, run_id):
        if not self.run(run_id):
            raise ValueError('任务不存在')
        return read_json(self.path(run_id, 'result.json'))

    def export(self, run_id, target_seconds=None):
        from .agent.state import dicts_to_windows
        if target_seconds is not None and (not math.isfinite(target_seconds) or target_seconds <= 0):
            raise ValueError('目标时长必须为正的有限秒数')
        with self.execution_lock():
            run = self.run(run_id)
            if not run or run['status'] != 'completed':
                raise ValueError('只有完成的定位任务可以导出')
            result = self.result(run_id) or {}
            windows = dicts_to_windows(result.get('selected_windows', []))
            if not windows:
                raise ValueError('该任务没有可导出的区间')
            video = self.video(run['video_id'])
            from .schemas import VideoAsset
            asset = VideoAsset.model_validate(run.get('source_asset') or video['asset'])
            if any(w.end_sec > asset.duration_sec for w in windows):
                raise ValueError('导出区间超过视频时长')
            run.update(export_status='running')
            self._put('runs', run)
            try:
                # An export has its own immutable folder, separate from localization.
                export_id = new_job_id()
                saved_cfg = Config.model_validate(run['config'])
                out, manifest = export_clip(asset, windows, self.root, saved_cfg,
                                            job_id=export_id, target_seconds=target_seconds)
                manifest = manifest.model_copy(update={'query': run['query'], 'strategy': run['strategy']})
                write_json_atomic(self.path(export_id, 'manifest.json'), manifest.model_dump(mode='json'))
                run.update(export_status='completed', export_id=export_id, clip=str(out),
                           export_duration_sec=manifest.output_duration_sec, export_error=None)
            except Exception as exc:
                run.update(export_status='failed', export_error=f'导出失败（{type(exc).__name__}），定位结果已保留')
            self._put('runs', run)
            return run

    def artifact(self, run_id, name):
        run = self.run(run_id)
        if not run:
            raise ValueError('任务不存在')
        if name == 'clip.mp4':
            if run.get('export_status') != 'completed':
                raise ValueError('尚无完成的导出')
            return self.path(run['export_id'], name)
        if name not in {'result.json', 'metrics.json', 'costs.json', 'observations.json'}:
            raise ValueError('未知下载文件')
        return self.path(run_id, name)
