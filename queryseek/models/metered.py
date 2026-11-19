"""Observation session shared across retrieval, Agent nodes and API retries."""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

from ..observation_budget import ACTIVE_CALL, ACTIVE_SESSION, ObservationLedger
from ..observation_state import ObservationState
from .interface import InspectionResult
from .mock import MockVLM
from .compatible_api import CompatibleAPIVLM


class ObservationSession:
    def __init__(self, limits, asset):
        self.ledger = ObservationLedger(limits, asset.sha256, asset.duration_sec)
        self.state = ObservationState()
        self.adapters = []
        self.state.add(source='program', kind='metadata', status='measured',
                       content=f'视频 SHA256={asset.sha256}; duration_sec={asset.duration_sec}', turn=0)

    @contextmanager
    def activate(self):
        token = ACTIVE_SESSION.set(self)
        try:
            yield self
        finally:
            ACTIVE_SESSION.reset(token)
            for adapter in self.adapters:
                close = getattr(adapter, 'close', None)
                if close:
                    close()

    def wrap(self, adapter):
        if isinstance(adapter, MeteredVLM):
            return adapter
        self.adapters.append(adapter)
        if hasattr(adapter, 'max_output_tokens'):
            adapter.max_output_tokens = self.ledger.limits.max_output_tokens_per_call
        return MeteredVLM(adapter, self)


class MeteredVLM:
    def __init__(self, adapter, session):
        self.adapter, self.session = adapter, session
        self.model_id = adapter.model_id
        self.max_frames_per_call = getattr(adapter, 'max_frames_per_call', 8)
        if type(self.max_frames_per_call) is not int or self.max_frames_per_call <= 0:
            raise ValueError('Invalid frames-per-call limit')
        self.mock = isinstance(adapter, MockVLM)
        self.accounting = 'mock' if self.mock else 'http' if isinstance(adapter, CompatibleAPIVLM) else 'unknown'

    @property
    def last_usage(self):
        return getattr(self.adapter, 'last_usage', {})

    def _invoke(self, method, frames, fn):
        ledger, state = self.session.ledger, self.session.state
        row = ledger.begin_call(method, frames, self.model_id, mock=self.mock, accounting=self.accounting)
        started = time.monotonic()
        token = ACTIVE_CALL.set((ledger, row))
        try:
            if self.accounting != 'http':
                ledger.submit(row, len(frames))
            result = fn()
            row['outcome'] = 'success'
            for frame, key in zip(frames, row['source_keys']):
                ledger.observed.add(key)
                known = frame.frame_id in state.frames
                state.register_observed(frame.frame_id, key, row['call_id'])
                if not known:
                    # Metadata is measured; it does not assert image semantics.
                    state.add(source='program', kind='metadata', status='measured',
                              content=f'frame_id={frame.frame_id}; source_time_sec={frame.timestamp_sec}',
                              turn=row['call_id'], evidence_frame_ids=[frame.frame_id])
            return result, row['call_id']
        except BaseException as exc:
            row.update(outcome='error', error_type=type(exc).__name__)
            raise
        finally:
            row['elapsed_sec'] = round(time.monotonic() - started, 6)
            ACTIVE_CALL.reset(token)

    def inspect_frames(self, query, frames, context=''):
        relevant, judgments, summaries = [], [], []
        if not frames:
            raise ValueError('No frames supplied')
        for offset in range(0, len(frames), self.max_frames_per_call):
            chunk = frames[offset:offset + self.max_frames_per_call]
            result, turn = self._invoke('inspect_frames', chunk,
                                       lambda: self.adapter.inspect_frames(query, chunk, context))
            valid = {f.frame_id for f in chunk}
            source = 'mock' if self.mock else 'model_visual'
            for judgment in result.judgments:
                if judgment.frame_id not in valid:
                    continue
                self.session.state.add(source=source, kind='observation', status='observed',
                                       content=judgment.reason or f'与查询相关：{judgment.relevant}',
                                       evidence_frame_ids=[judgment.frame_id], turn=turn, model_id=self.model_id)
            if result.summary:
                self.session.state.add(source=source, kind='observation', status='observed',
                                       content=result.summary, evidence_frame_ids=[f.frame_id for f in chunk],
                                       turn=turn, model_id=self.model_id)
                summaries.append(result.summary)
            relevant.extend(fid for fid in result.relevant_frame_ids if fid in valid)
            judgments.extend(j for j in result.judgments if j.frame_id in valid)
        return InspectionResult(relevant_frame_ids=list(dict.fromkeys(relevant)),
                                judgments=judgments, summary=' | '.join(summaries))

    def _verification(self, result, turn, frames):
        value = result.model_dump() if hasattr(result, 'model_dump') else result
        score = value.get('confidence')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            score = None
        valid = {f.frame_id for f in frames}
        refs = value.get('evidence_frame_ids') or []
        if not isinstance(refs, list):
            refs = []
        refs = [fid for fid in refs if isinstance(fid, str) and fid in valid]
        if frames and not refs:
            refs = [f.frame_id for f in frames]
            value = {**value, 'reason': '未返回有效帧引用；下列帧仅记录本次视觉输入。' + str(value.get('reason', ''))}
        source = 'mock' if self.mock else 'model_visual' if frames else 'model_text'
        self.session.state.add(source=source, kind='verification', status='tentative',
                               content=f"{value.get('verdict', 'insufficient')}: {value.get('reason', '')}",
                               evidence_frame_ids=refs, turn=turn, model_id=self.model_id,
                               score=score, score_kind=('mock_hint_ratio' if self.mock else 'model_self_report') if score is not None else None)

    def verify_window(self, query, window, evidence):
        # Existing verify_window submits text only: do not count its frame IDs as images.
        result, turn = self._invoke('verify_window_text', [],
                                   lambda: self.adapter.verify_window(query, window, evidence))
        self._verification(result, turn, [])
        return result

    def verify_span(self, query, window, frames, envelope):
        if not frames or len(frames) > self.max_frames_per_call:
            raise ValueError('Invalid visual verification frame count')
        result, turn = self._invoke('verify_span', frames,
                                   lambda: self.adapter.verify_span(query, window, frames, envelope))
        self._verification(result, turn, frames)
        return result
