"""Per-run VLM admission limits and honest attempt-level cost accounting.

Source identity is video SHA256 + source seconds rounded to six decimals (the
existing FFmpeg interface precision). Public timestamps remain float seconds.
The deadline prevents new work; it does not claim to preempt in-flight requests.
"""
from __future__ import annotations

from contextvars import ContextVar
import copy
import math
import re
import threading
import time

from pydantic import BaseModel, ConfigDict, Field

from .models.interface import VLMError


class ObservationBudget(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    max_unique_frames: int = Field(default=32, ge=1, le=4096, strict=True)
    max_model_calls: int = Field(default=8, ge=0, strict=True)
    max_http_attempts: int = Field(default=32, ge=0, strict=True)
    max_image_presentations: int | None = Field(default=None, ge=0, strict=True)
    max_total_tokens: int | None = Field(default=None, ge=0, strict=True)
    max_output_tokens_per_call: int = Field(default=512, ge=1, strict=True)
    query_deadline_sec: float = Field(default=180, gt=0, allow_inf_nan=False)


class ObservationBudgetExceeded(VLMError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__('Observation budget: ' + reason)


ACTIVE_SESSION = ContextVar('queryseek_observation_session', default=None)
ACTIVE_CALL = ContextVar('queryseek_observation_call', default=None)


def _count(value):
    return value if type(value) is int and value >= 0 else None


def parse_usage(raw):
    usage = raw if isinstance(raw, dict) else {}
    prompt = _count(usage.get('prompt_tokens'))
    output = _count(usage.get('completion_tokens'))
    total = _count(usage.get('total_tokens'))
    if total is None and prompt is not None and output is not None:
        total = prompt + output
    if total is not None and prompt is not None and output is not None and total < prompt + output:
        total = None
    details = usage.get('prompt_tokens_details')
    image = _count(details.get('image_tokens')) if isinstance(details, dict) else None
    if image is not None and (prompt is None or image > prompt):
        image = None
    return {'input_tokens': prompt, 'output_tokens': output, 'total_tokens': total,
            'visual_input_tokens': image,
            'text_input_tokens': prompt - image if prompt is not None and image is not None else None}


class ObservationLedger:
    def __init__(self, limits, video_sha256, duration_sec, *, clock=time.monotonic):
        if not re.fullmatch('[0-9a-fA-F]{64}', video_sha256):
            raise ValueError('Invalid video SHA256')
        if not math.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError('Invalid video duration')
        self.limits = limits
        self.video_sha256 = video_sha256.lower()
        self.duration_sec = duration_sec
        self.clock = clock
        self.started = clock()
        self.calls: list[dict] = []
        self.attempts: list[dict] = []
        self.submitted: set[str] = set()
        self.observed: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.rejections: list[str] = []
        self.presentations = 0
        self._lock = threading.RLock()

    def reject(self, reason):
        if reason not in self.rejections:
            self.rejections.append(reason)
        raise ObservationBudgetExceeded(reason)

    def source_key(self, frame):
        if not math.isfinite(frame.timestamp_sec) or not 0 <= frame.timestamp_sec < self.duration_sec:
            raise ValueError('Frame outside source video')
        return f'{self.video_sha256}@{frame.timestamp_sec:.6f}'

    def _deadline(self):
        if self.clock() - self.started >= self.limits.query_deadline_sec:
            self.reject('query_deadline')

    def begin_call(self, method, frames, model_id, *, mock=False, accounting='unknown'):
        with self._lock:
            self._deadline()
            if len(self.calls) >= self.limits.max_model_calls:
                self.reject('max_model_calls')
            # Current API backends provide no exact preflight image token counter.
            # Refuse an unsupported strict total-token cap, rather than fake one.
            if self.limits.max_total_tokens is not None and not mock:
                self.reject('token_preflight_unavailable')
            sources = [self.source_key(f) for f in frames]
            coordinates = dict(self.aliases)
            for frame, key in zip(frames, sources):
                if frame.frame_id in coordinates and coordinates[frame.frame_id] != key:
                    raise ValueError('Frame ID changed its source coordinate')
                coordinates[frame.frame_id] = key
            if len(self.submitted | set(sources)) > self.limits.max_unique_frames:
                self.reject('max_unique_frames')
            if (self.limits.max_image_presentations is not None
                    and self.presentations + len(frames) > self.limits.max_image_presentations):
                self.reject('max_image_presentations')
            if accounting == 'http' and len(self.attempts) >= self.limits.max_http_attempts:
                self.reject('max_http_attempts')
            row = {'call_id': len(self.calls) + 1, 'method': method, 'model_id': model_id,
                   'frame_ids': [f.frame_id for f in frames],
                   'source_times_sec': [float(f.timestamp_sec) for f in frames],
                   'source_keys': sources, 'mock': mock, 'accounting': accounting, 'outcome': 'running'}
            self.calls.append(row)
            return row

    def submit(self, call, n_images, *, http=False):
        with self._lock:
            self._deadline()
            if n_images != len(call['source_keys']):
                raise ValueError('Image count and source references disagree')
            if http and len(self.attempts) >= self.limits.max_http_attempts:
                self.reject('max_http_attempts')
            sources = set(call['source_keys'])
            if len(self.submitted | sources) > self.limits.max_unique_frames:
                self.reject('max_unique_frames')
            if (self.limits.max_image_presentations is not None
                    and self.presentations + n_images > self.limits.max_image_presentations):
                self.reject('max_image_presentations')
            self.submitted.update(sources)
            self.presentations += n_images
            self.aliases.update(zip(call['frame_ids'], call['source_keys']))
            if http:
                row = {'attempt_id': len(self.attempts) + 1, 'call_id': call['call_id'],
                       'n_images': n_images, 'outcome': 'running'}
                self.attempts.append(row)
                return row

    def finish_attempt(self, row, *, status, error_type, usage, elapsed_sec):
        with self._lock:
            row.update(status=status, error_type=error_type,
                       elapsed_sec=round(elapsed_sec, 6),
                       outcome='success' if status is not None and 200 <= status < 300 and not error_type else 'error',
                       **parse_usage(usage))

    def summary(self):
        unknown = any(c['accounting'] == 'unknown' for c in self.calls)
        def total(field):
            if any(c['accounting'] == 'unknown' for c in self.calls):
                return None
            if any(a.get(field) is None for a in self.attempts):
                return None
            return sum(a[field] for a in self.attempts)
        return {'unique_source_frames': len(self.submitted),
                'unique_observed_source_frames': len(self.observed),
                'image_presentations': self.presentations,
                'model_calls': len(self.calls),
                'http_attempts': None if unknown else len(self.attempts),
                **{k: total(k) for k in ('input_tokens', 'output_tokens', 'total_tokens',
                                        'visual_input_tokens', 'text_input_tokens')},
                'mock': bool(self.calls) and all(c['mock'] for c in self.calls),
                'cost_scope': 'vlm_only',
                'transport_accounting_complete': not unknown,
                'presentation_basis': 'adapter_inputs' if unknown else 'attempted_submissions',
                'remaining_unique_frames': max(0, self.limits.max_unique_frames - len(self.submitted)),
                'remaining_model_calls': max(0, self.limits.max_model_calls - len(self.calls)),
                'remaining_http_attempts': None if unknown else max(0, self.limits.max_http_attempts - len(self.attempts)),
                'peak_gpu_memory_bytes': None,
                'query_elapsed_sec': round(self.clock() - self.started, 6),
                'deadline_exceeded': self.clock() - self.started >= self.limits.query_deadline_sec,
                'budget_rejections': list(self.rejections)}

    def snapshot(self):
        return {'version': 'observation-cost-v1', 'source_identity': 'video_sha256+source_seconds_6dp',
                'limits': self.limits.model_dump(), 'summary': self.summary(),
                'calls': copy.deepcopy(self.calls), 'http_attempts': copy.deepcopy(self.attempts)}
