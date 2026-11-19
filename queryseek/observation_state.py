"""Append-only evidence state: provenance is independent of semantic truth."""
from __future__ import annotations

from typing import Literal
import uuid
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StateRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    id: str
    source: Literal['program', 'model_visual', 'model_text', 'mock', 'subtitle', 'retrieval']
    kind: Literal['metadata', 'observation', 'hypothesis', 'gap', 'verification', 'retrieval_hint']
    status: Literal['measured', 'observed', 'tentative', 'supported', 'refuted']
    content: str
    evidence_frame_ids: tuple[str, ...] = ()
    created_turn: int = Field(ge=0)
    model_id: str | None = None
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    score_kind: Literal['model_self_report', 'mock_hint_ratio', 'retrieval_similarity'] | None = None
    supersedes: str | None = None
    semantic_verified: Literal[False] = False

    @model_validator(mode='after')
    def validate_provenance(self):
        if self.status == 'measured' and (self.source != 'program' or self.kind != 'metadata'):
            raise ValueError('Only program metadata may be marked measured')
        if self.kind == 'hypothesis' and self.status not in {'tentative', 'supported', 'refuted'}:
            raise ValueError('Hypotheses must remain explicitly labelled')
        if (self.score is None) != (self.score_kind is None):
            raise ValueError('A score must state what it measures')
        return self


class ObservationState:
    def __init__(self):
        self._frames: dict[str, str] = {}  # Only successfully observed frame IDs -> source keys.
        self._source_turns: dict[str, int] = {}
        self._records: list[StateRecord] = []

    @property
    def frames(self):
        return MappingProxyType(self._frames)

    def register_observed(self, frame_id: str, source_key: str, turn: int = 0):
        if type(turn) is not int or turn < 0:
            raise ValueError('Invalid observation turn')
        if frame_id in self.frames and self.frames[frame_id] != source_key:
            raise ValueError('Frame ID changed its source coordinate')
        self._frames[frame_id] = source_key
        self._source_turns.setdefault(source_key, turn)

    def add(self, *, source, kind, status, content, turn, evidence_frame_ids=(),
            model_id=None, score=None, score_kind=None):
        if status in {'supported', 'refuted'}:
            raise ValueError('Use an evidence-bearing revision for hypothesis updates')
        refs = tuple(dict.fromkeys(evidence_frame_ids))
        if any(fid not in self.frames for fid in refs):
            raise ValueError('Cannot cite an unobserved frame')
        if turn < max((*self._source_turns.values(), *(r.created_turn for r in self._records)), default=0):
            raise ValueError('Cannot backdate evidence state')
        if source == 'model_visual' and not refs:
            raise ValueError('Visual claims require observed image references')
        record = StateRecord(id='obs_' + uuid.uuid4().hex[:16], source=source, kind=kind,
                             status=status, content=content, evidence_frame_ids=refs,
                             created_turn=turn, model_id=model_id, score=score, score_kind=score_kind)
        self._records.append(record)
        return record

    def revise(self, record_id, *, status, content, new_evidence_frame_ids, turn):
        previous = next((r for r in self.current() if r.id == record_id), None)
        if not previous or previous.kind != 'hypothesis' or status not in {'supported', 'refuted'}:
            raise ValueError('Only a current hypothesis may be updated')
        if turn <= previous.created_turn:
            raise ValueError('A revision must follow its original observation')
        if turn < max((*self._source_turns.values(), *(r.created_turn for r in self._records)), default=0):
            raise ValueError('Cannot backdate evidence state')
        refs = tuple(dict.fromkeys(new_evidence_frame_ids))
        if not refs or any(fid not in self.frames for fid in refs):
            raise ValueError('Revision requires newly observed evidence')
        old_sources = {self.frames[fid] for fid in previous.evidence_frame_ids}
        fresh_sources = {self.frames[fid] for fid in refs} - old_sources
        if not any(previous.created_turn < self._source_turns[key] <= turn for key in fresh_sources):
            raise ValueError('Repeated references or aliases cannot confirm a hypothesis')
        revised = StateRecord(id='obs_' + uuid.uuid4().hex[:16], source=previous.source,
                              kind='hypothesis', status=status, content=content,
                              evidence_frame_ids=tuple(dict.fromkeys((*previous.evidence_frame_ids, *refs))),
                              created_turn=turn, model_id=previous.model_id, supersedes=previous.id)
        self._records.append(revised)
        return revised

    def current(self):
        replaced = {r.supersedes for r in self._records if r.supersedes}
        return [r for r in self._records if r.id not in replaced]

    def snapshot(self):
        return [r.model_dump(mode='json') for r in self._records]
