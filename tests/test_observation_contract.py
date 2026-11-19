"""Evidence provenance, source-time budgets and real transport accounting, offline."""
import json

import httpx
import pytest
from pydantic import ValidationError

from queryseek.models.compatible_api import CompatibleAPIVLM
from queryseek.models.metered import ObservationSession
from queryseek.models.mock import MockVLM
from queryseek.observation_budget import ObservationBudget, ObservationBudgetExceeded, ObservationLedger
from queryseek.observation_state import ObservationState, StateRecord
from queryseek.schemas import VideoFrame, TimeWindow


@pytest.fixture()
def source(tmp_path, demo):
    # Adapter unit fixture: MockTransport never decodes or sends these bytes.
    image = tmp_path / 'input.jpg'
    image.write_bytes(b'private-image-bytes-not-for-trace')
    from queryseek.ingest.ffprobe import asset_from_probe
    asset = asset_from_probe('unit', demo['video'])
    return asset, lambda fid, t: VideoFrame(frame_id=fid, timestamp_sec=t, image_path=str(image))


def api_adapter(handler, **kwargs):
    return CompatibleAPIVLM(api_key='secret-only-in-header', base_url='https://unit.invalid/v1',
                            model='unit-vlm', client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def response(content=None, usage=True):
    obj = {'choices': [{'message': {'content': content or '{"relevant_frame_ids": [], "summary": "ok"}'}}]}
    if usage:
        obj['usage'] = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
    return obj


def test_provenance_is_not_truth_and_revisions_need_new_sources():
    state = ObservationState()
    state.register_observed('one', 'sha@1', 1)
    state.register_observed('alias', 'sha@1', 1)
    hypothesis = state.add(source='model_visual', kind='hypothesis', status='tentative',
                           content='可能正在放入杯子', turn=1, evidence_frame_ids=['one'])
    original = state.snapshot()
    for refs in (['one'], ['alias'], ['unseen']):
        with pytest.raises(ValueError):
            state.revise(hypothesis.id, status='supported', content='confirmed',
                         new_evidence_frame_ids=refs, turn=2)
    state.register_observed('two', 'sha@2', 2)
    revised = state.revise(hypothesis.id, status='supported', content='新增画面支持该假设',
                            new_evidence_frame_ids=['two'], turn=2)
    assert revised.source == 'model_visual' and revised.semantic_verified is False
    assert hypothesis.status == 'tentative' and original[0] == state.snapshot()[0]
    assert state.current() == [revised]
    with pytest.raises(ValueError):
        state.revise(hypothesis.id, status='refuted', content='stale revision',
                     new_evidence_frame_ids=['two'], turn=3)
    with pytest.raises(TypeError):
        state.frames['one'] = 'other'
    with pytest.raises(ValidationError):
        revised.content = 'mutated'


def test_cannot_promote_self_report_or_cite_unobserved_frame():
    state = ObservationState()
    with pytest.raises(ValueError):
        state.add(source='model_visual', kind='observation', status='observed',
                  content='claim', turn=1, evidence_frame_ids=['unseen'])
    with pytest.raises(ValueError):
        state.add(source='model_text', kind='verification', status='supported', content='yes', turn=1)
    with pytest.raises(ValidationError):
        StateRecord(id='x', source='model_text', kind='metadata', status='measured',
                    content='model says so', created_turn=1)
    with pytest.raises(ValidationError):
        StateRecord(id='x', source='model_text', kind='verification', status='tentative',
                    content='yes', created_turn=1, semantic_verified=True)


def test_old_unused_frames_and_backdated_state_cannot_promote_hypotheses():
    state = ObservationState()
    state.register_observed('old', 'sha@1', 1)
    state.register_observed('unused', 'sha@2', 1)
    hypothesis = state.add(source='model_visual', kind='hypothesis', status='tentative',
                           content='possible action', turn=2, evidence_frame_ids=['old'])
    with pytest.raises(ValueError):
        state.revise(hypothesis.id, status='supported', content='unsupported upgrade',
                     new_evidence_frame_ids=['unused'], turn=3)
    state.register_observed('fresh', 'sha@3', 4)
    with pytest.raises(ValueError, match='backdate'):
        state.add(source='model_text', kind='gap', status='tentative', content='gap', turn=3)


def test_same_source_different_frame_ids_charge_once_but_every_presentation(source):
    asset, frame = source
    session = ObservationSession(ObservationBudget(max_unique_frames=1), asset)
    vlm = session.wrap(MockVLM())
    vlm.inspect_frames('query', [frame('first', 1)])
    vlm.inspect_frames('query', [frame('alias', 1.00000000001)])
    with pytest.raises(ObservationBudgetExceeded, match='max_unique_frames'):
        vlm.inspect_frames('query', [frame('new', 2)])
    summary = session.ledger.summary()
    assert summary['unique_source_frames'] == 1
    assert summary['image_presentations'] == 2
    assert summary['model_calls'] == 2 and summary['http_attempts'] == 0
    snapshot = session.ledger.snapshot()
    snapshot['calls'][0]['frame_ids'].append('future')
    assert session.ledger.calls[0]['frame_ids'] == ['first']


def test_frame_id_conflicts_rejected_before_dispatch(source):
    asset, frame = source
    session = ObservationSession(ObservationBudget(), asset)
    vlm = session.wrap(MockVLM())
    with pytest.raises(ValueError, match='coordinate'):
        vlm.inspect_frames('q', [frame('same', 1), frame('same', 2)])
    assert session.ledger.summary()['model_calls'] == 0


def test_missing_usage_is_unknown_not_zero(source):
    asset, frame = source
    session = ObservationSession(ObservationBudget(), asset)
    vlm = session.wrap(api_adapter(lambda request: httpx.Response(200, json=response(usage=False))))
    with session.activate():
        vlm.inspect_frames('query', [frame('one', 1)])
    summary = session.ledger.summary()
    assert summary['input_tokens'] is None and summary['total_tokens'] is None
    assert summary['http_attempts'] == 1 and summary['image_presentations'] == 1


def test_http_retries_are_charged_without_claiming_unreported_tokens(source, monkeypatch):
    asset, frame = source
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(429, json={}) if len(requests) == 1 else httpx.Response(200, json=response())
    monkeypatch.setattr('queryseek.models.compatible_api.time.sleep', lambda _: None)
    session = ObservationSession(ObservationBudget(max_http_attempts=2), asset)
    with session.activate():
        session.wrap(api_adapter(handler)).inspect_frames('q', [frame('one', 1)])
    summary = session.ledger.summary()
    assert summary['http_attempts'] == 2 and summary['model_calls'] == 1
    assert summary['image_presentations'] == 2 and summary['unique_source_frames'] == 1
    assert summary['total_tokens'] is None
    assert session.ledger.attempts[1]['total_tokens'] == 15
    serialized = json.dumps(session.ledger.snapshot())
    assert 'secret-only-in-header' not in serialized and 'private-image-bytes' not in serialized


def test_http_attempt_limit_stops_retry_before_network(source, monkeypatch):
    asset, frame = source
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(500, json={})
    monkeypatch.setattr('queryseek.models.compatible_api.time.sleep', lambda _: None)
    session = ObservationSession(ObservationBudget(max_http_attempts=2), asset)
    with session.activate(), pytest.raises(ObservationBudgetExceeded, match='max_http_attempts'):
        session.wrap(api_adapter(handler)).inspect_frames('q', [frame('one', 1)])
    assert len(requests) == 2
    assert session.ledger.summary()['image_presentations'] == 2


def test_chunks_and_effective_output_cap_are_measured(source):
    asset, frame = source
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=response())
    session = ObservationSession(ObservationBudget(max_output_tokens_per_call=128), asset)
    with session.activate():
        session.wrap(api_adapter(handler)).inspect_frames('q', [frame(str(i), i / 10) for i in range(17)])
    summary = session.ledger.summary()
    assert summary['model_calls'] == 3 and summary['http_attempts'] == 3
    assert summary['image_presentations'] == 17
    assert summary['input_tokens'] == 30 and summary['output_tokens'] == 15
    assert summary['visual_input_tokens'] is None and summary['text_input_tokens'] is None
    assert all(p['max_tokens'] == 128 for p in requests)


def test_invalid_model_json_still_charges_returned_usage(source):
    asset, frame = source
    session = ObservationSession(ObservationBudget(), asset)
    with session.activate(), pytest.raises(Exception):
        session.wrap(api_adapter(lambda r: httpx.Response(200, json=response('not json')))).inspect_frames('q', [frame('one', 1)])
    assert session.ledger.summary()['total_tokens'] == 15
    assert session.ledger.summary()['unique_observed_source_frames'] == 0
    assert session.state.frames == {}


def test_text_verification_does_not_count_evidence_as_image_inputs(source):
    asset, frame = source
    session = ObservationSession(ObservationBudget(), asset)
    vlm = session.wrap(MockVLM([{'start_sec': 0., 'end_sec': 5., 'label': 'cup'}]))
    verdict = vlm.verify_window('cup', TimeWindow(start_sec=0, end_sec=5),
                                [{'frame_id': 'unseen', 'type': 'frame'}])
    assert verdict.verdict == 'confirmed'
    assert session.ledger.summary()['image_presentations'] == 0
    record = session.state.snapshot()[-1]
    assert record['status'] == 'tentative' and not record['semantic_verified']
    assert record['evidence_frame_ids'] == [] and record['score_kind'] == 'mock_hint_ratio'


def test_strict_token_limit_without_estimator_is_refused_before_network(source):
    asset, frame = source
    session = ObservationSession(ObservationBudget(max_total_tokens=10), asset)
    def must_not_send(request):
        pytest.fail('strict token preflight must stop before HTTP')
    with session.activate(), pytest.raises(ObservationBudgetExceeded, match='token_preflight_unavailable'):
        session.wrap(api_adapter(must_not_send)).inspect_frames('q', [frame('one', 1)])
    assert session.ledger.summary()['model_calls'] == 0
    assert session.ledger.summary()['http_attempts'] == 0


def test_admission_deadline_and_image_budget(source):
    asset, frame = source
    now = [0.]
    ledger = ObservationLedger(ObservationBudget(query_deadline_sec=1), asset.sha256, 10, clock=lambda: now[0])
    now[0] = 2
    with pytest.raises(ObservationBudgetExceeded, match='query_deadline'):
        ledger.begin_call('inspect', [frame('one', 1)], 'mock', mock=True)
    assert not ledger.calls
    session = ObservationSession(ObservationBudget(max_image_presentations=1), asset)
    vlm = session.wrap(MockVLM())
    vlm.inspect_frames('q', [frame('one', 1)])
    with pytest.raises(ObservationBudgetExceeded, match='max_image_presentations'):
        vlm.inspect_frames('q', [frame('one', 1)])
    assert session.ledger.summary()['model_calls'] == 1


@pytest.mark.parametrize('raw, expected', [
    ({}, (None, None, None)),
    ({'prompt_tokens': True, 'completion_tokens': -1}, (None, None, None)),
    ({'prompt_tokens': 10, 'completion_tokens': 5}, (10, 5, 15)),
    ({'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 2}, (10, 5, None)),
])
def test_usage_counters_are_strict_and_missing_values_remain_unknown(raw, expected):
    from queryseek.observation_budget import parse_usage
    usage = parse_usage(raw)
    assert (usage['input_tokens'], usage['output_tokens'], usage['total_tokens']) == expected


def test_explicit_visual_token_details_are_kept_separate():
    from queryseek.observation_budget import parse_usage
    assert parse_usage({'prompt_tokens': 100, 'prompt_tokens_details': {'image_tokens': 80}}) == {
        'input_tokens': 100, 'output_tokens': None, 'total_tokens': None,
        'visual_input_tokens': 80, 'text_input_tokens': 20}
