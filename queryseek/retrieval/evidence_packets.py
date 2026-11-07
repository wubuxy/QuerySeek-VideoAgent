"""Candidate-local observations and bounded evidence decisions; no labels or LLM timestamps."""
from __future__ import annotations

import json
from queryseek.eval.grounded_qa import checked_intervals
from queryseek.retrieval.query_refinement import padded_window
from queryseek.retrieval.query_reader import choice_payload

STATES = ('supported', 'contradicted', 'insufficient')
MISSING = ('none', 'visual_detail', 'event_relation', 'temporal_extent', 'global_scope')


def propose_packets(retrieval, duration, max_seeds=3, seed_width=64.):
    """Keep top five; uncovered displayed frames seed search windows, never proof."""
    if retrieval['status'] != 'ok':
        raise ValueError('Successful frozen retrieval required')
    candidates = []
    for w in retrieval['windows'][:5]:
        span = checked_intervals([[w['start_sec'], w['end_sec']]], duration)[0]
        candidates.append({'candidate_id': f'c{len(candidates)}', 'interval_sec': list(span),
                           'origin': 'retrieval', 'seed_time_sec': None})
    scores = dict(zip(retrieval['observed_source_times_sec'], retrieval['scores'], strict=True))
    review = retrieval['review_source_times_sec']
    if any(t not in scores or not 0 <= t < duration for t in review):
        raise ValueError('Seed must be an actually observed frame')
    for t in sorted(review, key=lambda t: (-scores[t], t)):
        if sum(c['origin']=='seed_window' for c in candidates) >= max_seeds:
            break
        if any(c['interval_sec'][0] <= t < c['interval_sec'][1] for c in candidates):
            continue
        span = padded_window([max(0., t-.000001), min(duration, t+.000001)], duration, 0., seed_width)
        candidates.append({'candidate_id': f'c{len(candidates)}', 'interval_sec': list(span),
                           'origin': 'seed_window', 'seed_time_sec': float(t)})
    remaining = [t for t in review if not any(c['interval_sec'][0] <= t < c['interval_sec'][1] for c in candidates)]
    return candidates, remaining


def packet_frames_valid(packet, duration):
    a,b = checked_intervals([packet['interval_sec']], duration)[0]
    times = [f['timestamp_sec'] for f in packet['frames']]
    if (len(times) < 2 or len(times) != len(packet['frame_sha256'])
            or any(not a <= t < b for t in times) or any(x >= y for x,y in zip(times,times[1:]))):
        raise ValueError('Packet frames must be distinct, ordered and inside the candidate')
    return times


def joint_view(packets, duration, reverse=False):
    frames, candidates = [], []
    for packet in reversed(packets) if reverse else packets:
        times = packet_frames_valid(packet, duration)
        start = len(frames);frames.extend(packet['frames'])
        candidates.append({'candidate_id': packet['candidate_id'], 'interval_sec': packet['interval_sec'],
            'displayed_source_times_sec': times, 'frame_indices': list(range(start, len(frames)))})
    return frames, candidates


def verification_prompt(request, packet):
    payload, letters = choice_payload(request, True)
    times = packet_frames_valid(packet, request['duration_sec'])
    prompt = (
        'Inspect ONLY this candidate and its sparse observed images. Preserve the full question: '
        'subject, action, object, attributes, ordinal conditions, before/after and duration all matter. '
        'A familiar object alone does not verify the event. Missing images do not prove absence. '
        'supported means the event relation AND a choice answer are supported by these observations. '
        'contradicted requires an explicit visible conflict with the requested event, not failure to see it. '
        'Otherwise use insufficient. First/last across the whole video cannot be established from a local '
        'clip alone; report global_scope when that broader coverage is needed. Duration needs visible '
        'boundaries; do not infer it from the candidate length. The interval is a search hypothesis. '
        'Cite local image indices that support the observed fact. supported requires at least two distinct '
        'indices, a choice answer, and missing_reason=none. contradicted requires at least one index, '
        'answer=abstain and missing_reason=none. insufficient requires answer=abstain and a non-none '
        'missing_reason. Give a short concrete observed_fact, not reasoning steps. Return JSON only.\n'
        + payload + '\n' + json.dumps({'candidate_id':packet['candidate_id'],
            'interval_sec':packet['interval_sec'], 'source_times_sec':times}))
    schema = {'type':'object','additionalProperties':False,
        'required':['state','answer','evidence_frames','missing_reason','observed_fact'],
        'properties':{'state':{'type':'string','enum':list(STATES)},
            'answer':{'type':'string','enum':letters+['abstain']},
            'evidence_frames':{'type':'array','maxItems':len(times),
                'items':{'type':'integer','enum':list(range(len(times)))}},
            'missing_reason':{'type':'string','enum':list(MISSING)},
            'observed_fact':{'type':'string','minLength':1,'maxLength':240}}}
    return prompt,schema


def validate_verdict(value, packet, choices, duration):
    times = packet_frames_valid(packet,duration)
    if set(value) != {'state','answer','evidence_frames','missing_reason','observed_fact'}:
        raise ValueError('Invalid verifier keys')
    refs = value['evidence_frames'];state = value['state'];missing = value['missing_reason']
    letters = [chr(65+i) for i in range(len(choices))]
    if (state not in STATES or missing not in MISSING or not isinstance(refs,list)
            or any(type(i) is not int or not 0 <= i < len(times) for i in refs) or len(set(refs)) != len(refs)
            or not isinstance(value['observed_fact'],str) or not 1 <= len(value['observed_fact'].strip()) <= 240):
        raise ValueError('Invalid evidence references or state')
    if state == 'supported':
        valid = value['answer'] in letters and missing == 'none' and len(refs)>=2
    elif state == 'contradicted':
        valid = value['answer']=='abstain' and missing=='none' and len(refs)>=1
    else:
        valid = value['answer']=='abstain' and missing!='none'
    if not valid:
        raise ValueError('Inconsistent evidence state/answer')
    return {**value, 'evidence_source_times_sec':[times[i] for i in refs]}


def aggregate(packets, verdicts):
    """Rank is frozen before inference; conflicting supported answers cause abstention."""
    if len({p['candidate_id'] for p in packets}) != len(packets):
        raise ValueError('Duplicate candidate identity')
    if len(verdicts) != len(packets) or {v['candidate_id'] for v in verdicts} != {p['candidate_id'] for p in packets}:
        raise ValueError('Incomplete verifier denominator')
    supported = {v['candidate_id']:v for v in verdicts if v['status']=='ok' and v['state']=='supported'}
    answers = {v['answer'] for v in supported.values()}
    result = {'status':'ok','answer':'abstain','candidate_id':'none','abstained':True,'intervals':[]}
    if len(answers)>1:
        return {**result, 'reason':'conflicting_supported_answers'}
    if not supported:
        return {**result, 'reason':'no_supported_candidate'}
    packet = next(p for p in packets if p['candidate_id'] in supported)
    verdict = supported[packet['candidate_id']]
    return {**result, 'answer':verdict['answer'],'candidate_id':packet['candidate_id'],'abstained':False,
            'intervals':[packet['interval_sec']], 'reason':'supported_claim',
            'evidence_source_times_sec':verdict['evidence_source_times_sec']}


def next_action(packets, verdicts, observation_budget):
    decision = aggregate(packets,verdicts)
    if not decision['abstained']:
        return {'action':'answer','candidate_id':decision['candidate_id']}
    if decision['reason']=='conflicting_supported_answers':
        return {'action':'abstain','reason':decision['reason']}
    lookup = {v['candidate_id']:v for v in verdicts}
    eligible = [p for p in packets if lookup[p['candidate_id']]['status']=='ok'
        and lookup[p['candidate_id']]['state']=='insufficient'
        and lookup[p['candidate_id']]['missing_reason']!='global_scope']
    if eligible and observation_budget>0:
        return {'action':'observe_more','candidate_id':eligible[0]['candidate_id']}
    return {'action':'abstain', 'reason':'observation_budget_exhausted' if eligible else 'no_locally_resolvable_candidate'}
