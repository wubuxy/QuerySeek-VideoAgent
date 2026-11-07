"""Preserve partial visual evidence and route one observation to the missing fact."""
from itertools import combinations
import json

from queryseek.eval.grounded_qa import checked_intervals
from queryseek.retrieval.evidence_packets import aggregate, packet_frames_valid
from queryseek.retrieval.query_reader import choice_payload

REGIONS = {'full': (0., 0., 1., 1.), 'upper_left': (0., 0., .6, .6),
           'upper_right': (.4, 0., 1., .6), 'lower_left': (0., .4, .6, 1.),
           'lower_right': (.4, .4, 1., 1.), 'center': (.2, .2, .8, .8)}
NEEDS = ('target_unobserved', 'target_ambiguous', 'relation_unverified',
         'attribute_unreadable', 'temporal_unverified', 'global_unverified')


def plan_prompt(question):
    prompt = ('Specify the evidence required to answer this question, without answering it. '
        'Use only entities and relations in the question. Do not invent attributes or hints. '
        'target is the object whose attribute is requested; relation identifies its ownership, '
        'action or context. scope=single_view when a clear view can establish the requested attribute '
        '(including a count within one visible scene); event identity must still be checked. '
        'scope=temporal for order, duration or change across time; scope=global for first/last or '
        'counts over the whole video. Return only target, relation, scope.\n' + json.dumps({'question': question}))
    return prompt, {'type': 'object', 'additionalProperties': False,
        'required': ['target', 'relation', 'scope'], 'properties': {
            'target': {'type': 'string', 'minLength': 1, 'maxLength': 160},
            'relation': {'type': 'string', 'minLength': 1, 'maxLength': 160},
            'scope': {'type': 'string', 'enum': ['single_view', 'temporal', 'global']}}}


def validate_plan(value):
    if (set(value) != {'target', 'relation', 'scope'}
            or value['scope'] not in ('single_view', 'temporal', 'global')
            or any(not isinstance(value[k], str) or not 1 <= len(value[k].strip()) <= 160
                   for k in ['target', 'relation'])):
        raise ValueError('Invalid evidence requirement plan')
    return value


def views(packet, duration):
    packet_frames_valid(packet, duration)
    base = packet['frames']; extra = packet.get('detail_views', [])
    if len(extra) > 1:
        raise ValueError('Only one detail view allowed')
    for view in extra:
        i = view['parent_index']
        if (type(i) is not int or not 0 <= i < len(base) or view['region'] not in REGIONS
                or view['region'] == 'full' or view['frame']['timestamp_sec'] != base[i]['timestamp_sec']
                or view['parent_sha256'] != packet['frame_sha256'][i]):
            raise ValueError('Detail view must preserve parent identity and source PTS')
    return base + [v['frame'] for v in extra]


def evidence_options(packet, duration, scope):
    frames = views(packet, duration)
    singles = [str(i) for i in range(len(frames))]
    pairs = [f'{a},{b}' for a, b in combinations(range(len(frames)), 2)
             if frames[a]['timestamp_sec'] != frames[b]['timestamp_sec']]
    return singles, pairs if scope == 'temporal' else singles


def verdict_prompt(request, packet, plan):
    validate_plan(plan); payload, letters = choice_payload(request, True)
    frames = views(packet, request['duration_sec'])
    singles, answer_refs = evidence_options(packet, request['duration_sec'], plan['scope'])
    common = {'target_description': {'type': 'string', 'minLength': 1, 'maxLength': 160},
              'observed_fact': {'type': 'string', 'minLength': 1, 'maxLength': 240}}
    branches = []
    def branch(decisions, refs, regions):
        branches.append({'type': 'object', 'additionalProperties': False,
            'required': ['decision', 'evidence', 'region', 'target_description', 'observed_fact'],
            'properties': {**common, 'decision': {'type': 'string', 'enum': decisions},
                           'evidence': {'type': 'string', 'enum': refs},
                           'region': {'type': 'string', 'enum': regions}}})
    branch(['target_unobserved'], [''], ['full'])
    for need in NEEDS[1:]:
        branch([need], singles, list(REGIONS)[1:] if need in ('target_ambiguous', 'attribute_unreadable')
               else list(REGIONS))
    branch(['contradicted'], singles, list(REGIONS))
    if plan['scope'] != 'global':
        branch(['answer:' + x for x in letters], answer_refs, list(REGIONS))
    prompt = ('Inspect the actual images for this candidate. The question is authoritative; the text-only '
        'evidence plan may be imperfect. First identify WHICH visible object is requested, distinguishing '
        'nearby objects. Describe its visible location and appearance in target_description. Then check '
        'ownership, interaction and contextual relations; only then read/count its requested attribute. '
        'Do not treat a familiar interface or object as proof of its owner or event.\n'
        'decision states: target_unobserved = no target observed, not proof of absence; '
        'target_ambiguous = plausible objects but requested identity unclear; relation_unverified = '
        'target/interface visible but its requested owner, action or context unverified; '
        'attribute_unreadable = target and relation identified but requested detail unreadable; '
        'temporal_unverified = relevant evidence visible but time/order/boundaries missing; '
        'global_unverified = full-video coverage missing; contradicted needs a visible explicit conflict. '
        'answer:X requires both the requested event/ownership and the chosen answer to be supported. '
        'A single clear frame can support a static attribute; do not require repetition. Temporal answers '
        'require two distinct source times. These are evidence requirements, not guarantees of truth.\n'
        'Keep partial evidence even when unable to answer: evidence is a local image index (or an ordered '
        'pair for temporal answers). Only target_unobserved uses an empty evidence string. '
        'For ambiguous/unreadable targets select the coarse region to inspect more closely. '
        'Regions are full, upper_left, upper_right, lower_left, lower_right, center, defined by normalized '
        'boxes below. A crop and its full image are the SAME source moment, not two temporal observations. '
        'observed_fact is a short concrete visual fact, not reasoning steps. Return JSON only.\n'
        + payload + '\n' + json.dumps({'evidence_plan': plan, 'candidate_id': packet['candidate_id'],
          'interval_sec': packet['interval_sec'], 'regions_xyxy': REGIONS,
          'views': [{'index': i, 'source_time_sec': f['timestamp_sec'],
                     'detail': i >= len(packet['frames'])} for i, f in enumerate(frames)]}))
    return prompt, {'oneOf': branches}


def decode_verdict(value, request, packet, plan):
    _, schema = verdict_prompt(request, packet, plan)
    keys = {'decision', 'evidence', 'region', 'target_description', 'observed_fact'}
    if set(value) != keys or any(not isinstance(value[k], str) for k in keys):
        raise ValueError('Invalid evidence-needs fields')
    matches = [b for b in schema['oneOf'] if all(value[k] in b['properties'][k]['enum']
               for k in ['decision', 'evidence', 'region'])]
    if len(matches) != 1 or any(not 1 <= len(value[k].strip()) <= maximum
                               for k, maximum in [('target_description', 160), ('observed_fact', 240)]):
        raise ValueError('Invalid cross-field evidence-needs contract')
    refs = [int(i) for i in value['evidence'].split(',')] if value['evidence'] else []
    supported = value['decision'].startswith('answer:')
    return {**value, 'state': 'supported' if supported else
            ('contradicted' if value['decision'] == 'contradicted' else 'insufficient'),
            'answer': value['decision'].split(':')[1] if supported else 'abstain',
            'missing_reason': 'none' if supported or value['decision'] == 'contradicted' else value['decision'],
            'evidence_frames': refs,
            'evidence_source_times_sec': [views(packet, request['duration_sec'])[i]['timestamp_sec'] for i in refs]}


def select_action(packets, verdicts, budget, policy):
    if policy not in ('rank_uniform', 'evidence_routed'):
        raise ValueError('Unknown observation policy')
    result = aggregate(packets, verdicts)
    if not result['abstained']:
        return {'action': 'answer', 'candidate_id': result['candidate_id']}
    if result['reason'] == 'conflicting_supported_answers':
        return {'action': 'abstain', 'reason': result['reason']}
    lookup = {v['candidate_id']: v for v in verdicts}
    eligible = [p for p in packets if lookup[p['candidate_id']]['status'] == 'ok'
                and lookup[p['candidate_id']]['state'] == 'insufficient'
                and lookup[p['candidate_id']]['missing_reason'] != 'global_unverified']
    if not eligible or budget <= 0:
        return {'action': 'abstain', 'reason': 'budget_exhausted' if eligible else 'no_local_action'}
    priorities = {'attribute_unreadable': 3, 'relation_unverified': 3, 'temporal_unverified': 3,
                  'target_ambiguous': 2, 'target_unobserved': 0}
    packet = (max(eligible, key=lambda p: priorities[lookup[p['candidate_id']]['missing_reason']])
              if policy == 'evidence_routed' else eligible[0])
    v = lookup[packet['candidate_id']]; need = v['missing_reason']
    action = 'uniform'
    if policy == 'evidence_routed':
        if need in ('attribute_unreadable', 'target_ambiguous'): action = 'detail'
        elif need in ('relation_unverified', 'temporal_unverified'): action = 'context'
    result = {'action': action, 'candidate_id': packet['candidate_id'], 'need': need}
    if action != 'uniform':
        result.update(anchor_index=v['evidence_frames'][0], region=v['region'])
    return result


def context_interval(anchor_sec, duration, before=24., after=8.):
    return list(checked_intervals([[max(0., anchor_sec-before), min(duration, anchor_sec+after)]], duration)[0])
