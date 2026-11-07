"""Question representations and candidate-bound answers; no evaluator labels."""
from __future__ import annotations

import json
import numpy as np

from queryseek.eval.grounded_qa import checked_intervals
from queryseek.retrieval.query_refinement import relevance

METHODS = ('raw', 'token_chunks', 'event_queries')
ORACLE_CONDITIONS = ('oracle_list_letter', 'oracle_labeled_json', 'oracle_labeled_letter')
TEMPORAL = ('none', 'first', 'before_after', 'duration', 'causal', 'multiple')


def planner_prompt(question):
    prompt = (
        'Rewrite the question into 1 to 3 short visually observable event descriptions for image retrieval. '
        'Describe only entities, actions and context explicitly mentioned in the question. '
        'Do not answer the question, guess missing attributes, resolve alternative claims, or invent timestamps. '
        'Keep the event that identifies where to look; do not replace it with generic question words. '
        'Record temporal requirements separately; these descriptions alone do not enforce event order. '
        'Each description and answer_focus must be at most 120 characters. Return JSON only.\n'
        + json.dumps({'question': question}, ensure_ascii=False))
    schema = {'type': 'object', 'additionalProperties': False,
        'required': ['queries', 'temporal_requirement', 'answer_focus'], 'properties': {
        'queries': {'type': 'array', 'minItems': 1, 'maxItems': 3,
                    'items': {'type': 'string', 'minLength': 1, 'maxLength': 120}},
        'temporal_requirement': {'type': 'string', 'enum': list(TEMPORAL)},
        'answer_focus': {'type': 'string', 'minLength': 1, 'maxLength': 120}}}
    return prompt, schema


def validate_plan(plan):
    if (set(plan) != {'queries', 'temporal_requirement', 'answer_focus'}
            or not isinstance(plan['queries'], list) or not 1 <= len(plan['queries']) <= 3
            or any(not isinstance(q, str) or not 1 <= len(q.strip()) <= 120 for q in plan['queries'])
            or len(set(plan['queries'])) != len(plan['queries'])
            or plan['temporal_requirement'] not in TEMPORAL
            or not isinstance(plan['answer_focus'], str) or not 1 <= len(plan['answer_focus'].strip()) <= 120):
        raise ValueError('Invalid question-only plan')
    return plan


def token_chunks(question, tokenizer, width=60, max_chunks=3):
    """Use the official CLIP tokenizer; no silent second truncation."""
    ids = tokenizer.encode(question)
    if not ids or len(ids) > width * max_chunks:
        raise ValueError('Question exceeds frozen chunk budget')
    if len(ids) <= width:
        return [question]
    chunks = [tokenizer.decode(ids[i:i+width]).strip() for i in range(0, len(ids), width)]
    if any(not q or len(tokenizer.encode(q))+2 > tokenizer.context_length for q in chunks):
        raise ValueError('Decoded chunk exceeds CLIP context')
    return chunks


def query_scores(observation, vectors):
    if len(vectors) == 0:
        raise ValueError('At least one query vector required')
    # An OR-like retrieval heuristic, not a conjunction/order verifier.
    return np.max(np.stack([relevance(observation, v) for v in vectors]), axis=0)


def candidates_for(windows, frames, duration, limit=2):
    result = []
    for i, w in enumerate(windows[:limit]):
        a, b = checked_intervals([[w.start_sec, w.end_sec]], duration)[0]
        times = [float(f.timestamp_sec) for f in frames if a <= f.timestamp_sec < b]
        if times:
            result.append({'candidate_id': f'c{i}', 'interval_sec': [a, b],
                           'displayed_source_times_sec': times})
    return result


def choice_payload(request, labeled):
    letters = [chr(65+i) for i in range(len(request['choices']))]
    if not 2 <= len(letters) <= 8:
        raise ValueError('Expected 2 to 8 choices')
    if labeled:
        text = request['question'] + '\n' + '\n'.join(
            f'{letter}. {choice}' for letter, choice in zip(letters, request['choices']))
    else:
        text = json.dumps({'question': request['question'], 'choices': request['choices'],
                           'choice_letters': letters}, ensure_ascii=False)
    return text, letters


def reader_prompt(request, condition, candidates=None):
    bound = condition in METHODS
    labeled = bound or 'labeled' in condition
    letter_only = condition.endswith('_letter')
    payload, letters = choice_payload(request, labeled)
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['answer'],
              'properties': {'answer': {'type': 'string', 'enum': letters}}}
    if bound:
        if candidates is None or len({c['candidate_id'] for c in candidates}) != len(candidates):
            raise ValueError('Explicit unique candidates required')
        prompt = (
            'Answer using the supplied sparse video observations. Choose one candidate interval that '
            'contains the evidence for your answer. These are sparse observations, not a continuous clip. '
            'Do not treat an unobserved event as absent. If the displayed evidence is insufficient, '
            'return answer="abstain" and candidate_id="none". Otherwise return a choice letter and '
            'one of the supplied candidate IDs. Return JSON with answer and candidate_id only.\n')
        payload += '\nCandidates (seconds, half-open intervals):\n' + json.dumps(candidates)
        schema['properties']['answer']['enum'] = letters + ['abstain']
        schema['properties']['candidate_id'] = {'type': 'string',
            'enum': [c['candidate_id'] for c in candidates] + ['none']}
        schema['required'].append('candidate_id')
    elif letter_only:
        prompt = 'Answer the multiple-choice question. Return only one choice letter.\n'
    else:
        prompt = 'Answer the multiple-choice question. Return one JSON object with answer (one choice letter).\n'
    return prompt + payload, schema, letter_only


def bind_answer(prediction, candidates, choices, duration):
    letters = [chr(65+i) for i in range(len(choices))]
    if set(prediction) != {'answer', 'candidate_id'}:
        raise ValueError('Invalid bound answer keys')
    answer, cid = prediction['answer'], prediction['candidate_id']
    if answer == 'abstain' and cid == 'none':
        return {**prediction, 'abstained': True, 'intervals': []}
    lookup = {c['candidate_id']: c for c in candidates}
    if answer not in letters or cid not in lookup:
        raise ValueError('Answer/candidate inconsistency')
    intervals = checked_intervals([lookup[cid]['interval_sec']], duration)
    return {**prediction, 'abstained': False, 'intervals': [list(x) for x in intervals]}


def letter_token_ids(tokenizer, letters):
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in letters]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != len(letters):
        raise ValueError('Choices must be distinct single tokens')
    return [x[0] for x in ids]
