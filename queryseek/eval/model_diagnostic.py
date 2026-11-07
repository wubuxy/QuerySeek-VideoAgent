"""Paired reader diagnostics, explicitly distinct from autonomous retrieval scores."""
import json

from queryseek.retrieval.query_reader import choice_payload


def conditions():
    result = []
    for pixels, size in [(65536, 'low'), (262144, 'high')]:
        for constrained in [True, False]:
            result.append({'name': f'nf4_{size}_{"json" if constrained else "free"}',
                'precision': 'nf4', 'pixels': pixels, 'constrained': constrained,
                'input': 'oracle', 'prompt': 'minimal'})
    for mode in ['text', 'wrong', 'needs']:
        result.append({'name': 'nf4_high_'+mode, 'precision': 'nf4', 'pixels': 262144,
            'constrained': True, 'input': mode if mode != 'needs' else 'oracle',
            'prompt': 'needs' if mode == 'needs' else 'minimal'})
    result.append({'name': 'bf16_high_json', 'precision': 'bf16', 'pixels': 262144,
        'constrained': True, 'input': 'oracle', 'prompt': 'minimal'})
    return result


def minimal_prompt(request):
    payload, letters = choice_payload(request, True)
    return ('Answer the multiple-choice question using the supplied images. '
        'Return only a JSON object with one key, answer, whose value is one choice letter.\n'
        + payload, {'type': 'object', 'additionalProperties': False, 'required': ['answer'],
        'properties': {'answer': {'type': 'string', 'enum': letters}}})


def parse_answer(raw, choices):
    """No repair, markdown stripping, explanation extraction or answer substitution."""
    value = json.loads(raw)
    letters = [chr(65+i) for i in range(len(choices))]
    if not isinstance(value, dict) or set(value) != {'answer'} or value['answer'] not in letters:
        raise ValueError('Invalid answer contract')
    return value['answer']


def donor_indices(rows):
    if len(rows) < 2 or len({r['request']['video_id'] for r in rows}) != len(rows):
        raise ValueError('Wrong-image control requires distinct sources')
    return [(i+1) % len(rows) for i in range(len(rows))]


def assert_visual_transport(audit, image_count):
    grids = audit['image_grid_thw']
    if len(grids) != image_count or any(t <= 0 or h <= 0 or w <= 0 for t, h, w in grids):
        raise ValueError('Image grid cardinality mismatch')
    expected = sum(t*h*w//4 for t,h,w in grids)
    if audit['image_token_count'] != expected or audit['visual_tokens'] != expected:
        raise ValueError('Image placeholder/grid mismatch')
    if image_count:
        if not audit['pixel_tensor_sha256'] or audit['pixel_tensor_std'] <= 0:
            raise ValueError('Empty or constant visual tensor')
    elif audit['pixel_tensor_sha256'] is not None:
        raise ValueError('Text-only control has visual pixels')


def summarize(records, labels, condition_names, control_answers):
    ids = set(labels)
    expected = {(q, c) for q in ids for c in condition_names}
    main = [r for r in records if not r['control']]
    if len(main) != len(expected) or {(r['qid'],r['condition']) for r in main} != expected:
        raise ValueError('Incomplete or duplicate diagnostic denominator')
    metrics = {}
    for c in condition_names:
        rows = [r for r in main if r['condition'] == c]
        ok = [r for r in rows if r['status'] == 'ok']
        answered = [r for r in ok if r['answer'] != 'abstain']
        metrics[c] = {'n': len(ids), 'valid': len(ok), 'answered': len(answered),
            'correct': sum(r['answer'] == labels[r['qid']] for r in answered)}
    controls = {}
    for precision in ['nf4','bf16']:
        rows = [r for r in records if r['control'] and r['precision'] == precision]
        if len(rows) != len(control_answers) or {r['qid'] for r in rows} != set(control_answers):
            raise ValueError('Incomplete synthetic controls')
        controls[precision] = {'n': len(rows), 'valid': sum(r['status']=='ok' for r in rows),
            'correct': sum(r['status']=='ok' and r['answer']==control_answers[r['qid']] for r in rows)}
    comparisons = {}
    index = {(r['qid'],r['condition']):r for r in main}
    for a,b in [('nf4_low_json','nf4_high_json'),('nf4_low_json','nf4_low_free'),
                ('nf4_high_json','nf4_high_free'),('nf4_high_json','bf16_high_json'),
                ('nf4_high_json','nf4_high_text'),('nf4_high_json','nf4_high_wrong'),
                ('nf4_high_json','nf4_high_needs')]:
        paired = [(index[q,a],index[q,b]) for q in sorted(ids)]
        correct = lambda r: r['status']=='ok' and r['answer']==labels[r['qid']]
        comparisons[a+'__'+b] = {'n':len(paired),
            'answer_changed':sum(x['answer']!=y['answer'] for x,y in paired),
            'gained':sum(not correct(x) and correct(y) for x,y in paired),
            'lost':sum(correct(x) and not correct(y) for x,y in paired)}
    return {'metrics':metrics,'synthetic_controls':controls,'paired':comparisons,
            'scope':'Repeated development sources; privileged temporal evidence; no localization score.'}
