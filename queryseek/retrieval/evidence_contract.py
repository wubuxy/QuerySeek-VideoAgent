"""Atomic evidence decisions with parser-enforced cross-field constraints (v2)."""
from copy import deepcopy
from itertools import combinations
from queryseek.retrieval.evidence_packets import verification_prompt,validate_verdict


def strict_prompt(request,packet):
    prompt,old=verification_prompt(request,packet)
    letters=[x for x in old['properties']['answer']['enum'] if x!='abstain']
    count=len(packet['frames'])
    common={'type':'object','additionalProperties':False,'required':['decision','evidence','observed_fact'],
        'properties':{'observed_fact':deepcopy(old['properties']['observed_fact'])}}
    branches=[]
    decisions=[['supported:'+x for x in letters],['contradicted'],
               ['insufficient:'+x for x in ['visual_detail','event_relation','temporal_extent','global_scope']]]
    evidence=[[f'{a},{b}' for a,b in combinations(range(count),2)],
              [str(i) for i in range(count)],['']]
    for ds,refs in zip(decisions,evidence):
        branch=deepcopy(common)
        branch['properties'].update(decision={'type':'string','enum':ds},evidence={'type':'string','enum':refs})
        branches.append(branch)
    prompt+='\nOUTPUT ENCODING: Return only decision, evidence, observed_fact. '
    prompt+='Encode support as decision="supported:A" (substitute the choice letter), with evidence="i,j" '
    prompt+='for exactly two distinct local image indices in increasing order. Encode explicit contradiction '
    prompt+='as decision="contradicted" with evidence="i" for one index. Encode insufficiency as '
    prompt+='decision="insufficient:visual_detail", "insufficient:event_relation", '
    prompt+='"insufficient:temporal_extent" or "insufficient:global_scope", with evidence="". '
    prompt+='These strings encode the state, answer and missing reason above; do not emit those as separate fields.'
    return prompt,{'oneOf':branches}


def decode_verdict(value,packet,choices,duration):
    if set(value)!={'decision','evidence','observed_fact'}:raise ValueError('Invalid atomic decision keys')
    decision=value['decision'];evidence=value['evidence']
    if not isinstance(decision,str) or not isinstance(evidence,str):raise ValueError('Invalid decision type')
    if decision.startswith('supported:'):
        state,answer,missing='supported',decision.split(':',1)[1],'none'
        refs=evidence.split(',')
        if len(refs)!=2 or any(not x.isdigit() for x in refs):raise ValueError('Expected evidence pair')
        refs=[int(x) for x in refs]
        if refs[0]>=refs[1] or evidence!=f'{refs[0]},{refs[1]}':raise ValueError('Noncanonical evidence pair')
    elif decision=='contradicted':
        state,answer,missing='contradicted','abstain','none'
        if not evidence.isdigit() or str(int(evidence))!=evidence:raise ValueError('Expected one evidence index')
        refs=[int(evidence)]
    elif decision.startswith('insufficient:'):
        state,answer,missing='insufficient','abstain',decision.split(':',1)[1]
        if evidence:raise ValueError('Insufficiency must not pretend to cite support')
        refs=[]
    else:raise ValueError('Unknown atomic decision')
    return validate_verdict({'state':state,'answer':answer,'missing_reason':missing,
        'evidence_frames':refs,'observed_fact':value['observed_fact']},packet,choices,duration)
