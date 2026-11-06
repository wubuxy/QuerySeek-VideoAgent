import json

import pytest

from scripts.prepare_localizer_split import identity_sources, source_disjoint_split, valid_localization_rows
from scripts.make_qvhighlights_subset import source_video_id


def entries():
    return [{'qid':i*2+j,'vid':f'source{i}_'+('0.0_10.0' if j==0 else '10.0_20.0'),
             'query':'A visible event.','duration':10.,'relevant_windows':[[1.,5.]]}
            for i in range(10) for j in range(2)]


def test_source_group_split_excludes_all_crops_and_has_no_train_val_overlap():
    train,val = source_disjoint_split(entries(),{'source3'},42,.2)
    assert len(train)+len(val) == 18
    a={source_video_id(e['vid']) for e in train}; b={source_video_id(e['vid']) for e in val}
    assert not a&b and 'source3' not in a|b
    assert (train,val)==source_disjoint_split(list(reversed(entries())),{'source3'},42,.2)
    with pytest.raises(ValueError):
        source_disjoint_split(entries(),{f'source{i}' for i in range(10)},42,.2)


def test_metadata_time_contract_invalid_labels_and_duplicate_qids_are_not_silently_used():
    base = entries()[0]
    candidates = [base,{**base,'qid':21,'relevant_windows':[[1.,11.]]},
                  {**base,'qid':22,'vid':'../escape'},base,{**base,'qid':23,'relevant_windows':[]}]
    good,bad = valid_localization_rows(candidates)
    assert good == [base] and len(bad) == 4


def test_independent_selection_only_identity_fields_can_affect_exclusion(tmp_path):
    path=tmp_path/'selection.json'
    path.write_text(json.dumps({'entries':[{'vid':'source_0.0_10.0','query':'DO NOT USE','relevant_windows':[[1,2]]}]}))
    first=identity_sources(path,'fresh')
    path.write_text(json.dumps({'entries':[{'vid':'source_10.0_20.0','query':'changed query','relevant_windows':[[3,4]]}]}))
    assert first==identity_sources(path,'fresh')=={'source'}
