import copy
from scripts.localizer_study import nested_selection


def test_source_extension_preserves_prefix_and_all_validation_queries():
    train=[{'qid':i,'vid':f'train{i}_0.0_20.0'} for i in range(9)]
    val=[{'qid':100+i,'vid':f'val{i}_0.0_20.0'} for i in range(3)]
    cfg={'source_seed':42,'train_scales':[2,4],'validation_sources':2}
    small=nested_selection(train,val,train[:1],val[:1],cfg,[])
    cfg=copy.deepcopy(cfg);cfg['train_scales']=[4,8]
    large=nested_selection(train,val,train[:1],val[:1],cfg,[])
    assert small['train_4']==large['train_4'] and small['validation']==large['validation']
    assert {r['qid'] for r in small['train_4']} < {r['qid'] for r in large['train_8']}
