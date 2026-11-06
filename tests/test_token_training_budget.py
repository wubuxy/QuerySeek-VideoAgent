import copy
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from scripts import token_training_budget as budget
from scripts import token_localizer_study as base
from queryseek.ingest.cache import write_json_atomic


def configs():
    old=base.read(base.ROOT/'configs/token_localizer_v1.json')
    c=base.read(base.ROOT/'configs/token_training_budget_v1.json')
    return old,c


def test_prefix_rejects_changed_learning_rate_data_or_optimizer():
    old,c=configs();new=budget.child_protocol({'config':old},c,'delayed')['config']
    plan=[{'qids':[1],'positive_positions':[0]}]*800
    budget.check_prefix(old,new,plan,plan*2,400)
    with pytest.raises(ValueError,match='Learning-rate'):budget.check_prefix(old,new,plan,plan*2,401)
    changed=copy.deepcopy(new);changed['weight_decay']=.5
    with pytest.raises(ValueError,match='Non-schedule'):budget.check_prefix(old,changed,plan,plan*2,400)
    altered=copy.deepcopy(plan*2);altered[399]['qids']=[2]
    with pytest.raises(ValueError,match='Sampling'):budget.check_prefix(old,new,plan,altered,400)


def test_schedules_isolate_steps_and_decay_with_unchanged_warmup():
    old,c=configs();p={'config':old}
    early=budget.child_protocol(p,c,'early')['config'];delayed=budget.child_protocol(p,c,'delayed')['config']
    assert p['config']['steps']==800 and early['steps']==delayed['steps']==1600
    assert early['warmup_steps']==delayed['warmup_steps']==40
    for i in range(1600):assert base.learning_rate(early,i)==base.learning_rate(old,i)
    assert base.learning_rate(delayed,799)==1e-4
    assert np.isclose(base.learning_rate(delayed,800),1e-5)
    assert np.isclose(base.learning_rate(delayed,1280),1e-6)


def test_seed_copy_is_read_only_and_changed_seed_is_rejected(tmp_path):
    old,c=configs();parent=tmp_path/'parent';parent.mkdir();out=tmp_path/'new';out.mkdir()
    plan=[{'qids':[1],'positive_positions':[0]}]*800
    write_json_atomic(parent/'sampling_plan_256.json',plan);write_json_atomic(out/'sampling_plan.json',plan*2)
    source=parent/'models/256/large/step_0400.pt';source.parent.mkdir(parents=True);source.write_bytes(b'old optimizer and rng')
    p={'config':c,'parent_protocol':{'config':old}}
    root,_=budget.seed_child(parent,out,p,'delayed')
    target=root/'models/256/large/step_0400.pt'
    assert target.read_bytes()==source.read_bytes()
    assert base.read(root/'models/256/large/progress.json')['checkpoint_step']==400
    budget.seed_child(parent,out,p,'delayed')
    target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Seed checkpoint'):budget.seed_child(parent,out,p,'delayed')
    assert source.read_bytes()==b'old optimizer and rng'


def test_snapshot_routing_uses_parent_only_for_identical_prefix(tmp_path):
    _,cfg=configs();cfg['parent_run']=str(tmp_path/'parent');p={'config':cfg};out=tmp_path/'new'
    assert budget.snapshot_path(out,p,'early',800)==tmp_path/'parent/models/256/large/step_0800.pt'
    assert budget.snapshot_path(out,p,'delayed',400)==tmp_path/'parent/models/256/large/step_0400.pt'
    assert budget.snapshot_path(out,p,'delayed',600)==out/'arms/delayed/models/256/large/step_0600.pt'


def test_frozen_plan_and_notes_tamper_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(budget,'ROOT',tmp_path);monkeypatch.setattr(base,'verify_protocol',lambda _:None)
    out=tmp_path/'outputs/run';out.mkdir(parents=True)
    (out/'sampling_plan.json').write_text('[]');(out/'protocol_notes_frozen.md').write_text('frozen')
    p={'config':{'parent_run':'parent'},'parent_artifacts_sha256':{},'source_sha256':{},
       'plan_sha256':budget.sha256_file(out/'sampling_plan.json'),'notes_sha256':budget.sha256_file(out/'protocol_notes_frozen.md')}
    write_json_atomic(out/'protocol.json',p);budget.verify(out)
    (out/'sampling_plan.json').write_text('[1]')
    with pytest.raises(ValueError,match='plan/notes'):budget.verify(out)


def test_synthetic_continuation_matches_direct_training_and_legal_seconds(tmp_path,monkeypatch):
    import scripts.sampling_probe as sampling
    monkeypatch.setattr(sampling,'ROOT',tmp_path)
    result=budget.mock(tmp_path/'outputs/mock')
    assert result['mock'] and result['continuation_equals_uninterrupted']
    assert result['fixed_eval_losses_repeatable'] and all(np.isfinite(v) for v in result['fixed_eval_losses'].values())
    for pred in result['predictions'].values():
        assert len(pred)==4
        for windows in pred.values():
            assert len(windows)<=5
            assert all(0<=a<b<=12. and all(isinstance(v,float) for v in (a,b,s)) for a,b,s in windows)
