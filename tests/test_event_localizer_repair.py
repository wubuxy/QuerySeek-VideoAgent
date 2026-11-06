import pytest
torch=pytest.importorskip('torch')
from scripts import event_localizer_repair as repair
from scripts import token_localizer_study as base


def test_trainer_injection_restores_frozen_bindings_on_failure():
    previous=(base.make_network,base.make_criterion)
    with pytest.raises(RuntimeError):
        with repair.trainer_bindings('event_relative'):
            assert base.make_criterion is not previous[1]
            raise RuntimeError('interrupted')
    assert (base.make_network,base.make_criterion)==previous


def test_warm_start_factory_preserves_parent_weight_bytes_and_rng():
    spec={'hidden':16,'heads':4,'layers':1,'ffn':32};torch.manual_seed(8);parent=base.make_network(spec)
    weights=parent.state_dict();rng=torch.get_rng_state()
    with repair.trainer_bindings('event_relative',weights,rng):
        model=base.make_network(spec)
        assert base.state_sha(model.core)==base.state_sha(parent)
        assert torch.equal(torch.get_rng_state(),rng)


def test_sanity_selection_is_train_only_stratified_and_source_distinct():
    records=[{'qid':i,'source':str(i)} for i in range(10)]
    entries={i:{'relevant_windows':[[0.,6. if i<5 else 60.]]} for i in range(10)}
    selected=repair.sanity_selection(records,entries,4)
    assert [r['qid'] for r in selected]==[0,1,2,3,5,6,7,8]
    with pytest.raises(ValueError):repair.sanity_selection(records,entries,6)


def test_sanity_disables_dropout_but_retains_gradient_training():
    model=repair.SanityModel({'hidden':16,'heads':4,'layers':1,'ffn':32})
    model.train();assert not model.training and all(not m.training for m in model.modules())
    assert any(p.requires_grad for p in model.parameters())


def test_mock_runs_both_objectives_from_same_weights_with_variable_lengths(tmp_path,monkeypatch):
    import scripts.sampling_probe as sampling
    monkeypatch.setattr(sampling,'ROOT',tmp_path)
    result=repair.mock(tmp_path/'outputs/mock')
    assert result['mock'] and result['two_modes_train_and_reload']
