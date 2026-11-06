import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('scipy')
pytest.importorskip('sklearn')
from queryseek.retrieval.univtg_adapter import UniVTGHead, study_input
from queryseek.retrieval.localizer import localization_loss, supervision
from scripts.localizer_study import nested_selection, make_model, parameter_count, fit_study, verify


def entry(i, source, crop=0):
    return {'qid':i,'vid':f'{source}_{crop}.0_{crop+10}.0','duration':10.,
            'query':'visible object','relevant_windows':[[2.,6.]]}


def test_nested_identity_selection_retains_pilots_all_crop_queries_and_is_order_invariant():
    train = [entry(i,f't{i}') for i in range(8)]
    train += [dict(train[0],qid=100),entry(101,'t0',20)]
    val = [entry(20+i,f'v{i}') for i in range(4)]
    cfg = {'source_seed':7,'train_scales':[3,6],'validation_sources':3}
    selected = nested_selection(train,val,train[:1],val[:1],cfg,set())
    again = nested_selection(train[::-1],val[::-1],train[:1],val[:1],cfg,set())
    assert selected == again
    ids = lambda name: {e['qid'] for e in selected[name]}
    assert ids('train_3') <= ids('train_6')
    assert {0,100} <= ids('train_3') and 101 not in ids('train_6')
    assert len({e['vid'] for e in selected['validation']}) == 3
    with pytest.raises(ValueError,match='isolation'):
        nested_selection(train,val,[],[],cfg,{'t0'})
    with pytest.raises(ValueError,match='Insufficient'):
        nested_selection(train,val,[],[],{**cfg,'train_scales':[100]},set())


def test_retrained_controls_remove_semantics_and_inference_input_never_reads_gt():
    record = {'images':np.eye(2),'query_embedding':np.array([1,0]),'times':[0.,1.],'duration_sec':2.}
    full = study_input(record)
    no_query = study_input(record,'no_query')
    position = study_input(record,'position_only')
    assert torch.count_nonzero(no_query[:,2:7]) == 0
    assert torch.count_nonzero(position[:,:-2]) == 0
    assert torch.equal(full[:,-2:],position[:,-2:])
    assert torch.equal(full,study_input({**record,'entry':{'relevant_windows':[[999,1000]]}}))
    with pytest.raises(ValueError): study_input(record,'unknown')


def test_upstream_network_receives_finite_gradients_and_has_global_context():
    torch.set_num_threads(1); torch.manual_seed(4)
    model = UniVTGHead(2,8).eval()
    record = {'images':np.tile(np.eye(2),(5,1)),'query_embedding':np.array([1,0]),
              'times':np.arange(10.),'duration_sec':10.}
    x = study_input(record)
    logits,distances = model(x)
    y,d = supervision(record['times'],10.,[[2.,6.]])
    loss,_ = localization_loss(logits,distances,y,d,{'relevance':1,'offset':1,'iou':1})
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert (distances >= 0).all() and (distances <= 1).all()
    changed = x.clone(); changed[-1,:2] += 2
    assert not torch.allclose(model(x)[0][0],model(changed)[0][0],atol=1e-7)


def test_capacity_controls_match_trainable_parameters_without_changing_inputs():
    cfg = {'univtg_hidden':64}
    target = parameter_count(make_model('univtg',512,cfg))
    for variant in ('matched_mlp','matched_conv'):
        actual = parameter_count(make_model(variant,512,cfg))
        assert abs(actual-target)/target < .015
    assert parameter_count(make_model('univtg_no_query',512,cfg)) == target
    assert parameter_count(make_model('univtg_position_only',512,cfg)) == target


def test_checkpoint_selection_and_exact_reload_for_retrained_no_query(tmp_path):
    torch.set_num_threads(1)
    rr = [{'images':np.eye(2,dtype=np.float32).repeat(3,axis=0),
           'query_embedding':np.array([1,0],np.float32),'times':np.arange(6.),'duration_sec':6.,
           'entry':entry(i,f'source{i}')} for i in range(3)]
    for r in rr: r['entry']['relevant_windows'] = [[2.,5.]]
    config = {'head_seed':1,'univtg_hidden':8,'learning_rate':.001,'weight_decay':.0001,
              'epochs':2,'validation_every':1,'gradient_clip':1.}
    cfg = {'loss_weights':{'relevance':1,'offset':1,'iou':1},'top_k':5,'nms_iou':.7,'min_span_sec':.25}
    result = fit_study(rr[:2],rr[2:],'univtg_no_query',config,cfg,tmp_path/'fit')
    history = json.loads((tmp_path/'fit/history.json').read_text())
    winner = max(history,key=lambda row:row['validation_metrics']['MR-full-mAP'])
    assert result['selected_epoch'] == winner['epoch']
    assert result['checkpoint_reload_exact']
    assert len(history) == 2 and result['trained_epochs'] == 2


def test_freeze_verification_rejects_manifest_tampering(tmp_path,monkeypatch):
    import scripts.localizer_study as study
    from queryseek.ingest.ffprobe import sha256_file
    monkeypatch.setattr(study,'ROOT',tmp_path)
    fresh = tmp_path/'data/qvhighlights_fresh80/selection.json'
    fresh.parent.mkdir(parents=True); fresh.write_text('{}')
    run = tmp_path/'outputs/run'; run.mkdir(parents=True)
    selection = run/'train_64.jsonl'; selection.write_text('{}')
    protocol = {'fresh_sha256':sha256_file(fresh),'selection_sha256':{selection.name:sha256_file(selection)},'source_sha256':{}}
    (run/'protocol.json').write_text(json.dumps(protocol))
    verify(run)
    selection.write_text('changed')
    with pytest.raises(ValueError,match='selection'): verify(run)


def test_standard_generated_mock_fixture_has_source_identities(demo):
    from scripts.localizer_study import study_mock_examples
    from scripts.localizer_pilot import read
    from scripts.make_qvhighlights_subset import source_video_id
    from scripts.sampling_probe import ROOT
    records = study_mock_examples(demo['video'],read(ROOT/'configs/localizer_pilot_v1.json'))
    assert len(records) == 6
    assert len({source_video_id(r['entry']['vid']) for r in records}) == 6
    assert all(0 <= r['times'][0] < r['duration_sec'] for r in records)


def test_failed_validation_queries_are_empty_and_remain_metric_denominator():
    from scripts.localizer_study import evaluate
    torch.set_num_threads(1)
    record = {'images':np.eye(2,dtype=np.float32),'query_embedding':np.array([1,0],np.float32),
              'times':[0.,1.],'duration_sec':2.,'entry':entry(1,'source')}
    record['entry']['relevant_windows'] = [[0.,2.]]
    cfg = {'top_k':5,'nms_iou':.7,'min_span_sec':.25}
    gt = {1:record['entry'],2:dict(record['entry'],qid=2)}
    _,predictions,diagnostic = evaluate(UniVTGHead(2,8).eval(),[record],'full',cfg,gt)
    assert predictions[2] == []
    assert diagnostic['queries'] == 2 and diagnostic['available_queries'] == 1


@pytest.mark.parametrize('network_failure',[False,True])
def test_availability_quality_policy_keeps_selection_and_blocks_network_failures(tmp_path,monkeypatch,network_failure):
    import scripts.localizer_study as study
    monkeypatch.setattr(study,'ROOT',tmp_path)
    cfg = {'mirror_repo':'synthetic','mirror_revision':'fixed','max_video_bytes':1000}
    protocol = {'feature_config':cfg,'config':{'download_workers':1,'quality_rejections_allowed':True}}
    monkeypatch.setattr(study,'verify',lambda run:protocol)
    selected = [('train',entry(1,'a')),('validation',entry(2,'b'))]
    monkeypatch.setattr(study,'all_entries',lambda *args:selected)
    (tmp_path/'mirror_index.json').write_text('{}')
    monkeypatch.setattr(study,'acquire',lambda entry,cfg:(tmp_path/'synthetic',{}))
    def check(entry,path,cfg):
        if entry['qid']==1:
            if network_failure: raise TimeoutError('must not store network details')
            raise ValueError('Ground truth exceeds actual visual duration; no clipping allowed')
        return {'duration_sec':10.,'bytes':100}
    monkeypatch.setattr(study,'check_video',check)
    if network_failure:
        with pytest.raises(ValueError,match='availability gate'): study.prepare(tmp_path)
    else:
        study.prepare(tmp_path)
        assert json.loads((tmp_path/'status.json').read_text())['phase']=='prepared_with_quality_rejections'
    results = [json.loads(x) for x in (tmp_path/'availability.jsonl').read_text().splitlines()]
    assert {r['entry']['qid'] for r in results}=={1,2}
    assert sum(r['available'] for r in results)==1
    if network_failure: assert 'must not store' not in (tmp_path/'availability.jsonl').read_text()
