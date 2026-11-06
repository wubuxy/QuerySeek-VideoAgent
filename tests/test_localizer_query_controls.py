import json
import time

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('scipy')
from scripts.localizer_query_controls import fit_control, infer, make_model, study_input, verify_controls


def records():
    return [{'images':np.eye(2,dtype=np.float32).repeat(3,axis=0),
             'query_embedding':np.array([1.,0.],np.float32), 'times':np.arange(6.),'duration_sec':6.,
             'entry':{'qid':i,'vid':f'source{i}_0.0_6.0','query':'object','relevant_windows':[[2.,5.]]}}
            for i in range(3)]


def config():
    return {'head_seed':5,'univtg_hidden':8,'epochs':2,'learning_rate':.001,
            'weight_decay':.0001,'gradient_clip':1.}


def features():
    return {'loss_weights':{'relevance':1,'offset':1,'iou':1},'top_k':5,'nms_iou':.7,'min_span_sec':.25}


@pytest.mark.parametrize('variant',['matched_mlp','matched_conv'])
@pytest.mark.parametrize('control',['no_query','position_only'])
def test_retraining_uses_last_epoch_no_validation_and_roundtrips(tmp_path,variant,control):
    torch.set_num_threads(1)
    rr = records(); out = tmp_path/'fit'
    result = fit_control(rr,variant,control,config(),features(),out,time.perf_counter()+60)
    assert result['trained_epochs'] == 2 and result['checkpoint_reload_exact']
    history = json.loads((out/'history.json').read_text())
    assert all(set(row)=={'epoch','train_query_mean_loss'} for row in history)
    checkpoint = torch.load(out/'last.pt',weights_only=True)
    assert checkpoint['epoch'] == 2 and checkpoint['control'] == control
    model = make_model(variant,2,config()); model.load_state_dict(checkpoint['state_dict'])
    original = infer(model,rr,control,features())
    # Replacing query text/vector has no effect on retrained query-free arms.
    changed = [{**r,'query_embedding':np.array([0.,1.],np.float32),
                'entry':{**r['entry'],'query':'other','relevant_windows':[[999.,1000.]]}} for r in rr]
    assert infer(model,changed,control,features()) == original
    if control=='position_only':
        changed = [{**r,'images':np.flip(r['images'],axis=1).copy()} for r in changed]
        assert infer(model,changed,control,features()) == original
    for predictions in original[0].values():
        assert set(predictions)=={0,1,2}
        assert all(0 <= a < b <= 6 for pp in predictions.values() for a,b,_ in pp)


def test_training_budget_cap_retains_failure_and_no_checkpoint(tmp_path):
    torch.set_num_threads(1)
    with pytest.raises(TimeoutError,match='wall-time'):
        fit_control(records(),'matched_mlp','no_query',config(),features(),tmp_path/'fit',0.)
    assert not (tmp_path/'fit/last.pt').exists()


def test_full_inference_is_gt_invariant_but_query_input_changes():
    torch.set_num_threads(1); torch.manual_seed(4)
    rr = records(); model = make_model('matched_mlp',2,config())
    changed = [{**r,'entry':{**r['entry'],'relevant_windows':[[999.,1000.]]}} for r in rr]
    assert infer(model,rr,'full',features()) == infer(model,changed,'full',features())
    other = {**rr[0],'query_embedding':np.array([0.,1.],np.float32)}
    assert not torch.equal(study_input(rr[0]),study_input(other))


def test_preserved_artifact_tamper_is_rejected(tmp_path,monkeypatch):
    import scripts.localizer_query_controls as controls
    from queryseek.ingest.ffprobe import sha256_file
    monkeypatch.setattr(controls,'ROOT',tmp_path)
    old = tmp_path/'old.pt'; old.write_bytes(b'checkpoint')
    out = tmp_path/'run'; out.mkdir()
    (out/'protocol.json').write_text(json.dumps({'source_sha256':{},'preserved_sha256':{'old.pt':sha256_file(old)}}))
    old.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen input'):
        verify_controls(out)
