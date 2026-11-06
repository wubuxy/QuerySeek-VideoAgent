import numpy as np
import pytest
torch=pytest.importorskip('torch')
from queryseek.retrieval.token_localizer import (make_network,make_criterion,training_target,
    collate_inputs,collate_targets,total_loss,decode_dense)
from queryseek.vendor.univtg.univtg import SetCriterion


def entries():
    return {'relevant_windows':[[2.,6.]],'relevant_clip_ids':[1,2],
            'saliency_scores':[[2,3,4],[4,4,4]]}


def records():
    rng=np.random.default_rng(1)
    rr=[{'qid':i,'images':rng.normal(size=(8+i,8)).astype(np.float32),
         'times':np.arange(8+i,dtype=float),'duration_sec':10.,'irrelevant_gt':[[-10,100]]} for i in range(2)]
    text={0:rng.normal(size=(3,8)).astype(np.float32),1:rng.normal(size=(5,8)).astype(np.float32)}
    return rr,text


def test_half_open_targets_and_saliency_are_mapped_to_real_pts():
    t=np.array([0.,1.999,2.,3.9,4.,5.99,6.,8.])
    target=training_target(t,10.,entries())
    assert target['timestamp_window'].tolist()==[False,False,True,True,True,True,False,False]
    np.testing.assert_allclose(target['timestamp'][:,0],t/10,rtol=1e-6)
    np.testing.assert_allclose(target['saliency_scores'],[0,0,3,3,4,4,0,0])


@pytest.mark.parametrize('mode',['outside','no_positive','saliency_shape','bad_time'])
def test_invalid_supervision_is_rejected_without_fabricating_foreground(mode):
    e=entries();t=np.arange(10,dtype=float)
    if mode=='outside':e['relevant_windows']=[[2.,11.]]
    elif mode=='no_positive':e['relevant_windows']=[[2.1,2.2]]
    elif mode=='saliency_shape':e['saliency_scores']=[[1,2]]
    else:t[1]=0
    with pytest.raises(ValueError):training_target(t,10.,e)


def test_token_batch_preserves_variable_lengths_and_does_not_accept_gt():
    rr,text=records();x=collate_inputs(rr,text)
    assert x['src_vid'].shape==(2,9,10) and x['src_txt'].shape==(2,5,8)
    assert x['src_txt_mask'].sum(1).tolist()==[3,5]
    assert x['src_vid_mask'].sum(1).tolist()==[8,9]
    np.testing.assert_allclose(x['src_vid'][0,:8,-2],rr[0]['times']/10,atol=1e-7)
    for r in rr:r['irrelevant_gt']=[[1.,2.]]
    altered=collate_inputs(rr,text)
    assert all(torch.equal(x[k],altered[k]) for k in x)


def test_chunked_original_giou_loss_and_gradients_match_official_criterion():
    rr,text=records();rng=np.random.default_rng(1)
    targets={r['qid']:training_target(r['times'],r['duration_sec'],entries()) for r in rr}
    y=collate_targets(rr,targets,rng)
    offsets=torch.zeros(2,9,2);offsets[:,:,0]=-.2;offsets[:,:,1]=.2
    a=offsets.clone().requires_grad_();b=offsets.clone().requires_grad_()
    ours=make_criterion();official=SetCriterion(None,ours.weight_dict,.1,['spans'],.07,'l1',256)
    first=ours.loss_spans({'pred_spans':a},y,None);second=official.loss_spans({'pred_spans':b},y,None)
    for key in first:torch.testing.assert_close(first[key],second[key],rtol=1e-6,atol=1e-7)
    sum(first.values()).backward();sum(second.values()).backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=1e-6,atol=1e-7)


def test_all_five_losses_active_finite_gradients_and_model_reload(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(1);rr,text=records();rng=np.random.default_rng(1)
    targets={r['qid']:training_target(r['times'],r['duration_sec'],entries()) for r in rr}
    x=collate_inputs(rr,text);y=collate_targets(rr,targets,rng)
    spec={'hidden':16,'heads':4,'layers':1,'ffn':32};model=make_network(spec,embedding_dim=8)
    loss,parts=total_loss(make_criterion(),model(**x),y);loss.backward()
    assert set(parts)=={'loss_f','loss_b','loss_g','loss_s_inter','loss_s_intra'}
    assert all(torch.isfinite(v) for v in parts.values())
    assert model.weightedpool.weight.grad is not None and model.span_embed.layers[-1].weight.grad is not None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    path=tmp_path/'own.pt';torch.save(model.state_dict(),path)
    restored=make_network(spec,embedding_dim=8);restored.load_state_dict(torch.load(path,weights_only=True));model.eval();restored.eval()
    with torch.inference_mode():
        for key in ('pred_logits','pred_spans','saliency_scores'):torch.testing.assert_close(model(**x)[key],restored(**x)[key],rtol=0,atol=0)


def test_dense_decode_uses_signed_offsets_actual_seconds_and_preserves_bounds():
    r={'times':np.array([0.,2.,4.]),'duration_sec':5.}
    p=np.array([.2,.8,.5]);o=np.array([[-.1,.2],[-.2,.4],[-.2,.8]])
    selected,pool=decode_dense(r,p,o)
    assert pool==[(0.,1.,.2),(1.,4.,.8),(3.,5.,.5)]
    assert selected[0]==(1.,4.,.8)
    assert all(isinstance(a,float) and 0<=a<b<=5. for a,b,_ in pool)
    with pytest.raises(ValueError):decode_dense(r,p,np.abs(o))


def test_capacity_changes_trainable_parameters_without_changing_token_interface():
    a=make_network({'hidden':16,'heads':4,'layers':1,'ffn':32},8)
    b=make_network({'hidden':32,'heads':4,'layers':2,'ffn':128},8)
    assert sum(p.numel() for p in b.parameters() if p.requires_grad)>sum(p.numel() for p in a.parameters() if p.requires_grad)
    rr,text=records();x=collate_inputs(rr,text)
    assert a(**x)['pred_spans'].shape==b(**x)['pred_spans'].shape==(2,9,2)
