import numpy as np
import pytest
torch=pytest.importorskip('torch')
from queryseek.retrieval.masked_event_localizer import MaskedLocalizer,criterion,event_weights
from queryseek.retrieval.token_localizer import make_network,collate_inputs,collate_targets,training_target,total_loss


SPEC={'hidden':32,'heads':4,'layers':1,'ffn':64}

def examples():
    rng=np.random.default_rng(6);records=[];text={};targets={}
    for i,n in enumerate((12,20)):
        r={'qid':i,'times':np.arange(n,dtype=float),'duration_sec':float(n),'images':rng.normal(size=(n,512)).astype(np.float32)*.1}
        records.append(r);text[i]=rng.normal(size=(4,512)).astype(np.float32)*.1
        entry={'relevant_windows':[[2.,6.]],'relevant_clip_ids':[1,2],'saliency_scores':[[3,3,3]]*2}
        targets[i]=training_target(r['times'],r['duration_sec'],entry)
    return records,text,targets


def test_masked_heads_make_variable_length_batch_match_single_including_tail():
    torch.set_num_threads(1);torch.manual_seed(6);model=MaskedLocalizer(SPEC).eval();records,text,_=examples()
    with torch.inference_mode():
        batch=model(**collate_inputs(records,text))
        for i,r in enumerate(records):
            single=model(**collate_inputs([r],text))
            for k in ('pred_logits','pred_spans','saliency_scores'):
                n=len(r['times']);assert torch.allclose(single[k][0],batch[k][i,:n],atol=2e-6,rtol=2e-6)
    assert model.core.span_embed.mask is None


def test_unpadded_outputs_and_weights_match_preserved_network():
    torch.manual_seed(6);native=make_network(SPEC).eval();torch.manual_seed(6);fixed=MaskedLocalizer(SPEC).eval()
    assert set(native.state_dict())==set(fixed.core.state_dict())
    for k,v in native.state_dict().items():assert torch.equal(v,fixed.core.state_dict()[k])
    records,text,_=examples();x=collate_inputs(records[:1],text)
    with torch.inference_mode():
        a=native(**x);b=fixed(**x)
        for k in a:assert torch.equal(a[k],b[k])


def test_each_event_weight_equal_despite_twentyfold_positive_count():
    gt=torch.zeros(1,42,2);gt[0,:2]=torch.tensor([0.,.02]);gt[0,2:]=torch.tensor([.1,.5])
    target={'timestamp_mask':torch.ones(1,42),'timestamp_window':torch.ones(1,42,dtype=torch.bool),'span_labels_nn':gt}
    w=event_weights(target)
    assert torch.isclose(w[0,:2].sum(),torch.tensor(.5)) and torch.isclose(w[0,2:].sum(),torch.tensor(.5))


def test_relative_boundary_error_invariant_to_video_duration_units():
    c=criterion('event_relative');losses=[]
    for scale in (1.,.1):
        gt=torch.tensor([[[.2,.3],[.2,.3]]])*scale;anchor=torch.tensor([[[.22,.22],[.25,.25]]])*scale
        target={'timestamp':anchor,'span_labels_nn':gt,'timestamp_mask':torch.ones(1,2),'timestamp_window':torch.ones(1,2,dtype=torch.bool)}
        outputs={'pred_spans':(gt-anchor)+torch.tensor([[[-.01,.01],[-.01,.01]]])*scale}
        losses.append(c.loss_spans(outputs,target,None))
    for k in losses[0]:assert torch.allclose(losses[0][k],losses[1][k],atol=1e-6)


@pytest.mark.parametrize('mode',['mask_only','event_relative'])
def test_five_losses_finite_and_all_trainable_parameters_receive_gradients(mode):
    records,text,targets=examples();model=MaskedLocalizer(SPEC)
    x=collate_inputs(records,text);y=collate_targets(records,targets,np.random.default_rng(2))
    loss,parts=total_loss(criterion(mode),model(**x),y);loss.backward()
    assert len(parts)==5 and all(torch.isfinite(v) for v in parts.values())
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)


def test_event_weight_ignores_padding_and_rejects_empty_query():
    _,_,targets=examples();r,t,_=examples();y=collate_targets(r,targets,np.random.default_rng(1));w=event_weights(y)
    assert not w[0,12:].any() and torch.allclose(w.sum(-1),torch.ones(2))
    y['timestamp_window'][0]=False
    with pytest.raises(ValueError,match='without positive'):event_weights(y)
