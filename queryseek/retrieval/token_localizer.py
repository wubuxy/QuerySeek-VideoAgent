"""Token-conditioned dense localization using the preserved official UniVTG.

Source-PTS 1fps CLIP-only research adaptation, not a paper reproduction or a
released checkpoint-compatible input. Official network and five losses are
reused; target/collation/time conversion are explicit glue.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from queryseek.vendor.univtg.univtg import Model, SetCriterion
from queryseek.vendor.univtg.transformer_encoder_droppath import Transformer
from queryseek.vendor.univtg.position_encoding import PositionEmbeddingSine, TrainablePositionalEncoding
from queryseek.vendor.univtg.span_utils import generalized_temporal_iou
from queryseek.retrieval.curve_spans import select_curve_windows
from queryseek.schemas import TimeWindow


def make_network(spec,embedding_dim=512):
    transformer=Transformer(d_model=spec['hidden'],nhead=spec['heads'],num_encoder_layers=spec['layers'],
        dim_feedforward=spec['ffn'],dropout=.1,droppath=.1)
    model=Model(transformer,PositionEmbeddingSine(spec['hidden'],normalize=True),
                TrainablePositionalEncoding(32,spec['hidden']),embedding_dim,embedding_dim+2,.5,n_input_proj=2,
                max_v_l=256,use_txt_pos=False)
    # Upstream disables text position embeddings in this configuration.
    for parameter in model.txt_position_embed.parameters():parameter.requires_grad_(False)
    return model


class BoundedCriterion(SetCriterion):
    """Same official loss; compute GIoU diagonal in blocks to bound CPU memory."""
    def loss_spans(self,outputs,targets,indices):
        predicted=targets['timestamp']+outputs['pred_spans'];gt=targets['span_labels_nn']
        positive=targets['timestamp_window'].bool();mask=positive.unsqueeze(-1).expand_as(predicted)
        if not positive.any():raise ValueError('Batch contains no foreground points')
        loss_b=(F.smooth_l1_loss(predicted,gt,reduction='none')*mask).sum()/positive.sum()
        p,g=predicted[positive],gt[positive]
        diagonal=[torch.diag(generalized_temporal_iou(p[start:start+64],g[start:start+64]))
                  for start in range(0,len(p),64)]
        return {'loss_b':loss_b,'loss_g':(1-torch.cat(diagonal)).mean()}


def make_criterion():
    weights={'loss_b':10.,'loss_g':1.,'loss_f':10.,'loss_s_intra':.05,'loss_s_inter':.01}
    return BoundedCriterion(None,weights,.1,['spans','labels','saliency'],.07,'l1',256)


def training_target(times,duration_sec,entry):
    """Half-open source-time labels; human 2-second saliency cells map to PTS."""
    times=np.asarray(times,dtype=np.float64)
    if not len(times) or not np.isfinite(times).all() or not np.isfinite(duration_sec) or duration_sec<=0:
        raise ValueError('Invalid source timeline')
    if times[0]<0 or times[-1]>=duration_sec or np.any(np.diff(times)<=0):raise ValueError('Invalid source seconds')
    windows=entry['relevant_windows']
    if not windows or any(not np.isfinite([a,b]).all() or not 0<=a<b<=duration_sec for a,b in windows):
        raise ValueError('Training GT exceeds actual duration')
    nearest=np.zeros((len(times),2),np.float32);positive=np.zeros(len(times),bool)
    # QVHL windows are normally disjoint. Deterministically prefer the shortest
    # containing interval if annotations overlap, without changing the labels.
    for a,b in sorted(windows,key=lambda w:(w[1]-w[0],w[0],w[1])):
        match=(times>=a)&(times<b)&~positive
        nearest[match]=np.array([a,b],np.float32)/duration_sec;positive|=match
    if not positive.any():raise ValueError('No observed positive point; do not fabricate foreground')
    ids=entry['relevant_clip_ids'];scores=np.asarray(entry['saliency_scores'],dtype=np.float64)
    if len(ids)!=len(scores) or scores.ndim!=2 or scores.shape[1]!=3 or not np.isfinite(scores).all():
        raise ValueError('Invalid human saliency labels')
    if len(set(ids))!=len(ids) or any(not isinstance(i,int) or i<0 for i in ids):raise ValueError('Invalid saliency clip IDs')
    lookup={i:float(row.mean()) for i,row in zip(ids,scores)}
    saliency=np.array([lookup.get(int(np.floor(t/2.)),0.) for t in times],np.float32)
    if saliency.sum()<=0:raise ValueError('No human saliency support')
    return {'timestamp':torch.from_numpy(np.repeat((times/duration_sec)[:,None],2,axis=1).astype(np.float32)),
            'timestamp_window':torch.from_numpy(positive),'span_labels_nn':torch.from_numpy(nearest),
            'saliency_scores':torch.from_numpy(saliency)}


def collate_inputs(records,text_features):
    """GT-free padded video/token inputs and masks; source time is dimensionless."""
    if not records:raise ValueError('Empty localization batch')
    nv=max(len(r['times']) for r in records);nt=max(len(text_features[r['qid']]) for r in records)
    d=records[0]['images'].shape[1];batch=len(records)
    video=torch.zeros(batch,nv,d+2);text=torch.zeros(batch,nt,d)
    vm=torch.zeros(batch,nv);tm=torch.zeros(batch,nt)
    for i,r in enumerate(records):
        times=np.asarray(r['times'],dtype=np.float64);images=np.asarray(r['images']);tokens=np.asarray(text_features[r['qid']])
        if (images.shape!=(len(times),d) or tokens.ndim!=2 or tokens.shape[1]!=d or not 2<=len(tokens)<=32
                or not np.isfinite(images).all() or not np.isfinite(tokens).all() or not np.isfinite(times).all()
                or times[0]<0 or times[-1]>=r['duration_sec'] or np.any(np.diff(times)<=0)):
            raise ValueError('Invalid token/video contract')
        # Explicit source-time interval feature, different from upstream logical TEF.
        end=np.r_[times[1:],r['duration_sec']]
        tef=np.stack([times/r['duration_sec'],end/r['duration_sec']],axis=-1).astype(np.float32)
        n=len(times);q=len(tokens)
        video[i,:n]=torch.from_numpy(np.concatenate([images.astype(np.float32),tef],axis=-1))
        text[i,:q]=torch.from_numpy(tokens.astype(np.float32));vm[i,:n]=1;tm[i,:q]=1
    return {'src_vid':video,'src_txt':text,'src_vid_mask':vm,'src_txt_mask':tm}


def collate_targets(records,targets,rng,selected_positions=None):
    nv=max(len(r['times']) for r in records);batch=len(records)
    result={'timestamp':torch.zeros(batch,nv,2),'timestamp_mask':torch.zeros(batch,nv),
            'timestamp_window':torch.zeros(batch,nv,dtype=torch.bool),'span_labels_nn':torch.zeros(batch,nv,2),
            'saliency_scores':torch.zeros(batch,nv),'saliency_pos_labels':torch.zeros(batch,1,dtype=torch.long)}
    for i,r in enumerate(records):
        target=targets[r['qid']];n=len(r['times']);result['timestamp_mask'][i,:n]=1
        for key in ('timestamp','timestamp_window','span_labels_nn','saliency_scores'):result[key][i,:n]=target[key]
        # Original easy-positive policy, mapped to source observations. Batch
        # sampler prevents same-source examples becoming inter-video negatives.
        positives=target['timestamp_window'].nonzero().flatten().numpy()
        chosen=int(rng.choice(positives)) if selected_positions is None else int(selected_positions[i])
        if chosen not in positives:raise ValueError('Planned saliency positive is not foreground')
        result['saliency_pos_labels'][i,0]=chosen
    return result


def total_loss(criterion,outputs,targets):
    parts=criterion(outputs,targets)
    if set(parts)!=set(criterion.weight_dict):raise ValueError('Missing official loss component')
    if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in parts.values()):
        raise ValueError('Nonfinite or inactive loss component')
    return sum(criterion.weight_dict[k]*v for k,v in parts.items()),parts


def decode_dense(record,probabilities,offsets,top_k=5,nms_iou=.7):
    """Signed upstream offsets, decoded at source PTS; no GT input."""
    times=np.asarray(record['times'],dtype=np.float64);p=np.asarray(probabilities).reshape(-1);o=np.asarray(offsets)
    duration=float(record['duration_sec'])
    if p.shape!=times.shape or o.shape!=(len(times),2) or not np.isfinite(p).all() or not np.isfinite(o).all():
        raise ValueError('Invalid dense localization outputs')
    if np.any(o[:,0]>0) or np.any(o[:,1]<0):raise ValueError('Expected signed left/right offsets')
    pool=[]
    for t,score,(left,right) in zip(times,p,o):
        a=max(0.,float(t+left*duration));b=min(duration,float(t+right*duration))
        if a<b:pool.append(TimeWindow(start_sec=a,end_sec=b,score=float(score)))
    selected=select_curve_windows(pool,top_k,nms_iou)
    convert=lambda ws:[(w.start_sec,w.end_sec,w.score) for w in ws]
    return convert(selected),convert(pool)
