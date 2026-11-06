"""Mask-safe UniVTG heads and event-balanced relative boundary supervision.

The preserved network, convolutions, BCE, SmoothL1, GIoU, and saliency losses
are reused. Changes are mask propagation and supervised loss reductions.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from queryseek.retrieval.token_localizer import make_network as original_network,BoundedCriterion
from queryseek.vendor.univtg.span_utils import generalized_temporal_iou


class MaskedConvHead(nn.Module):
    def __init__(self,original):
        super().__init__();self.layers=original.layers;self.mask=None

    def forward(self,x):
        if self.mask is None:raise ValueError('Video mask required for convolution head')
        mask=self.mask[:,None,:].to(x.dtype);x=x.transpose(1,2)*mask
        for i,layer in enumerate(self.layers):
            x=layer(x)
            if i+1<len(self.layers):x=F.relu(x)
            x=x*mask
        return x.transpose(1,2)


class MaskedLocalizer(nn.Module):
    def __init__(self,spec):
        super().__init__();self.core=original_network(spec)
        self.core.span_embed=MaskedConvHead(self.core.span_embed)
        self.core.class_embed=MaskedConvHead(self.core.class_embed)

    def forward(self,src_vid_mask,**inputs):
        self.core.span_embed.mask=src_vid_mask;self.core.class_embed.mask=src_vid_mask
        try:return self.core(src_vid_mask=src_vid_mask,**inputs)
        finally:self.core.span_embed.mask=None;self.core.class_embed.mask=None


def event_weights(targets):
    """Each annotated assigned event has equal weight within its query."""
    positive=targets['timestamp_window'].bool()&targets['timestamp_mask'].bool()
    weights=torch.zeros_like(targets['timestamp_mask'])
    for i in range(len(positive)):
        selected=targets['span_labels_nn'][i,positive[i]]
        if not len(selected):raise ValueError('Query without positive event')
        _,inverse,counts=torch.unique(selected,dim=0,return_inverse=True,return_counts=True)
        weights[i,positive[i]]=1./(len(counts)*counts[inverse].to(weights.dtype))
    return weights


class EventRelativeCriterion(BoundedCriterion):
    """Original five losses; query/event balanced BCE and relative SmoothL1."""
    def loss_spans(self,outputs,targets,indices):
        predicted=targets['timestamp']+outputs['pred_spans'];gt=targets['span_labels_nn']
        weights=event_weights(targets);positive=weights>0
        length=(gt[...,1]-gt[...,0])[positive]
        if not torch.isfinite(length).all() or (length<=0).any():raise ValueError('Invalid event duration')
        error=(predicted[positive]-gt[positive])/length[:,None]
        per_point=F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none').sum(-1)
        p,g=predicted[positive],gt[positive]
        giou=torch.cat([torch.diag(generalized_temporal_iou(p[a:a+64],g[a:a+64])) for a in range(0,len(p),64)])
        w=weights[positive];batch=len(weights)
        return {'loss_b':(per_point*w).sum()/batch,'loss_g':((1-giou)*w).sum()/batch}

    def loss_labels(self,outputs,targets,indices,log=True):
        probabilities=outputs['pred_logits'].squeeze(-1);positive=targets['timestamp_window'].bool()
        valid=targets['timestamp_mask'].bool();weights=event_weights(targets)
        bce=F.binary_cross_entropy(probabilities,positive.to(probabilities.dtype),reduction='none')
        pos=(bce*weights).sum(-1);negative=valid&~positive
        neg=(bce*negative).sum(-1)/negative.sum(-1).clamp_min(1)
        background=float(self.empty_weight[1])
        return {'loss_f':((pos+background*neg)/(1+background)).mean()}


def criterion(mode):
    if mode not in ('mask_only','event_relative'):raise ValueError('Unknown frozen objective')
    cls=BoundedCriterion if mode=='mask_only' else EventRelativeCriterion
    weights={'loss_b':10.,'loss_g':1.,'loss_f':10.,'loss_s_intra':.05,'loss_s_inter':.01}
    return cls(None,weights,.1,['spans','labels','saliency'],.07,'l1',256)
