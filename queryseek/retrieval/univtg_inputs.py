"""Research input-contract helpers; no locator, weights or inference GT.

Extract hidden states by observing the existing CLIP ln_final module. Do not
reimplement CLIP transformers or substitute pooled/projected text for tokens.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def text_hidden_states(model, tokens):
    """Unprojected ln_final states and projected pool from one native forward."""
    if tokens.ndim != 2 or tokens.dtype != torch.long:
        raise ValueError('Expected a batch of integer CLIP token IDs')
    captured = []
    handle = model.ln_final.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
    try:
        with torch.inference_mode():
            result = model.encode_text(tokens)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape[:2] != tokens.shape:
        raise ValueError('CLIP ln_final contract mismatch')
    pool = result['pooler_output'] if isinstance(result,dict) else result
    if not torch.isfinite(captured[0]).all() or not torch.isfinite(pool).all():
        raise ValueError('Nonfinite CLIP text features')
    return captured[0].float(), pool.float()


def valid_text_features(hidden, tokens, eos_id=49407, length_policy='eot'):
    """Return variable-length, row-normalized tokens; preserve SOT/EOT.

    upstream_nonzero reproduces UniVTG's published feature extractor exactly.
    eot is explicit because token ID zero can also be a valid vocabulary item.
    These policies must not be silently interchanged in a frozen experiment.
    """
    if hidden.ndim != 3 or hidden.shape[:2] != tokens.shape:
        raise ValueError('Text/token shape mismatch')
    if length_policy not in ('eot','upstream_nonzero'):
        raise ValueError('Unknown text-length policy')
    result = []
    for h,t in zip(hidden,tokens):
        locations = (t==eos_id).nonzero().flatten()
        if not len(locations): raise ValueError('Missing EOT token')
        n = int(locations[0])+1 if length_policy=='eot' else int((t!=0).sum())
        if n < 2 or n > len(t): raise ValueError('Invalid valid-token length')
        result.append(F.normalize(h[:n],dim=-1))
    return result


def upstream_temporal_inputs(n_clips, clip_length_sec=2.):
    """Exact published TEF/anchor convention, retaining its unusual anchor rule.

    TEF is dimensionless. anchors are internal normalized coordinates; external
    windows are converted to float seconds separately. No source PTS claim.
    """
    if isinstance(n_clips,bool) or not isinstance(n_clips,int) or n_clips<=0:
        raise ValueError('Expected a positive clip count')
    if not np.isfinite(clip_length_sec) or clip_length_sec<=0:
        raise ValueError('Expected positive clip length in seconds')
    index = torch.arange(n_clips,dtype=torch.float32)
    tef = torch.stack([index/n_clips,(index+1)/n_clips],dim=-1)
    anchors = ((index+clip_length_sec/2)/n_clips)[:,None].repeat(1,2)
    return tef, anchors


def numeric_difference(first, second):
    a = np.asarray(first,dtype=np.float64); b = np.asarray(second,dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Comparison requires equal, finite, nonempty arrays')
    return {'shape':list(a.shape),'max_abs':float(np.abs(a-b).max()),
            'mean_abs':float(np.abs(a-b).mean()),'rmse':float(np.sqrt(np.mean((a-b)**2)))}
