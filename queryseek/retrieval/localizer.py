"""Experimental point-based localization heads built from unmodified torch.nn.

This is a small learning-head study, not a reproduction of QD-DETR/UniVTG.
Torch stays optional: production retrieval does not import this module.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from queryseek.schemas import TimeWindow
from queryseek.retrieval.fusion import nms_dedupe


def validate_timeline(times, duration):
    t = np.asarray(times, dtype=np.float64)
    if (t.ndim != 1 or not len(t) or not np.isfinite(t).all()
            or not np.isfinite(duration) or duration <= 0
            or np.any(t < 0) or np.any(t >= duration) or np.any(np.diff(t) <= 0)):
        raise ValueError("Expected strictly increasing source PTS in [0,duration)")
    return t


def supervision(times, duration, windows):
    """Window membership is benchmark relevance, not a semantic absence label."""
    t = validate_timeline(times, duration)
    ws = sorted((TimeWindow(start_sec=float(a), end_sec=float(b)) for a, b in windows),
                key=lambda w: (w.duration_sec, w.start_sec, w.end_sec))
    if not ws or any(w.end_sec > duration for w in ws):
        raise ValueError("Ground truth must be nonempty and within real duration")
    relevance = np.zeros(len(t), np.float32)
    offsets = np.zeros((len(t), 2), np.float32)
    for i, point in enumerate(t):
        match = next((w for w in ws if w.start_sec <= point < w.end_sec), None)
        if match:
            relevance[i] = 1
            offsets[i] = [(point-match.start_sec)/duration, (match.end_sec-point)/duration]
    if not relevance.any():
        raise ValueError("No positive sampled point: retain as untrainable, do not fabricate labels")
    return torch.from_numpy(relevance), torch.from_numpy(offsets)


def condition_features(images, query, times, duration, drop_query=False):
    t = validate_timeline(times, duration)
    v = torch.as_tensor(images, dtype=torch.float32)
    q = torch.as_tensor(query, dtype=torch.float32)
    if (v.ndim != 2 or v.shape[0] != len(t) or q.shape != (v.shape[1],)
            or not torch.isfinite(v).all() or not torch.isfinite(q).all()):
        raise ValueError("Invalid frozen features")
    v = F.normalize(v, dim=-1)
    q = F.normalize(q, dim=-1)
    if drop_query:
        q = torch.zeros_like(q)
    q = q.expand_as(v)
    dot = (v*q).sum(-1, keepdim=True)
    pts = torch.tensor(t/duration, dtype=torch.float32)[:, None]
    gaps = torch.tensor(np.diff(t, prepend=t[0]), dtype=torch.float32)[:, None]
    return torch.cat([v, q, v*q, dot, pts, gaps], dim=-1)


class LocalizationHead(nn.Module):
    def __init__(self, embedding_dim=512, hidden=128, architecture="frame_mlp"):
        super().__init__()
        if architecture not in ("frame_mlp", "temporal_conv") or min(embedding_dim, hidden) <= 0:
            raise ValueError("Unsupported localization architecture/dimension")
        self.architecture = architecture
        self.project = nn.Sequential(nn.Linear(embedding_dim*3+3, hidden), nn.GELU())
        if architecture == "frame_mlp":
            self.context = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                         nn.Linear(hidden, hidden), nn.GELU())
        else:
            self.context = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
                                         nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.predict = nn.Linear(hidden, 3)
        # Small initial distances prevent a full-video default boundary prior.
        with torch.no_grad():
            self.predict.bias[1:].fill_(-3.)

    def forward(self, conditioned):
        h = self.project(conditioned)
        if self.architecture == "temporal_conv":
            h = self.context(h.T.unsqueeze(0)).squeeze(0).T
        else:
            h = self.context(h)
        output = self.predict(h)
        return output[:, 0], F.softplus(output[:, 1:])


def localization_loss(logits, distances, relevance, target_offsets, weights):
    classification = F.binary_cross_entropy_with_logits(logits, relevance)
    positive = relevance.bool()
    if not positive.any():
        raise ValueError("No positive point for boundary supervision")
    d, target = distances[positive], target_offsets[positive]
    offset = F.smooth_l1_loss(d, target)
    intersection = torch.minimum(d[:, 0], target[:, 0])+torch.minimum(d[:, 1], target[:, 1])
    union = d.sum(-1)+target.sum(-1)-intersection
    iou = (1-intersection/union.clamp_min(1e-8)).mean()
    total = weights['relevance']*classification+weights['offset']*offset+weights['iou']*iou
    return total, {'relevance': float(classification.detach()), 'offset': float(offset.detach()),
                   'iou': float(iou.detach())}


def decode_spans(times, duration, logits, distances, top_k=5, nms_iou=.7, min_span_sec=.25):
    t = validate_timeline(times, duration)
    scores = torch.sigmoid(torch.as_tensor(logits)).detach().cpu().numpy()
    d = torch.as_tensor(distances).detach().cpu().numpy()
    if (scores.shape != (len(t),) or d.shape != (len(t), 2)
            or not np.isfinite(scores).all() or not np.isfinite(d).all() or np.any(d < 0)
            or top_k < 0 or not 0 <= nms_iou <= 1 or min_span_sec <= 0):
        raise ValueError("Invalid decoded prediction")
    windows = []
    for point, score, pair in zip(t, scores, d):
        start, end = max(0., float(point-pair[0]*duration)), min(float(duration), float(point+pair[1]*duration))
        if end-start >= min_span_sec:
            windows.append(TimeWindow(start_sec=start, end_sec=end, score=float(score)))
    return nms_dedupe(windows, nms_iou)[:top_k]
