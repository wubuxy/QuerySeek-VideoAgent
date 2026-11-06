"""Research-only UniVTG network adapter, not a paper reproduction.

Official network; pooled CLIP query, CLIP-only video + source time, and the
existing point loss/decoder differ from the paper's training/pretraining.
"""
import numpy as np
import torch
from torch import nn

from queryseek.vendor.univtg.univtg import Model
from queryseek.vendor.univtg.transformer_encoder_droppath import Transformer
from queryseek.vendor.univtg.position_encoding import PositionEmbeddingSine, TrainablePositionalEncoding
from queryseek.retrieval.localizer import condition_features


class UniVTGHead(nn.Module):
    def __init__(self, embedding_dim=512, hidden=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        transformer = Transformer(d_model=hidden, nhead=4, num_encoder_layers=2,
                                  dim_feedforward=hidden*4, dropout=.1, droppath=.1)
        self.network = Model(transformer, PositionEmbeddingSine(hidden, normalize=True),
                             TrainablePositionalEncoding(1, hidden), embedding_dim,
                             embedding_dim+2, .1, n_input_proj=2)
        # These modules are outside the common point-loss path.
        for module in (self.network.weightedpool, self.network.txt_position_embed):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, conditioned):
        d = self.embedding_dim
        video = torch.cat([conditioned[:, :d], conditioned[:, -2:]], -1).unsqueeze(0)
        query = conditioned[:1, d:2*d].unsqueeze(0)
        result = self.network(query, torch.ones((1, 1), device=video.device), video,
                              torch.ones(video.shape[:2], device=video.device))
        probabilities = result['pred_logits'][0, :, 0].clamp(1e-6, 1-1e-6)
        return torch.logit(probabilities), result['pred_spans'][0].abs()


def study_input(record, control='full'):
    if control not in ('full', 'no_query', 'position_only'):
        raise ValueError('Unknown training control')
    images = record['images'] if control != 'position_only' else np.zeros_like(record['images'])
    return condition_features(images, record['query_embedding'], record['times'],
                              record['duration_sec'], drop_query=control != 'full')
