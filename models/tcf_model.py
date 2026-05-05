#!/usr/bin/env python3
"""Target-conditioned fusion model for RepB inputs.

TCF-Net treats temporal dynamics, static/statistical descriptors, and their
interaction as separate source tokens. Two learned target queries attend to
these source tokens and produce target-specific predictions for y1 and y2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tabular_branch import TabularBranch


class TargetConditionedFusionModel(nn.Module):
    """Target-conditioned multi-source fusion for frequency response prediction.

    Input:
        x_temporal: (B, T, N_gen, C)
        x_static:   (B, n_static)
    Output:
        (B, 2) for [y1, y2]
    """

    def __init__(
        self,
        n_timesteps: int = 11,
        n_generators: int = 10,
        n_features: int = 5,
        n_static: int = 7,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        tab_hidden: int = 256,
        head_hidden: int = 128,
        output_dim: int = 2,
    ):
        super().__init__()
        if output_dim != 2:
            raise ValueError("TCF-Net is defined for the two target setup.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_timesteps = n_timesteps
        self.input_dim = n_generators * n_features
        self.d_model = d_model

        # Dynamic source: short post-disturbance sequence encoder.
        self.temporal_proj = nn.Linear(self.input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_timesteps, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.temporal_pool = nn.Linear(d_model, 1)
        self.temporal_drop = nn.Dropout(dropout)

        # Static/statistical source.
        self.static_encoder = TabularBranch(
            n_input=n_static,
            d_hidden=tab_hidden,
            d_output=d_model,
            dropout=dropout,
        )

        # Cross-source interaction token.
        self.interaction_encoder = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # Source identity and target-conditioned attention.
        self.source_embed = nn.Parameter(torch.zeros(1, 3, d_model))
        self.target_queries = nn.Parameter(torch.zeros(1, 2, d_model))
        nn.init.trunc_normal_(self.source_embed, std=0.02)
        nn.init.trunc_normal_(self.target_queries, std=0.02)

        self.source_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.target_norm = nn.LayerNorm(d_model)
        self.target_drop = nn.Dropout(dropout)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            for _ in range(2)
        ])

        self.last_source_attention = None
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue
            if 'heads' in name and name.endswith('weight'):
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'weight' in name:
                nn.init.xavier_uniform_(p)

    def encode_temporal(self, x_temporal: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x_temporal.shape
        x = x_temporal.reshape(B, T, N * C)
        x = self.temporal_proj(x) + self.pos_embed[:, :T, :]
        x = self.temporal_drop(x)
        x = self.temporal_encoder(x)
        weights = F.softmax(self.temporal_pool(x).squeeze(-1), dim=1)
        return torch.einsum('bt,btd->bd', weights, x)

    def forward(self, x_temporal: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        h_temporal = self.encode_temporal(x_temporal)
        h_static = self.static_encoder(x_static)
        h_interaction = self.interaction_encoder(
            torch.cat([h_temporal, h_static], dim=-1)
        )

        sources = torch.stack([h_temporal, h_static, h_interaction], dim=1)
        sources = self.source_norm(sources + self.source_embed)

        B = x_temporal.shape[0]
        queries = self.target_queries.expand(B, -1, -1)
        target_ctx, attn = self.cross_attn(
            queries, sources, sources,
            need_weights=True,
            average_attn_weights=True,
        )
        self.last_source_attention = attn.detach()

        target_ctx = self.target_norm(target_ctx + queries)
        target_ctx = self.target_drop(target_ctx)
        y1 = self.heads[0](target_ctx[:, 0]).squeeze(-1)
        y2 = self.heads[1](target_ctx[:, 1]).squeeze(-1)
        return torch.stack([y1, y2], dim=-1)


def build_tcf_from_trial(trial, n_timesteps=11, n_generators=10,
                         n_features=5, n_static=7):
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
    n_heads = trial.suggest_categorical('n_heads', valid_heads)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    d_ff = trial.suggest_categorical('d_ff', [128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    tab_hidden = trial.suggest_categorical('tab_hidden', [128, 256, 512])
    head_hidden = trial.suggest_categorical('head_hidden', [64, 128, 256])

    kwargs = dict(
        n_timesteps=n_timesteps,
        n_generators=n_generators,
        n_features=n_features,
        n_static=n_static,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        tab_hidden=tab_hidden,
        head_hidden=head_hidden,
    )
    return TargetConditionedFusionModel(**kwargs), kwargs
