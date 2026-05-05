#!/usr/bin/env python3
"""Mamba (Selective State Space Model) for sequence regression (RepB input).

Key design choices:
- Pre-LayerNorm residual blocks (matches Mamba paper convention).
- Weighted pooling over all timesteps instead of last-only (better for
  short sequences where information is spread across all 11 timesteps).
- Dual-branch architecture: temporal (Mamba) + tabular (MLP) branches
  fused before the output head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from models.tabular_branch import TabularBranch


class MambaFreqModel(nn.Module):
    """Mamba-based model for multi-task frequency prediction (dual-branch).

    Temporal branch: Mamba encodes (B, T, N*C) -> h_temporal (d_model)
    Tabular branch: MLP encodes x_static (B, n_static) -> h_tabular (tab_output)
    Fusion: concat + MLP head -> (B, 2)

    Input:
        x_temporal: (B, T, N_gen, C) -> flatten to (B, T, N*C)
        x_static:   (B, n_static)
    Output:
        (B, 2)
    """

    def __init__(self, n_timesteps: int = 11, n_generators: int = 10,
                 n_features: int = 7, d_model: int = 128,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 n_layers: int = 4, n_static: int = 7,
                 output_dim: int = 2, dropout: float = 0.1,
                 tab_hidden: int = 256, tab_output: int = 128):
        super().__init__()

        # ── Temporal branch: Mamba ──
        input_dim = n_generators * n_features

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_timesteps, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                Mamba(d_model=d_model, d_state=d_state,
                      d_conv=d_conv, expand=expand)
            )
            self.norms.append(nn.LayerNorm(d_model))

        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable temporal attention pooling (better than last-only for T=11)
        self.pool_attn = nn.Linear(d_model, 1)
        d_temporal = d_model

        # ── Tabular branch: processes x_static ──
        self.tabular_branch = TabularBranch(
            n_input=n_static, d_hidden=tab_hidden,
            d_output=tab_output, dropout=dropout,
        )

        # ── Fusion head with skip connection ──
        fusion_dim = d_temporal + tab_output
        self.fusion_fc1 = nn.Linear(fusion_dim, 128)
        self.fusion_fc2 = nn.Linear(128, 128)
        self.fusion_ln = nn.LayerNorm(128)
        self.fusion_drop = nn.Dropout(dropout)
        self.fusion_out = nn.Linear(128, output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'fusion_out.weight' in name:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'fusion_fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'input_proj.weight' in name:
                nn.init.xavier_uniform_(p)

    def forward(self, x_temporal, x_static):
        B, T, N, C = x_temporal.shape

        # ── Temporal branch ──
        x = x_temporal.reshape(B, T, N * C)  # (B, T, N*C)
        x = self.input_dropout(self.input_proj(x) + self.pos_embed[:, :T, :])

        for mamba_layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = mamba_layer(x)
            x = self.dropout(x) + residual

        x = self.final_norm(x)  # (B, T, d_model)

        # Attention-weighted temporal pooling
        attn_weights = F.softmax(self.pool_attn(x).squeeze(-1), dim=1)  # (B, T)
        h_temporal = torch.einsum('bt,btd->bd', attn_weights, x)  # (B, d_model)

        # ── Tabular branch ──
        h_tabular = self.tabular_branch(x_static)  # (B, tab_output)

        # ── Fusion head with skip connection ──
        combined = torch.cat([h_temporal, h_tabular], dim=-1)
        x = F.gelu(self.fusion_fc1(combined))
        x = self.fusion_drop(x)
        residual = x
        x = F.gelu(self.fusion_fc2(x))
        x = self.fusion_ln(x + residual)
        x = self.fusion_drop(x)
        return self.fusion_out(x)


def build_mamba_from_trial(trial, n_timesteps=11, n_generators=10,
                           n_features=7, n_static=7):
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    d_state = trial.suggest_categorical('d_state', [8, 16, 32])
    d_conv = trial.suggest_int('d_conv', 2, 4)
    expand = trial.suggest_int('expand', 1, 3)
    n_layers = trial.suggest_int('n_layers', 2, 6)
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    # Mamba benefits from large tabular branch (proven in v2)
    tab_h = trial.suggest_categorical('tab_hidden', [128, 256, 512])
    tab_o = trial.suggest_categorical('tab_output', [64, 128, 256])

    kwargs = dict(n_timesteps=n_timesteps, n_generators=n_generators,
                  n_features=n_features, d_model=d_model, d_state=d_state,
                  d_conv=d_conv, expand=expand, n_layers=n_layers,
                  n_static=n_static, dropout=dropout,
                  tab_hidden=tab_h, tab_output=tab_o)
    return MambaFreqModel(**kwargs), kwargs
