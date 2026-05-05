#!/usr/bin/env python3
"""PatchTST model for multivariate time series regression (RepB input).

Key improvements over vanilla implementation:
- Pre-LayerNorm transformer blocks (more stable training).
- Learnable position embeddings with proper initialization.
- Two-stage aggregation: patch-mean -> channel MLP with LayerNorm.
- Dual-branch architecture: temporal (PatchTST) + tabular (MLP) branches
  fused before the output head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# Disable all non-math SDP backends — channel-independent PatchTST uses
# B*n_vars as the batch dimension, which can exceed 65535 with large systems.
# The math backend has no such limit.
torch.backends.cuda.flash_sdp_enabled = False
torch.backends.cuda.mem_efficient_sdp_enabled = False
if hasattr(torch.backends.cuda, 'cudnn_sdp_enabled'):
    torch.backends.cuda.cudnn_sdp_enabled = False

from models.tabular_branch import TabularBranch


class PatchTSTFreqModel(nn.Module):
    """PatchTST adapted for short multivariate sequence -> scalar regression.

    Temporal branch: PatchTST encodes (B, T, N, C) -> h_temporal
    Tabular branch: MLP encodes x_static (B, n_static) -> h_tabular
    Fusion: concat + MLP head -> (B, 2)

    Input:
        x_temporal: (B, T, N_gen, C) -> flatten to (B, T, N*C)
        x_static:   (B, n_static)
    Output:
        (B, 2)
    """

    def __init__(self, n_timesteps: int = 11, n_generators: int = 10,
                 n_features: int = 7, patch_len: int = 3, stride: int = 2,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.1,
                 n_static: int = 7, output_dim: int = 2,
                 tab_hidden: int = 256, tab_output: int = 128):
        super().__init__()
        self.n_vars = n_generators * n_features
        self.n_timesteps = n_timesteps

        # Adapt patch_len for very short sequences
        self.patch_len = min(patch_len, n_timesteps)
        self.stride = min(stride, max(1, self.patch_len - 1))
        self.n_patches = max(1, (n_timesteps - self.patch_len) // self.stride + 1)

        # ── Temporal branch: PatchTST ──
        # Patch embedding: project patch of raw values to d_model
        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

        # Pre-norm Transformer encoder (norm_first=True is more stable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,  # Pre-LayerNorm for better gradient flow
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),  # final norm
        )

        # Channel aggregation: from n_vars * d_model -> compact
        agg_dim = min(256, self.n_vars * d_model)
        self.channel_agg = nn.Sequential(
            nn.LayerNorm(self.n_vars * d_model),
            nn.Linear(self.n_vars * d_model, agg_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        d_temporal = agg_dim

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
            if 'weight' in name and p.dim() >= 2:
                if 'fusion_out' in name:
                    nn.init.xavier_uniform_(p, gain=0.1)
                elif 'linear' in name.lower() or 'fc' in name.lower() or 'fusion' in name.lower():
                    nn.init.xavier_uniform_(p)

    def forward(self, x_temporal, x_static):
        B, T, N, C = x_temporal.shape

        # ── Temporal branch ──
        # Flatten spatial: (B, T, N*C) = (B, T, n_vars)
        x = x_temporal.reshape(B, T, N * C)  # (B, T, n_vars)

        # Channel-independent processing: (B, n_vars, T)
        x = x.permute(0, 2, 1)  # (B, n_vars, T)

        # Create patches: (B, n_vars, n_patches, patch_len)
        if T >= self.patch_len:
            patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        else:
            # Fallback: pad and use single patch
            x_padded = F.pad(x, (0, self.patch_len - T))
            patches = x_padded.unsqueeze(2)  # (B, n_vars, 1, patch_len)

        BV = B * self.n_vars
        P = patches.shape[2]

        # Embed: (B*n_vars, n_patches, d_model)
        patches = patches.reshape(BV, P, self.patch_len)
        embedded = self.embed_dropout(
            self.patch_embed(patches) + self.pos_embed[:, :P, :]
        )

        # Encode — force math SDP to avoid CUDA kernel batch-dimension limits
        # when B*n_vars is large (e.g. 69*9*64 = 39744 for IEEE 300-bus).
        with sdpa_kernel(SDPBackend.MATH):
            encoded = self.encoder(embedded)

        # Pool over patches: (B*n_vars, d_model)
        pooled = encoded.mean(dim=1)

        # Reshape back: (B, n_vars * d_model)
        pooled = pooled.reshape(B, self.n_vars, -1).reshape(B, -1)

        # Aggregate channels
        h_temporal = self.channel_agg(pooled)

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


def build_patchtst_from_trial(trial, n_timesteps=11, n_generators=10,
                               n_features=7, n_static=7):
    n_vars = n_generators * n_features
    if n_vars >= 512:
        # IEEE300-scale PatchTST has B*n_vars channel-independent tokens.
        # Keep the Transformer compact enough for a reproducible staged run.
        d_model = trial.suggest_categorical('d_model', [32, 64])
        n_heads = trial.suggest_categorical('n_heads', [2, 4])
        n_layers = trial.suggest_int('n_layers', 1, 3)
        d_ff = trial.suggest_categorical('d_ff', [64, 128])
        patch_options = [p for p in [3, 5] if p <= n_timesteps]
        if not patch_options:
            patch_options = [min(2, n_timesteps)]
        patch_len = trial.suggest_categorical('patch_len', patch_options)
        stride = trial.suggest_int('stride', min(2, patch_len), max(1, patch_len - 1))
    else:
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        n_layers = trial.suggest_int('n_layers', 2, 6)
        d_ff = trial.suggest_categorical('d_ff', [128, 256, 512])
        patch_len = trial.suggest_int('patch_len', 2, min(5, n_timesteps))
        stride = trial.suggest_int('stride', 1, max(1, patch_len - 1))
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    # PatchTST is channel-independent; tabular branch compensates for cross-channel blindness
    if n_vars >= 512:
        tab_h = trial.suggest_categorical('tab_hidden', [64, 128])
        tab_o = trial.suggest_categorical('tab_output', [32, 64])
    else:
        tab_h = trial.suggest_categorical('tab_hidden', [64, 128, 256])
        tab_o = trial.suggest_categorical('tab_output', [32, 64, 128])

    kwargs = dict(n_timesteps=n_timesteps, n_generators=n_generators,
                  n_features=n_features, patch_len=patch_len, stride=stride,
                  d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                  d_ff=d_ff, dropout=dropout, n_static=n_static,
                  tab_hidden=tab_h, tab_output=tab_o)
    return PatchTSTFreqModel(**kwargs), kwargs
