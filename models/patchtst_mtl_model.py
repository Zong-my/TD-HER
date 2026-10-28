#!/usr/bin/env python3
"""Exact PatchTST architecture exposed for conflict-aware MTL training.

The model retains ``PatchTSTFreqModel`` verbatim, including its two-output
linear head and initialization order.  Only the training script changes how
the two task losses are combined.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.patchtst_model import PatchTSTFreqModel


class PatchTSTConflictMTL(PatchTSTFreqModel):
    """PatchTST with a shared 128-D representation and two-output layer."""

    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs["output_dim"] = 2
        super().__init__(**kwargs)

    def encode(self, x_temporal: torch.Tensor,
               x_static: torch.Tensor) -> torch.Tensor:
        """Return the final shared representation before task heads."""
        batch_size, n_steps, n_nodes, n_channels = x_temporal.shape
        x = x_temporal.reshape(batch_size, n_steps, n_nodes * n_channels)
        x = x.permute(0, 2, 1)

        if n_steps >= self.patch_len:
            patches = x.unfold(
                dimension=2, size=self.patch_len, step=self.stride
            )
        else:
            x_padded = F.pad(x, (0, self.patch_len - n_steps))
            patches = x_padded.unsqueeze(2)

        n_vars = n_nodes * n_channels
        n_patches = patches.shape[2]
        patches = patches.reshape(
            batch_size * n_vars, n_patches, self.patch_len
        )
        embedded = self.embed_dropout(
            self.patch_embed(patches) + self.pos_embed[:, :n_patches, :]
        )
        with sdpa_kernel(SDPBackend.MATH):
            encoded = self.encoder(embedded)

        pooled = encoded.mean(dim=1)
        pooled = pooled.reshape(batch_size, n_vars, -1).reshape(batch_size, -1)
        h_temporal = self.channel_agg(pooled)
        h_tabular = self.tabular_branch(x_static)

        combined = torch.cat([h_temporal, h_tabular], dim=-1)
        hidden = F.gelu(self.fusion_fc1(combined))
        hidden = self.fusion_drop(hidden)
        residual = hidden
        hidden = F.gelu(self.fusion_fc2(hidden))
        hidden = self.fusion_ln(hidden + residual)
        return self.fusion_drop(hidden)

    def forward(self, x_temporal: torch.Tensor,
                x_static: torch.Tensor) -> torch.Tensor:
        shared = self.encode(x_temporal, x_static)
        return self.fusion_out(shared)

    def shared_parameters(self):
        """Parameters affected by both task losses."""
        for name, parameter in self.named_parameters():
            if not name.startswith("fusion_out."):
                yield parameter

    def gradnorm_parameters(self):
        """Last shared layer used by the GradNorm gradient measurement."""
        yield self.fusion_fc2.weight


class PatchTSTMMoE(PatchTSTConflictMTL):
    """PatchTST encoder with a canonical multi-gate MoE regression head.

    The submitted PatchTST encoder and fused 128-D representation are kept
    unchanged. Multiple shared MLP experts transform that representation, and
    each target learns its own softmax gate and prediction tower. This isolates
    an architecture-level MTL alternative while retaining the same inputs,
    data protocol, optimizer budget, and shared encoder initialization.
    """

    def __init__(self, num_experts: int = 4, expert_hidden: int = 64,
                 tower_hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        if num_experts < 2:
            raise ValueError("MMoE requires at least two shared experts")

        shared_dim = self.fusion_out.in_features
        dropout = float(kwargs.get("dropout", 0.1))
        # Remove the original two-output linear head after its initialization,
        # so all shared PatchTST parameters retain the submitted init order.
        self.fusion_out = nn.Identity()
        self.num_experts = int(num_experts)
        self.expert_hidden = int(expert_hidden)
        self.tower_hidden = int(tower_hidden)

        self.mmoe_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, expert_hidden),
                nn.GELU(),
            )
            for _ in range(num_experts)
        ])
        self.mmoe_gates = nn.ModuleList([
            nn.Linear(shared_dim, num_experts)
            for _ in range(2)
        ])
        self.mmoe_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden, tower_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(tower_hidden, 1),
            )
            for _ in range(2)
        ])
        self._init_mmoe_head()

    def _init_mmoe_head(self) -> None:
        for module in (
            list(self.mmoe_experts)
            + list(self.mmoe_gates)
            + list(self.mmoe_towers)
        ):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x_temporal: torch.Tensor,
                x_static: torch.Tensor) -> torch.Tensor:
        shared = self.encode(x_temporal, x_static)
        expert_outputs = torch.stack(
            [expert(shared) for expert in self.mmoe_experts], dim=1
        )
        predictions = []
        for gate, tower in zip(self.mmoe_gates, self.mmoe_towers):
            weights = F.softmax(gate(shared), dim=-1)
            mixed = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)
            predictions.append(tower(mixed).squeeze(-1))
        return torch.stack(predictions, dim=-1)
