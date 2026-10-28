#!/usr/bin/env python3
"""ConvLSTM wrapper classes for the second-backbone multi-task control block (protocol second_backbone_convlstm_v2).

ConvLSTMConflictMTL subclasses the frozen models/convlstm_model.py::ConvLSTMFreqModel without changing module
registration, initialisation or the forward pass (forward is inherited, so the wrapper is numerically the frozen model
with output_dim=2). It only adds the two accessors the frozen ConflictAwareTrainer needs: shared_parameters() and
gradnorm_parameters() (last shared fusion layer weight, the same rule as PatchTSTConflictMTL).

ConvLSTMMMoE replicates the head of models/patchtst_mtl_model.py::PatchTSTMMoE on the ConvLSTM fused representation:
the encoder is built first (identical RNG consumption, bitwise-equal encoder tensors under the same seed), fusion_out is
replaced by Identity, then num_experts shared MLP experts, one softmax gate per target and one tower per target are added
with xavier-uniform weights and zero biases in the same order as the PatchTST version.
"""
from __future__ import annotations

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.convlstm_model import ConvLSTMFreqModel  # frozen  # noqa: E402


class ConvLSTMConflictMTL(ConvLSTMFreqModel):
    """Frozen ConvLSTM dual-branch model with two outputs and the trainer accessors."""

    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs["output_dim"] = 2
        super().__init__(**kwargs)

    def shared_parameters(self):
        """Parameters affected by both task losses (everything before the output layer)."""
        for name, parameter in self.named_parameters():
            if not name.startswith("fusion_out."):
                yield parameter

    def gradnorm_parameters(self):
        """Last shared layer used by the GradNorm gradient measurement (same rule as PatchTSTConflictMTL)."""
        yield self.fusion_fc2.weight


class ConvLSTMMMoE(ConvLSTMConflictMTL):
    """ConvLSTM encoder with the canonical multi-gate MoE regression head of the PatchTST control."""

    def __init__(self, num_experts: int = 4, expert_hidden: int = 64,
                 tower_hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        if num_experts < 2:
            raise ValueError("MMoE requires at least two shared experts")
        shared_dim = self.fusion_out.in_features
        dropout = float(kwargs.get("dropout", 0.1))
        # Remove the original two-output linear head after its initialisation,
        # so all shared ConvLSTM parameters retain the frozen init order.
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
        self.mmoe_gates = nn.ModuleList([nn.Linear(shared_dim, num_experts) for _ in range(2)])
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
        for module in (list(self.mmoe_experts) + list(self.mmoe_gates) + list(self.mmoe_towers)):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def encode(self, x_temporal: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        # fusion_out is Identity, so the frozen forward returns the fused shared representation.
        return ConvLSTMFreqModel.forward(self, x_temporal, x_static)

    def forward(self, x_temporal: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        shared = self.encode(x_temporal, x_static)
        expert_outputs = torch.stack([expert(shared) for expert in self.mmoe_experts], dim=1)
        predictions = []
        for gate, tower in zip(self.mmoe_gates, self.mmoe_towers):
            weights = F.softmax(gate(shared), dim=-1)
            mixed = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)
            predictions.append(tower(mixed).squeeze(-1))
        return torch.stack(predictions, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
