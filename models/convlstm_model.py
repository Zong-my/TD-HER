#!/usr/bin/env python3
"""ConvLSTM model for spatio-temporal frequency prediction (RepB input).

Key design choices vs. naive ConvLSTM:
- Layer normalization instead of BatchNorm on gates (BN on recurrent gates
  introduces batch-dependent statistics that destabilize training).
- Forget-gate bias initialized to +1 following Jozefowicz et al. (2015).
- Dropout applied to hidden state, not to gates directly.
- Dual-branch architecture: temporal (ConvLSTM) + tabular (MLP) branches
  fused before the output head.
"""

import torch
import torch.nn as nn

from models.tabular_branch import TabularBranch


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell with LayerNorm and residual-friendly init."""

    def __init__(self, input_dim, hidden_dim, kernel_size, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size, padding=(pad_h, pad_w), bias=False,
        )
        # LayerNorm over channel dimension (applied after conv, before gate split)
        # Using GroupNorm with num_groups=1 is equivalent to LayerNorm for conv.
        self.norm = nn.GroupNorm(1, 4 * hidden_dim)
        self.dropout = nn.Dropout2d(dropout)

        # Initialize forget gate bias to +1 (encourage remembering early on)
        # The bias is in the norm layer; we add it via a learnable parameter.
        self.gate_bias = nn.Parameter(torch.zeros(4 * hidden_dim, 1, 1))
        # Set forget gate bias to 1.0
        nn.init.constant_(self.gate_bias[hidden_dim:2*hidden_dim], 1.0)

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.norm(self.conv(combined)) + self.gate_bias
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next = self.dropout(h_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        h, w = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
        )


class ConvLSTMFreqModel(nn.Module):
    """ConvLSTM for multi-task frequency prediction (dual-branch).

    Temporal branch: ConvLSTM encodes (B, T, N, C) -> h_temporal (d_temporal)
    Tabular branch: MLP encodes x_static (B, n_static) -> h_tabular (d_tabular)
    Fusion: concat + MLP head -> (B, 2)

    Input:
        x_temporal: (B, T, N_gen, C) e.g. (B, 11, 10, 7)
        x_static:   (B, n_static)
    Output:
        (B, 2) -- [y1, y2]
    """

    def __init__(self, n_features: int = 7, n_generators: int = 10,
                 hidden_channels: list = None, kernel_width: int = 3,
                 n_static: int = 7, mlp_hidden: int = 128,
                 output_dim: int = 2, dropout: float = 0.2,
                 tab_hidden: int = 256, tab_output: int = 128):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 128]

        # ── Temporal branch: ConvLSTM ──
        self.n_generators = n_generators
        self.cells = nn.ModuleList()
        in_ch = n_features
        for hch in hidden_channels:
            self.cells.append(ConvLSTMCell(in_ch, hch, (1, kernel_width), dropout))
            in_ch = hch

        self.gap = nn.AdaptiveAvgPool2d(1)
        d_temporal = hidden_channels[-1]

        # ── Tabular branch: processes x_static ──
        self.tabular_branch = TabularBranch(
            n_input=n_static, d_hidden=tab_hidden,
            d_output=tab_output, dropout=dropout,
        )

        # ── Fusion head ──
        fusion_dim = d_temporal + tab_output
        self.fusion_ln = nn.LayerNorm(fusion_dim)
        self.fusion_fc1 = nn.Linear(fusion_dim, mlp_hidden)
        self.fusion_fc2 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fusion_ln2 = nn.LayerNorm(mlp_hidden)
        self.fusion_drop = nn.Dropout(dropout)
        self.fusion_out = nn.Linear(mlp_hidden, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, orthogonal for conv."""
        for name, p in self.named_parameters():
            if 'conv.weight' in name and 'cells' in name:
                nn.init.orthogonal_(p)
            elif 'fusion_fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'fusion_out.weight' in name:
                nn.init.xavier_uniform_(p, gain=0.1)  # small init for output

    def forward(self, x_temporal, x_static):
        B, T, N, C = x_temporal.shape

        # ── Temporal branch ──
        # -> (B, T, C, 1, N)
        x = x_temporal.permute(0, 1, 3, 2).unsqueeze(3)

        device = x.device
        states = []
        for cell in self.cells:
            states.append(cell.init_hidden(B, (1, N), device))

        for t in range(T):
            inp = x[:, t]  # (B, C, 1, N)
            for i, cell in enumerate(self.cells):
                h, c = cell(inp if i == 0 else states[i-1][0], states[i])
                states[i] = (h, c)
                inp = h

        h_last = states[-1][0]  # (B, hidden[-1], 1, N)
        h_temporal = self.gap(h_last).flatten(1)  # (B, d_temporal)

        # ── Tabular branch ──
        h_tabular = self.tabular_branch(x_static)  # (B, tab_output)

        # ── Fusion ──
        combined = torch.cat([h_temporal, h_tabular], dim=-1)
        x = self.fusion_ln(combined)
        x = torch.nn.functional.gelu(self.fusion_fc1(x))
        x = self.fusion_drop(x)
        residual = x
        x = torch.nn.functional.gelu(self.fusion_fc2(x))
        x = self.fusion_ln2(x + residual)  # skip connection
        x = self.fusion_drop(x)
        return self.fusion_out(x)


def build_convlstm_from_trial(trial, n_features=7, n_generators=10, n_static=7):
    h1 = trial.suggest_categorical('hidden_1', [32, 64, 128])
    h2 = trial.suggest_categorical('hidden_2', [64, 128, 256])
    kw = trial.suggest_categorical('kernel_width', [3, 5])
    mlp = trial.suggest_categorical('mlp_hidden', [128, 256, 512])
    drop = trial.suggest_float('dropout', 0.05, 0.3)
    # ConvLSTM has strong temporal branch; keep tabular branch small-to-moderate
    tab_h = trial.suggest_categorical('tab_hidden', [32, 64, 128])
    tab_o = trial.suggest_categorical('tab_output', [16, 32, 64])

    kwargs = dict(n_features=n_features, n_generators=n_generators,
                  hidden_channels=[h1, h2], kernel_width=kw,
                  n_static=n_static, mlp_hidden=mlp, dropout=drop,
                  tab_hidden=tab_h, tab_output=tab_o)
    return ConvLSTMFreqModel(**kwargs), kwargs
