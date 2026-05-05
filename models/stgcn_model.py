#!/usr/bin/env python3
"""ST-GCN (Spatio-Temporal Graph Convolutional Network) for RepC input."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from models.tabular_branch import TabularBranch


class STBlock(nn.Module):
    """Spatio-Temporal block: GCN (spatial) + Conv1d (temporal)."""

    def __init__(self, in_channels, gcn_channels, tcn_channels,
                 tcn_kernel_size=3, dropout=0.1):
        super().__init__()
        self.gcn = GCNConv(in_channels, gcn_channels)
        self.tcn = nn.Conv1d(gcn_channels, tcn_channels, tcn_kernel_size,
                             padding=tcn_kernel_size // 2)
        self.bn = nn.BatchNorm1d(tcn_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (nn.Linear(in_channels, tcn_channels)
                         if in_channels != tcn_channels else nn.Identity())

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: (B, N, T, C)
        edge_index: (2, E)
        edge_weight: (E,) optional
        """
        B, N, T, C = x.shape

        # ── Spatial: GCN per timestep ──
        # Reshape to (B*T, N, C) for batched GCN
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, C)

        gcn_outs = []
        for bt in range(B * T):
            out = self.gcn(x_flat[bt], edge_index, edge_weight)
            gcn_outs.append(out)
        x_gcn = torch.stack(gcn_outs)  # (B*T, N, gcn_ch)
        x_gcn = x_gcn.reshape(B, T, N, -1).permute(0, 2, 1, 3)  # (B, N, T, gcn_ch)

        # ── Temporal: Conv1d per node ──
        B, N, T, C_gcn = x_gcn.shape
        x_tcn = x_gcn.reshape(B * N, T, C_gcn).permute(0, 2, 1)  # (B*N, C, T)
        x_tcn = self.dropout(F.relu(self.bn(self.tcn(x_tcn))))     # (B*N, tcn_ch, T)
        x_tcn = x_tcn.permute(0, 2, 1).reshape(B, N, T, -1)       # (B, N, T, tcn_ch)

        # ── Residual ──
        res = self.residual(x.reshape(-1, x.shape[-1])).reshape(B, N, T, -1)
        return F.relu(x_tcn + res)


class STGCNFreqModel(nn.Module):
    """ST-GCN for multi-task frequency prediction (non-batched, reference impl).

    Note: In practice, STGCNBatchedModel should be used for training
    as it is ~100x faster. This class is kept for compatibility.

    Input:
        x_node:     (B, N, T, C) node features
        x_static:   (B, n_static) operating condition parameters
        edge_index: (2, E) graph edges
        edge_weight: (E,) edge weights
    Output:
        (B, 2) -- [y1, y2]
    """

    def __init__(self, n_features: int = 7, n_nodes: int = 10,
                 gcn_channels: list = None, tcn_channels: list = None,
                 tcn_kernel_size: int = 3, n_static: int = 7,
                 output_dim: int = 2, dropout: float = 0.1,
                 tab_hidden: int = 256, tab_output: int = 128):
        super().__init__()
        if gcn_channels is None:
            gcn_channels = [32, 64]
        if tcn_channels is None:
            tcn_channels = [32, 64]

        self.blocks = nn.ModuleList()
        in_ch = n_features
        for gcn_ch, tcn_ch in zip(gcn_channels, tcn_channels):
            self.blocks.append(STBlock(in_ch, gcn_ch, tcn_ch, tcn_kernel_size, dropout))
            in_ch = tcn_ch

        # Readout: pool over nodes -> pool over time
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        d_temporal = tcn_channels[-1]

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

    def forward(self, x_node, x_static, edge_index, edge_weight=None):
        # x_node: (B, N, T, C)
        x = x_node
        for block in self.blocks:
            x = block(x, edge_index, edge_weight)

        # (B, N, T, C') -> mean over nodes -> (B, T, C')
        x = x.mean(dim=1)
        # Pool over time: (B, C', T) -> (B, C', 1)
        x = x.permute(0, 2, 1)
        h_temporal = self.time_pool(x).squeeze(-1)  # (B, d_temporal)

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


class STGCNBatchedModel(nn.Module):
    """Optimized ST-GCN using batched matrix multiplication (dual-branch).

    Uses dense adjacency matrix (registered as buffer) for efficient batched
    graph convolution. ~100x faster than looping over B*T individual GCN calls.

    Temporal branch: ST-GCN encodes (B, N, T, C) -> h_temporal
    Tabular branch: MLP encodes x_static (B, n_static) -> h_tabular
    Fusion: concat + MLP head -> (B, 2)

    Input:
        x_node:     (B, N, T, C) node features
        x_static:   (B, n_static) operating conditions
        edge_index: IGNORED (uses internal adj_norm buffer)
        edge_weight: IGNORED
    Output:
        (B, 2)
    """

    def __init__(self, n_features: int = 7, n_nodes: int = 10,
                 hidden_channels: list = None, tcn_kernel_size: int = 3,
                 n_static: int = 7, output_dim: int = 2, dropout: float = 0.1,
                 tab_hidden: int = 256, tab_output: int = 128):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64]

        self.gcn_linears = nn.ModuleList()
        self.tcn_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.residuals = nn.ModuleList()

        in_ch = n_features
        for hch in hidden_channels:
            self.gcn_linears.append(nn.Linear(in_ch, hch))
            self.tcn_convs.append(nn.Conv1d(hch, hch, tcn_kernel_size,
                                            padding=tcn_kernel_size // 2))
            self.bns.append(nn.BatchNorm1d(hch))
            self.residuals.append(nn.Linear(in_ch, hch) if in_ch != hch else nn.Identity())
            in_ch = hch

        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        d_temporal = hidden_channels[-1]

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
            if 'tcn_convs' in name and 'weight' in name:
                nn.init.kaiming_normal_(p, nonlinearity='relu')
            elif 'gcn_linears' in name and 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'fusion_out.weight' in name:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'fusion_fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(p)

    def set_adj(self, adj_norm: torch.Tensor):
        """Register normalized adjacency matrix as buffer."""
        self.register_buffer('adj_norm', adj_norm)

    def forward(self, x_node, x_static, edge_index=None, edge_weight=None):
        """
        x_node:   (B, N, T, C)
        x_static: (B, n_static)
        edge_index/edge_weight: ignored (uses self.adj_norm)
        """
        B, N, T, C = x_node.shape
        x = x_node

        for gcn_lin, tcn_conv, bn, res_lin in zip(
                self.gcn_linears, self.tcn_convs, self.bns, self.residuals):
            # Residual
            res = res_lin(x.reshape(-1, x.shape[-1])).reshape(B, N, T, -1)

            # GCN: A_norm @ (x @ W)
            x_bt = x.permute(0, 2, 1, 3)  # (B, T, N, C)
            x_bt = gcn_lin(x_bt)            # (B, T, N, H)
            x_bt = torch.matmul(self.adj_norm, x_bt)  # broadcasts over B,T
            x = x_bt.permute(0, 2, 1, 3)   # (B, N, T, H)

            # TCN per node
            H = x.shape[-1]
            x_tcn = x.reshape(B * N, T, H).permute(0, 2, 1)
            x_tcn = self.dropout(F.relu(bn(tcn_conv(x_tcn))))
            T_new = x_tcn.shape[-1]
            x = x_tcn.permute(0, 2, 1).reshape(B, N, T_new, -1)
            T = T_new

            # Residual connection
            x = F.relu(x + res)

        # Pool: nodes -> mean, time -> adaptive pool
        x = x.mean(dim=1)                     # (B, T, H)
        h_temporal = self.time_pool(x.permute(0, 2, 1)).squeeze(-1)  # (B, d_temporal)

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


def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    return torch.FloatTensor(adj_norm)


def build_stgcn_from_trial(trial, n_features=7, n_nodes=10, n_static=7,
                            adj_norm=None):
    h1 = trial.suggest_categorical('hidden_ch_1', [16, 32, 64])
    h2 = trial.suggest_categorical('hidden_ch_2', [32, 64, 128])
    kernel = trial.suggest_categorical('tcn_kernel', [3, 5])
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    # ST-GCN's graph conv is the core; keep tabular branch small to avoid overwhelming it
    tab_h = trial.suggest_categorical('tab_hidden', [32, 64, 128])
    tab_o = trial.suggest_categorical('tab_output', [16, 32, 64])

    kwargs = dict(n_features=n_features, n_nodes=n_nodes,
                  hidden_channels=[h1, h2], tcn_kernel_size=kernel,
                  n_static=n_static, dropout=dropout,
                  tab_hidden=tab_h, tab_output=tab_o)
    model = STGCNBatchedModel(**kwargs)
    if adj_norm is not None:
        model.set_adj(adj_norm if isinstance(adj_norm, torch.Tensor) else torch.FloatTensor(adj_norm))
    return model, kwargs
