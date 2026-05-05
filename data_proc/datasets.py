#!/usr/bin/env python3
"""
PyTorch Dataset classes for three representations.

TabularDataset  → LightGBM (numpy), KAN (torch)
TensorDataset   → ConvLSTM, PatchTST, Mamba
GraphDataset    → ST-GCN
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Representation A: flattened 1D features.

    Used by KAN (torch). LightGBM uses numpy arrays directly.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'x': self.X[idx], 'y': self.y[idx]}


class TensorDataset(Dataset):
    """Representation B: (T, N, C) temporal tensor + (7,) static features.

    Used by ConvLSTM, PatchTST, Mamba.
    """
    def __init__(self, X_temporal: np.ndarray, X_static: np.ndarray, y: np.ndarray):
        """
        Args:
            X_temporal: (N_samples, T, N_nodes, C_features)
            X_static:   (N_samples, 7)
            y:          (N_samples, 2)
        """
        self.X_temporal = torch.FloatTensor(X_temporal)
        self.X_static = torch.FloatTensor(X_static)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X_temporal)

    def __getitem__(self, idx):
        return {
            'x_temporal': self.X_temporal[idx],
            'x_static': self.X_static[idx],
            'y': self.y[idx],
        }


class GraphDataset(Dataset):
    """Representation C: (N_nodes, T, C) node features + shared adjacency.

    Used by ST-GCN. Edge index and weights are shared across all samples.
    """
    def __init__(self, X_node: np.ndarray, X_static: np.ndarray, y: np.ndarray,
                 edge_index: np.ndarray, edge_weight: np.ndarray = None):
        """
        Args:
            X_node:      (N_samples, N_nodes, T, C)
            X_static:    (N_samples, 7)
            y:           (N_samples, 2)
            edge_index:  (2, E) — shared across all samples
            edge_weight: (E,) — optional edge weights
        """
        self.X_node = torch.FloatTensor(X_node)
        self.X_static = torch.FloatTensor(X_static)
        self.y = torch.FloatTensor(y)
        self.edge_index = torch.LongTensor(edge_index)
        self.edge_weight = torch.FloatTensor(edge_weight) if edge_weight is not None else None

    def __len__(self):
        return len(self.X_node)

    def __getitem__(self, idx):
        item = {
            'x_node': self.X_node[idx],
            'x_static': self.X_static[idx],
            'y': self.y[idx],
            'edge_index': self.edge_index,
        }
        if self.edge_weight is not None:
            item['edge_weight'] = self.edge_weight
        return item


def adjacency_to_edge_index(adj_matrix: np.ndarray, threshold: float = 0.01):
    """Convert adjacency matrix to PyG-style edge_index and edge_weight.

    Args:
        adj_matrix: (N, N) adjacency matrix (with self-loops)
        threshold: Minimum weight to keep an edge

    Returns:
        edge_index: (2, E) np.ndarray
        edge_weight: (E,) np.ndarray
    """
    rows, cols = np.where(adj_matrix > threshold)
    edge_index = np.stack([rows, cols], axis=0)
    edge_weight = adj_matrix[rows, cols]
    return edge_index, edge_weight


def load_representations(rep_dir: str, ms: int, split: str, rep_type: str = 'all',
                         adj_path: str = None) -> dict:
    """Load pre-built representations from npy files.

    Args:
        rep_dir: Base directory containing repA/, repB/, repC/ subdirs
        ms: Multi-scale time step value
        split: Split name (train, val, test, cross_cond_test, etc.)
        rep_type: 'A', 'B', 'C', or 'all'
        adj_path: Path to adjacency.npy (required for rep_type='C' or 'all')

    Returns:
        dict with loaded arrays
    """
    result = {}

    if rep_type in ('A', 'all'):
        d = os.path.join(rep_dir, 'repA', f'ms{ms}')
        result['X_A'] = np.load(os.path.join(d, f'X_{split}.npy'))
        result['y_A'] = np.load(os.path.join(d, f'y_{split}.npy'))

    if rep_type in ('B', 'all'):
        d = os.path.join(rep_dir, 'repB', f'ms{ms}')
        result['X_temporal'] = np.load(os.path.join(d, f'X_temporal_{split}.npy'))
        result['X_static'] = np.load(os.path.join(d, f'X_static_{split}.npy'))
        result['y_B'] = np.load(os.path.join(d, f'y_{split}.npy'))

    if rep_type in ('C', 'all'):
        d = os.path.join(rep_dir, 'repC', f'ms{ms}')
        result['X_node'] = np.load(os.path.join(d, f'X_node_{split}.npy'))
        result['X_static_C'] = np.load(os.path.join(d, f'X_static_{split}.npy'))
        result['y_C'] = np.load(os.path.join(d, f'y_{split}.npy'))

        if adj_path:
            adj = np.load(adj_path)
            edge_index, edge_weight = adjacency_to_edge_index(adj)
            result['edge_index'] = edge_index
            result['edge_weight'] = edge_weight

    return result


import os  # needed for load_representations
