#!/usr/bin/env python3
"""KAN (Kolmogorov-Arnold Network) regressor for interpretable prediction.

Uses efficient-kan implementation. Two separate instances are created
for y1 (fpu_deltamax) and y2 (t_delta) since KAN works on tabular RepA.

Key notes:
- KAN's spline-based architecture naturally provides smooth function
  approximation, which suits the continuous physics of frequency response.
- Grid size and spline order are the key hyperparameters controlling
  approximation capacity vs. overfitting.
"""

import torch
import torch.nn as nn
from efficient_kan import KAN


class KANRegressor(nn.Module):
    """KAN model for single-target regression (RepA input).

    Two separate instances are created for y1 and y2.
    """
    def __init__(self, input_dim: int, hidden_dims: list = None,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = [input_dim] + hidden_dims + [1]
        self.kan = KAN(layers, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        """x: (B, D) -> (B,)"""
        return self.kan(x).squeeze(-1)


def build_kan_from_trial(trial, input_dim: int) -> KANRegressor:
    """Build KAN model from Optuna trial.

    Architecture sized for KAN's sweet spot: moderate hidden dims,
    limited grid_size to avoid parameter explosion on high-dim input.
    """
    h1 = trial.suggest_categorical('hidden_1', [32, 64, 128])
    h2 = trial.suggest_categorical('hidden_2', [16, 32, 64])
    h3 = trial.suggest_categorical('hidden_3', [8, 16, 32])
    grid = trial.suggest_int('grid_size', 3, 5)
    order = trial.suggest_int('spline_order', 2, 3)

    kwargs = dict(input_dim=input_dim, hidden_dims=[h1, h2, h3],
                  grid_size=grid, spline_order=order)
    return KANRegressor(**kwargs), kwargs
