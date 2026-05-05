#!/usr/bin/env python3
"""Shared TabularBranch module for DL models.

Processes the high-dimensional x_static vector (e.g. 1400 dims of statistical
features) into a compact embedding that the fusion head can consume alongside
the temporal branch output.
"""

import torch
import torch.nn as nn


class TabularBranch(nn.Module):
    """MLP branch for processing large tabular/static feature vectors.

    Architecture: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout

    Args:
        n_input:  dimensionality of x_static (e.g. 1400)
        d_hidden: hidden layer width (default 256)
        d_output: output embedding size (default 128)
        dropout:  dropout rate (default 0.1)
    """

    def __init__(self, n_input: int, d_hidden: int = 256,
                 d_output: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_input),
            nn.Linear(n_input, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_output),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_static: (B, n_input)
        Returns:
            (B, d_output)
        """
        return self.net(x_static)
