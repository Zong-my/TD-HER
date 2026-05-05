#!/usr/bin/env python3
"""FT-Transformer (Feature Tokenizer + Transformer) for tabular regression.

Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data,"
NeurIPS 2021. Each numerical feature is independently projected into a d-dimensional
embedding (feature tokenizer), producing D tokens. A [CLS] token is prepended and
processed by a standard Transformer encoder. The final [CLS] representation feeds
an MLP head for multi-task regression.

Key design choices:
- Pre-LayerNorm Transformer blocks for stable training.
- Learnable [CLS] token for aggregation (avoids mean-pool over features).
- Per-feature linear projection (no shared weights across features).
- Skip connection + LayerNorm in the output head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    """Projects each numerical feature to a d_model-dimensional embedding."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        # Each feature gets its own linear projection + bias
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) numerical features
        Returns: (B, D, d_model) feature tokens
        """
        # x.unsqueeze(-1): (B, D, 1) * weight: (D, d_model) -> (B, D, d_model)
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformerModel(nn.Module):
    """FT-Transformer for multi-task regression on tabular data (RepA).

    Input:
        x: (B, D) tabular features (post-mRMR, normalized)
    Output:
        (B, output_dim) — default (B, 2) for [y1, y2]
    """

    def __init__(self, n_features: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 output_dim: int = 2):
        super().__init__()
        self.n_features = n_features

        # Feature tokenizer: each feature -> d_model embedding
        self.tokenizer = FeatureTokenizer(n_features, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Input dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Pre-norm Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,  # Pre-LayerNorm for better gradient flow
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Attention dropout (applied to [CLS] attention scores)
        self.attn_dropout = nn.Dropout(attention_dropout)

        # Output head with skip connection
        self.head_fc1 = nn.Linear(d_model, d_model)
        self.head_fc2 = nn.Linear(d_model, d_model)
        self.head_ln = nn.LayerNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.head_out = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'head_out.weight' in name:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'head_fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) tabular features
        Returns: (B, output_dim)
        """
        B = x.shape[0]

        # Tokenize: (B, D) -> (B, D, d_model)
        tokens = self.tokenizer(x)

        # Prepend [CLS]: (B, 1+D, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.embed_dropout(tokens)

        # Transformer encoding
        encoded = self.encoder(tokens)  # (B, 1+D, d_model)

        # Extract [CLS] representation
        cls_out = encoded[:, 0]  # (B, d_model)

        # MLP head with skip connection
        x = F.gelu(self.head_fc1(cls_out))
        x = self.head_drop(x)
        residual = x
        x = F.gelu(self.head_fc2(x))
        x = self.head_ln(x + residual)
        x = self.head_drop(x)
        return self.head_out(x)


def build_ft_transformer_from_trial(trial, n_features: int, n_static: int = 0):
    """Build FT-Transformer from Optuna trial.

    Note: n_static is unused (FT-Transformer uses all features as tokens).
    Kept for interface consistency with other build functions.
    """
    if n_features >= 1024:
        # Full feature-token attention is quadratic in the number of selected
        # RepA features. IEEE300 uses a much larger mRMR union than IEEE39, so
        # the high-dimensional protocol intentionally keeps FT-Transformer as a
        # compact tabular Transformer baseline instead of an over-budget model.
        d_model = trial.suggest_categorical('d_model', [16, 32, 64])
        valid_heads = [h for h in [1, 2, 4] if d_model % h == 0]
        n_heads = trial.suggest_categorical('n_heads', valid_heads)
        n_layers = trial.suggest_int('n_layers', 1, 2)
        d_ff_factor = trial.suggest_categorical('d_ff_factor', [1, 2])
    else:
        d_model = trial.suggest_categorical('d_model', [64, 128, 192, 256])
        n_heads = trial.suggest_categorical('n_heads', [4, 8])
        n_layers = trial.suggest_int('n_layers', 2, 6)
        d_ff_factor = trial.suggest_categorical('d_ff_factor', [2, 3, 4])
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    attn_dropout = trial.suggest_float('attn_dropout', 0.0, 0.3)

    kwargs = dict(
        n_features=n_features, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_model * d_ff_factor,
        dropout=dropout, attention_dropout=attn_dropout,
        output_dim=2,
    )
    return FTTransformerModel(**kwargs), kwargs
