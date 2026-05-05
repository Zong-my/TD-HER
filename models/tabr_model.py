#!/usr/bin/env python3
"""TabR (Tabular Retrieval-augmented model) for tabular regression.

Reference: Gorishniy et al., "TabR: Tabular Deep Learning Meets Nearest Neighbors,"
ICLR 2024. Combines a learned encoder with k-nearest-neighbor retrieval from the
training set. For each query, the model retrieves K neighbors in the learned
embedding space, aggregates their labels via attention, and combines with the
query's own representation for prediction.

Key design choices:
- Residual MLP blocks for the encoder (following the paper).
- Efficient KNN via torch.cdist + topk (no FAISS dependency for 121k samples).
- Training set cache is precomputed and stored as a buffer.
- Attention-weighted neighbor label aggregation with learnable temperature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, d)
        self.linear2 = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class TabRModel(nn.Module):
    """TabR for multi-task regression on tabular data (RepA).

    Input:
        x: (B, D) tabular features (post-mRMR, normalized)
    Output:
        (B, output_dim) — default (B, 2) for [y1, y2]

    At training time, the model uses the current mini-batch + cached training
    set embeddings for retrieval. At inference time, the full training set
    cache is used.
    """

    def __init__(self, n_features: int, d_model: int = 256,
                 n_blocks: int = 3, k_neighbors: int = 96,
                 dropout: float = 0.1, output_dim: int = 2):
        super().__init__()
        self.d_model = d_model
        self.k_neighbors = k_neighbors
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Encoder: stack of residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(d_model, dropout) for _ in range(n_blocks)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Learnable temperature for neighbor attention
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        # Neighbor value projection: maps (d_model + output_dim) -> d_model
        self.neighbor_proj = nn.Linear(d_model + output_dim, d_model)

        # Output head with skip connection
        # Input: query_repr (d_model) + neighbor_context (d_model) = 2*d_model
        head_in = 2 * d_model
        self.head_fc1 = nn.Linear(head_in, d_model)
        self.head_fc2 = nn.Linear(d_model, d_model)
        self.head_ln = nn.LayerNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.head_out = nn.Linear(d_model, output_dim)

        self._init_weights()

        # Training set cache (set via set_training_cache)
        self._train_embeddings = None  # (N_train, d_model)
        self._train_labels = None      # (N_train, output_dim)

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'head_out.weight' in name:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'linear' in name and 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'input_proj.weight' in name:
                nn.init.xavier_uniform_(p)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features to embedding space."""
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.encoder_norm(x)

    @torch.no_grad()
    def build_training_cache(self, train_x: torch.Tensor, train_y: torch.Tensor,
                             batch_size: int = 4096):
        """Precompute and cache training set embeddings.

        Args:
            train_x: (N_train, D) training features
            train_y: (N_train, output_dim) training targets
            batch_size: batch size for encoding
        """
        was_training = self.training
        self.eval()
        embeddings = []
        for i in range(0, len(train_x), batch_size):
            batch = train_x[i:i+batch_size].to(next(self.parameters()).device)
            embeddings.append(self.encode(batch).cpu())
        self._train_embeddings = torch.cat(embeddings, dim=0)
        self._train_labels = train_y.detach().cpu().clone()
        self.train(was_training)

    def _retrieve_neighbors(self, query_emb: torch.Tensor):
        """Retrieve K nearest neighbors from training cache.

        Args:
            query_emb: (B, d_model) query embeddings

        Returns:
            neighbor_emb: (B, K, d_model) neighbor embeddings
            neighbor_labels: (B, K, output_dim) neighbor labels
            distances: (B, K) distances to neighbors
        """
        device = query_emb.device
        train_emb = self._train_embeddings.to(device)
        train_labels = self._train_labels.to(device)

        # Compute distances: (B, N_train)
        dists = torch.cdist(query_emb, train_emb)  # (B, N_train)

        # Top-K nearest (smallest distance)
        K = min(self.k_neighbors, train_emb.shape[0])
        topk_dists, topk_idx = dists.topk(K, dim=1, largest=False)  # (B, K)

        # Gather neighbor embeddings and labels
        neighbor_emb = train_emb[topk_idx]       # (B, K, d_model)
        neighbor_labels = train_labels[topk_idx]  # (B, K, output_dim)

        return neighbor_emb, neighbor_labels, topk_dists

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) tabular features
        Returns: (B, output_dim)
        """
        # Encode query
        query_emb = self.encode(x)  # (B, d_model)

        if self._train_embeddings is not None:
            # Retrieval mode: use cached training set
            neighbor_emb, neighbor_labels, dists = self._retrieve_neighbors(query_emb)

            # Attention-weighted aggregation of neighbors
            temperature = torch.exp(self.log_temperature).clamp(min=0.01)
            attn_logits = -dists / temperature  # (B, K)
            attn_weights = F.softmax(attn_logits, dim=1)  # (B, K)

            # Combine neighbor embedding + label -> value
            neighbor_features = torch.cat([neighbor_emb, neighbor_labels], dim=-1)
            neighbor_values = self.neighbor_proj(neighbor_features)  # (B, K, d_model)

            # Weighted sum
            context = torch.einsum('bk,bkd->bd', attn_weights, neighbor_values)
        else:
            # No cache (fallback): use zero context
            context = torch.zeros_like(query_emb)

        # Combine query representation + neighbor context
        combined = torch.cat([query_emb, context], dim=-1)  # (B, 2*d_model)

        # MLP head with skip connection
        x = F.gelu(self.head_fc1(combined))
        x = self.head_drop(x)
        residual = x
        x = F.gelu(self.head_fc2(x))
        x = self.head_ln(x + residual)
        x = self.head_drop(x)
        return self.head_out(x)


def build_tabr_from_trial(trial, n_features: int, n_static: int = 0):
    """Build TabR model from Optuna trial."""
    if n_features >= 1024:
        # IEEE300 RepA keeps a larger mRMR union. Retrieval cost scales with
        # batch size, training-set size, and embedding dimension, so the
        # scalability protocol uses a compact TabR search space.
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        n_blocks = trial.suggest_int('n_blocks', 1, 3)
        k_neighbors = trial.suggest_categorical('k_neighbors', [32, 64, 96])
    else:
        d_model = trial.suggest_categorical('d_model', [128, 256, 512])
        n_blocks = trial.suggest_int('n_blocks', 1, 4)
        k_neighbors = trial.suggest_categorical('k_neighbors', [32, 64, 96, 128, 192])
    dropout = trial.suggest_float('dropout', 0.0, 0.3)

    kwargs = dict(
        n_features=n_features, d_model=d_model, n_blocks=n_blocks,
        k_neighbors=k_neighbors, dropout=dropout, output_dim=2,
    )
    return TabRModel(**kwargs), kwargs
