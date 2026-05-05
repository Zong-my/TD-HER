#!/usr/bin/env python3
"""
Target-Asymmetric Decoder (TAD) Wrapper.

Wraps any dual-branch model to replace the shared fusion head with
target-specific gated decoders. Each target learns its own routing
weights between temporal and tabular branch outputs.

Uses forward hooks to intercept h_temporal and h_tabular from the
base model, making it architecture-agnostic.

After training, the learned gate values reveal which branch each target
relies on — providing architecture-level validation of the physical
asymmetry between Δf and t_nadir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TADWrapper(nn.Module):
    """Target-Asymmetric Decoder with learnable gated routing.

    Intercepts the base model's temporal and tabular branch outputs
    via hooks, then routes them through target-specific gated decoders.

    Gate parameterization: softmax over 2 logits per target.
    After training, get_gate_values() returns the learned routing weights.
    """

    def __init__(self, base_model: nn.Module,
                 d_temporal: int, d_tabular: int,
                 d_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.d_temporal = d_temporal
        self.d_tabular = d_tabular

        # Storage for intercepted branch outputs
        self._h_temporal = None
        self._h_tabular = None

        # Register hooks on the tabular branch output
        self._hook_handles = []
        self._register_hooks()

        # Learnable routing gates (logits, will be softmaxed)
        self.gate_y1_logits = nn.Parameter(torch.zeros(2))
        self.gate_y2_logits = nn.Parameter(torch.zeros(2))

        # Project both branches to common dimension
        self.proj_temporal = nn.Linear(d_temporal, d_hidden)
        self.proj_tabular = nn.Linear(d_tabular, d_hidden)

        # Target-specific decoder heads
        self.head_y1 = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )
        self.head_y2 = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.proj_temporal, self.proj_tabular,
                       self.head_y1, self.head_y2]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _register_hooks(self):
        """Register forward hooks to intercept branch outputs."""
        model = self.base_model

        # Hook on tabular_branch — captures h_tabular
        def tabular_hook(module, input, output):
            self._h_tabular = output

        if hasattr(model, 'tabular_branch'):
            h = model.tabular_branch.register_forward_hook(tabular_hook)
            self._hook_handles.append(h)

        # Hook on the layer right before fusion to get h_temporal
        # For ConvLSTM: gap layer output
        # For PatchTST: channel_agg output
        # For Mamba: temporal_proj output
        def temporal_hook(module, input, output):
            self._h_temporal = output.flatten(1) if output.dim() > 2 else output

        self._mamba_mode = False
        if hasattr(model, 'gap'):  # ConvLSTM
            h = model.gap.register_forward_hook(temporal_hook)
            self._hook_handles.append(h)
        elif hasattr(model, 'channel_agg'):  # PatchTST
            h = model.channel_agg.register_forward_hook(temporal_hook)
            self._hook_handles.append(h)
        elif hasattr(model, 'pool_attn'):  # Mamba
            # For Mamba, we hook final_norm to get (B,T,d_model), then
            # apply pool_attn ourselves in forward()
            self._mamba_mode = True
            def mamba_temporal_hook(module, input, output):
                self._h_temporal_raw = output  # (B, T, d_model)
            h = model.final_norm.register_forward_hook(mamba_temporal_hook)
            self._hook_handles.append(h)

    def forward(self, x_temporal, x_static):
        # Run base model forward (hooks will capture branch outputs)
        # We discard the base model's output and use our own decoders
        _ = self.base_model(x_temporal, x_static)

        # For Mamba: apply attention pooling to get h_temporal from raw
        if self._mamba_mode:
            x_raw = self._h_temporal_raw  # (B, T, d_model)
            attn_weights = F.softmax(
                self.base_model.pool_attn(x_raw).squeeze(-1), dim=1)
            self._h_temporal = torch.einsum('bt,btd->bd', attn_weights, x_raw)

        h_temporal = self._h_temporal
        h_tabular = self._h_tabular

        if h_temporal is None or h_tabular is None:
            raise RuntimeError("Hooks failed to capture branch outputs. "
                              f"h_temporal={h_temporal is not None}, "
                              f"h_tabular={h_tabular is not None}")

        # Project to common dimension
        h_t = self.proj_temporal(h_temporal)
        h_s = self.proj_tabular(h_tabular)

        # Gated routing per target
        gate_y1 = F.softmax(self.gate_y1_logits, dim=0)
        gate_y2 = F.softmax(self.gate_y2_logits, dim=0)

        h_y1 = gate_y1[0] * h_t + gate_y1[1] * h_s
        h_y2 = gate_y2[0] * h_t + gate_y2[1] * h_s

        y1 = self.head_y1(h_y1)
        y2 = self.head_y2(h_y2)

        return torch.cat([y1, y2], dim=-1)  # (B, 2)

    def get_gate_values(self):
        """Return learned gate weights (after softmax)."""
        with torch.no_grad():
            g1 = F.softmax(self.gate_y1_logits, dim=0).cpu().numpy()
            g2 = F.softmax(self.gate_y2_logits, dim=0).cpu().numpy()
        return {
            'y1_temporal_weight': float(g1[0]),
            'y1_tabular_weight': float(g1[1]),
            'y2_temporal_weight': float(g2[0]),
            'y2_tabular_weight': float(g2[1]),
        }

    def remove_hooks(self):
        """Clean up hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
