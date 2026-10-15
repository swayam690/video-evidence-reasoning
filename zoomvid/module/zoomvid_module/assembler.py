"""ZoomVid Assembler.

Assembles selected high-resolution frame patches and injects temporal position
embeddings, projecting them into the frozen LLM backbone's hidden dimension.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class TemporalPositionEncoding(nn.Module):
    """Sinusoidal temporal position encoding based on continuous timestamps in seconds."""

    def __init__(self, d_model: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal position embedding from timestamps.

        Args:
            timestamps: Tensor of shape [B, L] denoting timestamp in seconds.

        Returns:
            Tensor of shape [B, L, d_model].
        """
        device = timestamps.device
        half_dim = self.d_model // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        scaled = timestamps.unsqueeze(-1).float() * emb.unsqueeze(0).unsqueeze(0)
        sin_emb = torch.sin(scaled)
        cos_emb = torch.cos(scaled)
        pos_emb = torch.cat([sin_emb, cos_emb], dim=-1)
        proj_weight = self.proj[0].weight
        return self.proj(pos_emb.to(device=proj_weight.device, dtype=proj_weight.dtype))


class Assembler(nn.Module):
    """Assembles gathered visual tokens into a budget-compliant sequence."""

    def __init__(
        self,
        in_features: int = 2048,
        out_features: int = 2048,
        sketch_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Assembler.

        Args:
            in_features: Input vision feature dimension D (from patch merger).
            out_features: Target LLM hidden dimension D_llm.
            sketch_dim: Optional sketch feature dimension for global context fusion.
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sketch_dim = sketch_dim

        # Projection layer to LLM hidden dimension
        self.proj = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        # Context fusion projection
        if sketch_dim is not None:
            self.sketch_proj = nn.Sequential(
                nn.Linear(sketch_dim, in_features),
                nn.GELU(),
            )
        else:
            self.sketch_proj = None

        # Temporal position encoder
        self.temporal_pos_enc = TemporalPositionEncoding(d_model=out_features)

    def forward(
        self,
        frame_feats: torch.Tensor,
        selected_indices: torch.Tensor,
        timestamps: torch.Tensor,
        token_budget: int,
        scores: Optional[torch.Tensor] = None,
        sketch_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Gather selected frames, add temporal position embeddings, and project.

        Args:
            frame_feats: Original frame patch features [B, N, P, D].
            selected_indices: Indices of selected frames [B, k].
            timestamps: Original frame timestamps [B, N].
            token_budget: Maximum tokens allowed.
            scores: Optional router importance scores [B, N] for differentiable weighting.
            sketch_tokens: Optional sketch tokens [B, N, sketch_dim] for context fusion.

        Returns:
            Tuple of:
            - tokens: Assembled token sequence [B, T, D_llm] where T <= token_budget.
            - aux: Diagnostics dict.
        """
        b, n, p, d = frame_feats.shape
        _, k = selected_indices.shape

        proj_dtype = self.proj.weight.dtype
        proj_device = self.proj.weight.device

        # 1. Gather selected frame patches: [B, k, P, D]
        gather_idx = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(b, k, p, d)
        selected_feats = torch.gather(frame_feats.to(device=proj_device, dtype=proj_dtype), 1, gather_idx)

        # 2. Context fusion from SketchEncoder (enables end-to-end gradient flow)
        if sketch_tokens is not None and self.sketch_proj is not None:
            gather_sketch_idx = selected_indices.unsqueeze(-1).expand(b, k, self.sketch_dim)
            gathered_sketch = torch.gather(sketch_tokens, 1, gather_sketch_idx)
            sketch_ctx = self.sketch_proj(gathered_sketch).unsqueeze(2)  # [B, k, 1, D]
            selected_feats = selected_feats + sketch_ctx

        # 3. Differentiable score weighting from Router
        if scores is not None:
            gathered_scores = torch.gather(scores, 1, selected_indices).unsqueeze(-1).unsqueeze(-1)
            selected_feats = selected_feats * (1.0 + gathered_scores)

        # 4. Extract corresponding timestamps: [B, k] -> expand across P patches -> [B, k * P]
        selected_times = torch.gather(timestamps.to(proj_device), 1, selected_indices)
        patch_times = selected_times.unsqueeze(-1).expand(b, k, p).reshape(b, k * p)

        # 5. Flatten patches into sequence: [B, k * P, D]
        x = selected_feats.reshape(b, k * p, d)

        # 6. Project to LLM hidden dimension
        projected = self.proj(x)

        # 7. Add temporal position embedding
        time_pos = self.temporal_pos_enc(patch_times).to(dtype=projected.dtype)
        tokens = self.norm(projected + time_pos)
        tokens = self.dropout(tokens)

        # 8. Enforce budget constraint
        if tokens.shape[1] > token_budget:
            tokens = tokens[:, :token_budget]

        aux = {
            "num_frames_assembled": k,
            "tokens_produced": tokens.shape[1],
            "patches_per_frame": p,
        }
        return tokens, aux
