"""ZoomVid Sketch Encoder.

Compresses dense per-frame patch features into a temporally contextualized
global video sketch using spatial pooling and a lightweight bidirectional GRU.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class SketchEncoder(nn.Module):
    """Encodes dense video frames into a compact temporal sketch sequence."""

    def __init__(
        self,
        in_features: int = 2048,
        sketch_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize SketchEncoder.

        Args:
            in_features: Feature dimension from patch merger (D).
            sketch_dim: Hidden dimension for sketch representations.
            num_layers: Number of bidirectional GRU layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_features = in_features
        self.sketch_dim = sketch_dim

        # Spatial compression: projects pooled patch features to sketch_dim
        self.spatial_proj = nn.Sequential(
            nn.Linear(in_features, sketch_dim),
            nn.LayerNorm(sketch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal modeling: bidirectional GRU across frames
        self.temporal_rnn = nn.GRU(
            input_size=sketch_dim,
            hidden_size=sketch_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(sketch_dim)

    def forward(
        self,
        frame_feats: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through sketch encoder.

        Args:
            frame_feats: Tensor of shape [B, N, P, D].
            timestamps: Optional frame timestamps [B, N].

        Returns:
            Tuple of:
            - sketch_tokens: [B, N, sketch_dim]
            - aux: Dictionary with intermediate representations.
        """
        b, n, p, d = frame_feats.shape

        # 1. Spatial mean pooling across patches: [B, N, P, D] -> [B, N, D]
        # Ensure input dtype matches spatial projection
        proj_dtype = self.spatial_proj[0].weight.dtype
        proj_device = self.spatial_proj[0].weight.device
        x = frame_feats.to(device=proj_device, dtype=proj_dtype).mean(dim=2)

        # 2. Project to sketch dimension: [B, N, sketch_dim]
        spatial_summary = self.spatial_proj(x)

        # 3. Temporal contextualization via bidirectional GRU
        temporal_out, _ = self.temporal_rnn(spatial_summary)

        # 4. Residual connection + normalization
        sketch_tokens = self.norm(spatial_summary + temporal_out)

        aux = {
            "spatial_summary": spatial_summary,
            "temporal_out": temporal_out,
            "num_frames": n,
        }
        return sketch_tokens, aux
