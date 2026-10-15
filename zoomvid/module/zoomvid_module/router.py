"""ZoomVid Router.

Learns frame importance scores from sketch representations and question embeddings,
and selects the top-k most informative frames that fit within the token budget.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Learned temporal frame selection policy network."""

    def __init__(
        self,
        sketch_dim: int = 512,
        question_dim: Optional[int] = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Router.

        Args:
            sketch_dim: Dimension of sketch tokens from SketchEncoder.
            question_dim: Dimension of question embedding (if conditioned on query).
            hidden_dim: Intermediate MLP hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.sketch_dim = sketch_dim

        # Frame importance scorer MLP
        self.scorer = nn.Sequential(
            nn.Linear(sketch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Optional question conditioning projection
        self.question_proj = (
            nn.Linear(question_dim, sketch_dim) if question_dim is not None else None
        )

    def forward(
        self,
        sketch_tokens: torch.Tensor,
        token_budget: int,
        timestamps: torch.Tensor,
        patches_per_frame: int = 144,
        question_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute frame importance scores and select top-k frames.

        Args:
            sketch_tokens: Coarse video representations [B, N, sketch_dim].
            token_budget: Maximum tokens allocated for visual context.
            timestamps: Original frame timestamps [B, N].
            patches_per_frame: Number of spatial patch tokens per frame P.
            question_emb: Optional query embedding [B, D_q] or [B, L_q, D_q].

        Returns:
            Dictionary containing:
            - 'selected_indices': LongTensor [B, k] of selected frame indices (chronological).
            - 'scores': FloatTensor [B, N] of frame importance scores in [0, 1].
            - 'segment_spans': List of temporal intervals per batch item.
            - 'num_frames_selected': Number of selected frames k.
        """
        b, n, d = sketch_tokens.shape

        # 1. Compute frame importance logits from sketch tokens
        logits = self.scorer(sketch_tokens).squeeze(-1)  # [B, N]

        # 2. Query conditioning (if question embedding is available)
        if question_emb is not None and self.question_proj is not None:
            q = question_emb
            if q.ndim == 3:
                q = q.mean(dim=1)  # [B, D_q]
            q_proj = self.question_proj(q).unsqueeze(1)  # [B, 1, sketch_dim]
            # Cross-attention / cosine affinity between question and frames
            q_norm = F.normalize(q_proj, dim=-1)
            s_norm = F.normalize(sketch_tokens, dim=-1)
            query_affinity = (q_norm * s_norm).sum(dim=-1)  # [B, N]
            logits = logits + query_affinity

        scores = torch.sigmoid(logits)  # [B, N]

        # 3. Budget constraint: determine k frames that fit within budget
        max_k = max(1, token_budget // patches_per_frame)
        k = min(n, max_k)

        # 4. Top-k selection
        _, topk_indices = scores.topk(k, dim=1)  # [B, k]

        # 5. Sort selected indices chronologically
        selected_indices = topk_indices.sort(dim=1).values  # [B, k]

        # 6. Extract corresponding temporal spans
        selected_times = torch.gather(timestamps, 1, selected_indices)  # [B, k]
        segment_spans: List[List[Tuple[float, float]]] = []
        for b_idx in range(b):
            spans_b = [(float(t), float(t)) for t in selected_times[b_idx].tolist()]
            segment_spans.append(spans_b)

        return {
            "selected_indices": selected_indices,
            "scores": scores,
            "segment_spans": segment_spans,
            "num_frames_selected": k,
        }
