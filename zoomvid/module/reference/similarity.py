"""Similarity-based Frame Selection Baseline Module.

Selects frames by cosine similarity between each frame's mean-pooled features
and the question embedding, to fit within the token budget.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from zoomvid.module.interface import ContextModule, ModuleOutput, VideoBatch


class SimilarityModule(ContextModule):
    """Question-frame cosine similarity baseline.

    Scores each frame by the cosine similarity between its mean-pooled patch features
    and the question embedding. Selects top-k frames that fit within the token budget,
    sorted chronologically to preserve temporal order.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize SimilarityModule.

        Args:
            in_features: Input feature dimension D (from patch merger).
            out_features: Output feature dimension D_llm (expected by LLM backbone).
        """
        super().__init__(in_features=in_features, out_features=out_features)
        if in_features != out_features:
            self.proj: Optional[nn.Linear] = nn.Linear(in_features, out_features)
        else:
            self.proj = None

    def forward(self, batch: VideoBatch) -> ModuleOutput:
        """Select frames by question-frame similarity to satisfy token budget.

        If no question_emb is provided, falls back to uniform selection.

        Args:
            batch: VideoBatch with frame_feats [B, N, P, D], timestamps [B, N],
                   question_emb [B, D] or [B, L, D], and token_budget.

        Returns:
            ModuleOutput with tokens [B, T, D_llm] where T <= token_budget.
        """
        batch.validate()

        frame_feats = batch.frame_feats  # [B, N, P, D]
        timestamps = batch.timestamps    # [B, N]
        b, n, p, d = frame_feats.shape
        budget = batch.token_budget

        max_k_by_budget = max(1, budget // p)
        k = min(n, max_k_by_budget)

        # Mean-pool patches per frame: [B, N, D]
        frame_means = frame_feats.mean(dim=2)

        if batch.question_emb is not None:
            q_emb = batch.question_emb
            # If question_emb has sequence dim [B, L, D], mean-pool to [B, D]
            if q_emb.ndim == 3:
                q_emb = q_emb.mean(dim=1)
            # q_emb: [B, D]

            # Cosine similarity: [B, N]
            frame_means_norm = F.normalize(frame_means, dim=-1)
            q_emb_norm = F.normalize(q_emb, dim=-1).unsqueeze(1)  # [B, 1, D]
            similarities = (frame_means_norm * q_emb_norm).sum(dim=-1)  # [B, N]
        else:
            # No question embedding: uniform fallback (use position as score)
            similarities = torch.linspace(0, 1, n, device=frame_feats.device).unsqueeze(0).expand(b, -1)

        # Select top-k by similarity
        _, topk_indices = similarities.topk(k, dim=1)  # [B, k]

        # Sort selected indices chronologically to preserve temporal order
        topk_indices, _ = topk_indices.sort(dim=1)

        # Gather selected frames and timestamps
        # Expand topk_indices for gathering: [B, k, 1, 1] -> broadcast over P and D
        gather_idx = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(b, k, p, d)
        selected_feats = torch.gather(frame_feats, 1, gather_idx)  # [B, k, P, D]

        time_idx = topk_indices  # [B, k]
        selected_times = torch.gather(timestamps, 1, time_idx)  # [B, k]

        # Flatten: [B, k, P, D] -> [B, k*P, D]
        tokens = selected_feats.reshape(b, k * p, d)

        if self.proj is not None:
            tokens = tokens.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
            tokens = self.proj(tokens)

        segment_spans: List[List[Tuple[float, float]]] = []
        for b_idx in range(b):
            spans_b = [(float(t), float(t)) for t in selected_times[b_idx].tolist()]
            segment_spans.append(spans_b)

        output = ModuleOutput(
            tokens=tokens,
            segment_spans=segment_spans,
            aux={
                "selected_indices": topk_indices,
                "similarities": similarities,
                "num_frames_selected": k,
                "tokens_produced": tokens.shape[1],
                "patches_per_frame": p,
            },
        )
        output.validate(token_budget=budget)
        return output

