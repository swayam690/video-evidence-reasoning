"""Uniform Frame Sampling Baseline Module.

Selects frames at uniform temporal intervals to fit within the specified token budget.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from zoomvid.module.interface import ContextModule, ModuleOutput, VideoBatch


class UniformModule(ContextModule):
    """Uniform frame selection baseline.

    Given a video with N frames each having P patch tokens, determines the maximum
    number of frames k = min(N, max(1, token_budget // P)) that fit within the token
    budget, samples them uniformly across time, and flattens them into a sequence of
    shape [B, k * P, D_llm].
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize UniformModule.

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
        """Select frames uniformly to satisfy the token budget.

        Args:
            batch: VideoBatch containing frame_feats [B, N, P, D], timestamps [B, N],
                   and token_budget.

        Returns:
            ModuleOutput with tokens [B, T, D_llm] where T <= token_budget.
        """
        batch.validate()

        frame_feats = batch.frame_feats  # [B, N, P, D]
        timestamps = batch.timestamps    # [B, N]
        b, n, p, d = frame_feats.shape
        budget = batch.token_budget

        # Each frame consumes P patch tokens.
        # Determine maximum frames k that fit in budget:
        max_k_by_budget = max(1, budget // p)
        k = min(n, max_k_by_budget)

        if k >= n:
            # Keep all frames
            indices = torch.arange(n, device=frame_feats.device)
            selected_feats = frame_feats
            selected_times = timestamps
        else:
            # Uniformly sample k frame indices
            indices = torch.linspace(0, n - 1, steps=k, device=frame_feats.device).round().long()
            selected_feats = frame_feats[:, indices]  # [B, k, P, D]
            selected_times = timestamps[:, indices]    # [B, k]

        # Flatten frame patches: [B, k, P, D] -> [B, k * P, D]
        tokens = selected_feats.reshape(b, k * p, d)

        # Project to D_llm if necessary
        if self.proj is not None:
            tokens = self.proj(tokens)

        # Build temporal spans for selected frames: [(t, t)]
        segment_spans: List[List[Tuple[float, float]]] = []
        for b_idx in range(b):
            spans_b = [(float(t), float(t)) for t in selected_times[b_idx].tolist()]
            segment_spans.append(spans_b)

        output = ModuleOutput(
            tokens=tokens,
            segment_spans=segment_spans,
            aux={
                "selected_indices": indices,
                "num_frames_selected": k,
                "tokens_produced": tokens.shape[1],
                "patches_per_frame": p,
            },
        )
        output.validate(token_budget=budget)
        return output

