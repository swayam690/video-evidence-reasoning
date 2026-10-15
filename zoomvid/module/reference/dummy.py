"""Dummy Linear Context Module for Pipeline Smoke Testing.

A lightweight trainable module with random linear projections used exclusively to test
the end-to-end training harness, backpropagation, and checkpointing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from zoomvid.module.interface import ContextModule, ModuleOutput, VideoBatch


class DummyModule(ContextModule):
    """Tiny trainable linear module for pipeline testing.

    Projects input frame features into the LLM embedding space via a trainable linear
    layer and LayerNorm. Used to verify gradient propagation and loss decrease
    in trainer.py without running heavy models.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pool_patches: bool = True,
    ) -> None:
        """Initialize DummyModule.

        Args:
            in_features: Feature dimension D of input frames.
            out_features: Target feature dimension D_llm for the LLM.
            pool_patches: If True, averages spatial patch dimension P to save sequence length.
        """
        super().__init__(in_features=in_features, out_features=out_features)
        self.pool_patches = pool_patches
        self.proj = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, batch: VideoBatch) -> ModuleOutput:
        """Forward pass through trainable linear projection.

        Args:
            batch: VideoBatch containing frame_feats [B, N, P, D] and token_budget.

        Returns:
            ModuleOutput with tokens [B, T, D_llm] where T <= token_budget.
        """
        batch.validate()

        frame_feats = batch.frame_feats  # [B, N, P, D]
        timestamps = batch.timestamps    # [B, N]
        b, n, p, d = frame_feats.shape
        budget = batch.token_budget

        if self.pool_patches:
            # Pool spatial patches: [B, N, P, D] -> [B, N, D]
            x = frame_feats.mean(dim=2)
            t = min(n, budget)
            x = x[:, :t]
            selected_times = timestamps[:, :t]
        else:
            # Flatten all patches: [B, N * P, D]
            x = frame_feats.reshape(b, n * p, d)
            t = min(n * p, budget)
            x = x[:, :t]
            # Frame timestamps mapped roughly
            selected_times = timestamps

        # Trainable projection: [B, T, D] -> [B, T, D_llm]
        x = x.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        tokens = self.norm(self.proj(x))

        segment_spans: List[List[Tuple[float, float]]] = []
        for b_idx in range(b):
            spans_b = [(float(ts), float(ts)) for ts in selected_times[b_idx].tolist()]
            segment_spans.append(spans_b)

        output = ModuleOutput(
            tokens=tokens,
            segment_spans=segment_spans,
            aux={
                "tokens_produced": tokens.shape[1],
                "in_features": self.in_features,
                "out_features": self.out_features,
            },
        )
        output.validate(token_budget=budget)
        return output

