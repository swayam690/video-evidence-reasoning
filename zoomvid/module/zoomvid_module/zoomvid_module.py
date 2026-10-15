"""ZoomVid Full Context Module.

Combines SketchEncoder, Router, and Assembler into a pluggable ContextModule
that dynamically selects, contextualizes, and compresses video frames under
variable token budgets for frozen VLM backbones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from zoomvid.module.interface import ContextModule, ModuleOutput, VideoBatch
from zoomvid.module.zoomvid_module.sketch_encoder import SketchEncoder
from zoomvid.module.zoomvid_module.router import Router
from zoomvid.module.zoomvid_module.assembler import Assembler


class ZoomVidModule(ContextModule):
    """End-to-end trainable ZoomVid video-context module."""

    def __init__(
        self,
        in_features: int = 2048,
        out_features: int = 2048,
        sketch_dim: int = 512,
        question_dim: Optional[int] = None,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize ZoomVidModule.

        Args:
            in_features: Dimension of visual features from patch merger (D).
            out_features: Dimension expected by the frozen LLM backbone (D_llm).
            sketch_dim: Hidden dimension for global sketch representations.
            question_dim: Hidden dimension for question embeddings (if used).
            num_gru_layers: Number of bidirectional GRU layers in SketchEncoder.
            dropout: Dropout probability.
        """
        super().__init__(in_features=in_features, out_features=out_features)
        self.sketch_dim = sketch_dim

        # 1. Sketch Encoder
        self.sketch_encoder = SketchEncoder(
            in_features=in_features,
            sketch_dim=sketch_dim,
            num_layers=num_gru_layers,
            dropout=dropout,
        )

        # 2. Router
        self.router = Router(
            sketch_dim=sketch_dim,
            question_dim=question_dim,
            dropout=dropout,
        )

        # 3. Assembler
        self.assembler = Assembler(
            in_features=in_features,
            out_features=out_features,
            sketch_dim=sketch_dim,
            dropout=dropout,
        )

    def forward(self, batch: VideoBatch) -> ModuleOutput:
        """Run full ZoomVid pipeline on a batch of video features.

        Args:
            batch: VideoBatch containing frame_feats [B, N, P, D], timestamps [B, N],
                   token_budget, and optional question_emb.

        Returns:
            ModuleOutput with tokens [B, T, D_llm] satisfying T <= token_budget,
            and segment_spans for temporal grounding.
        """
        batch.validate()

        frame_feats = batch.frame_feats  # [B, N, P, D]
        timestamps = batch.timestamps    # [B, N]
        b, n, p, d = frame_feats.shape
        budget = batch.token_budget

        # Step 1: Encode coarse global temporal sketch
        sketch_tokens, sketch_aux = self.sketch_encoder(
            frame_feats=frame_feats,
            timestamps=timestamps,
        )

        # Step 2: Route and select key frames under token budget
        routing_out = self.router(
            sketch_tokens=sketch_tokens,
            token_budget=budget,
            timestamps=timestamps,
            patches_per_frame=p,
            question_emb=batch.question_emb,
        )
        selected_indices = routing_out["selected_indices"]  # [B, k]
        segment_spans = routing_out["segment_spans"]

        # Step 3: Assemble selected frame patches with temporal position embeddings
        tokens, assembler_aux = self.assembler(
            frame_feats=frame_feats,
            selected_indices=selected_indices,
            timestamps=timestamps,
            token_budget=budget,
            scores=routing_out["scores"],
            sketch_tokens=sketch_tokens,
        )

        output = ModuleOutput(
            tokens=tokens,
            segment_spans=segment_spans,
            aux={
                "selected_indices": selected_indices,
                "frame_scores": routing_out["scores"],
                "num_frames_selected": routing_out["num_frames_selected"],
                "tokens_produced": tokens.shape[1],
                "token_budget": budget,
                "sketch_aux": sketch_aux,
                "assembler_aux": assembler_aux,
            },
        )
        output.validate(token_budget=budget)
        return output

    def save(self, path: Union[str, Path]) -> None:
        """Save module weights and configuration."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "module_class": self.__class__.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "sketch_dim": self.sketch_dim,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, save_path)

    @classmethod
    def load(cls, path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> ZoomVidModule:
        """Load module from a checkpoint file."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        module = cls(
            in_features=checkpoint["in_features"],
            out_features=checkpoint["out_features"],
            sketch_dim=checkpoint.get("sketch_dim", 512),
        )
        module.load_state_dict(checkpoint["state_dict"])
        module.to(device)
        return module
