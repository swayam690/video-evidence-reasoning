"""ZoomVid Pluggable Context Module Interface Contract.

Defines the core data structures and abstract class that any video-context
module must adhere to in order to interface with the frozen VLM backbone.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class VideoBatch:
    """Input batch representation passed to any ContextModule.

    Attributes:
        frame_feats: Tensor of shape [B, N, P, D] where:
            - B: batch size
            - N: number of sampled frames
            - P: number of spatial/patch tokens per frame
            - D: feature dimension from vision encoder / patch merger
        timestamps: Tensor of shape [B, N] denoting timestamp in seconds for each frame.
        question_emb: Optional text embedding tensor representing the query or prompt,
            e.g. shape [B, D_text] or [B, L_q, D_text].
        mode: Processing mode, either "qa" or "narrate".
        token_budget: Maximum number of output visual tokens T to inject into LLM.
    """

    frame_feats: torch.Tensor
    timestamps: torch.Tensor
    question_emb: Optional[torch.Tensor] = None
    mode: str = "qa"
    token_budget: int = 2048

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate tensor shapes, dimensions, and chronological consistency."""
        if not isinstance(self.frame_feats, torch.Tensor):
            raise TypeError(f"frame_feats must be torch.Tensor, got {type(self.frame_feats)}")
        if not isinstance(self.timestamps, torch.Tensor):
            raise TypeError(f"timestamps must be torch.Tensor, got {type(self.timestamps)}")

        if self.frame_feats.ndim != 4:
            raise ValueError(
                f"frame_feats must be 4D [B, N, P, D], got shape {list(self.frame_feats.shape)}"
            )
        if self.timestamps.ndim != 2:
            raise ValueError(
                f"timestamps must be 2D [B, N], got shape {list(self.timestamps.shape)}"
            )

        b_feat, n_feat = self.frame_feats.shape[0], self.frame_feats.shape[1]
        b_time, n_time = self.timestamps.shape[0], self.timestamps.shape[1]

        if (b_feat, n_feat) != (b_time, n_time):
            raise ValueError(
                f"Batch size and frame count mismatch between frame_feats ({b_feat}, {n_feat}) "
                f"and timestamps ({b_time}, {n_time})"
            )

        if self.mode not in ("qa", "narrate"):
            raise ValueError(f"mode must be 'qa' or 'narrate', got '{self.mode}'")

        if self.token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {self.token_budget}")

        # Check chronological ordering: timestamps must be non-decreasing along frame dimension N
        if n_time > 1:
            diffs = self.timestamps[:, 1:] - self.timestamps[:, :-1]
            if (diffs < -1e-4).any():
                raise ValueError(
                    "timestamps must be in non-decreasing chronological order along dimension 1"
                )

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> VideoBatch:
        """Move batch tensors to target device and optional dtype."""
        frame_feats = self.frame_feats.to(device=device, dtype=dtype)
        timestamps = self.timestamps.to(device=device)
        question_emb = (
            self.question_emb.to(device=device, dtype=dtype)
            if self.question_emb is not None
            else None
        )
        return VideoBatch(
            frame_feats=frame_feats,
            timestamps=timestamps,
            question_emb=question_emb,
            mode=self.mode,
            token_budget=self.token_budget,
        )


@dataclass
class ModuleOutput:
    """Output representation returned by any ContextModule.

    Attributes:
        tokens: Injected visual tokens of shape [B, T, D_llm], where T <= token_budget.
        segment_spans: Optional list of temporal spans selected/focused on by the module,
            e.g. list per batch element: [[(start_s, end_s), ...], ...].
        aux: Dictionary storing auxiliary outputs, intermediate router scores,
            attention maps, or extra loss terms for training.
    """

    tokens: torch.Tensor
    segment_spans: Optional[List[List[Tuple[float, float]]]] = None
    aux: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self, token_budget: Optional[int] = None) -> None:
        """Validate output tensor dimensions and token budget compliance."""
        if not isinstance(self.tokens, torch.Tensor):
            raise TypeError(f"tokens must be torch.Tensor, got {type(self.tokens)}")
        if self.tokens.ndim != 3:
            raise ValueError(
                f"tokens must be 3D [B, T, D_llm], got shape {list(self.tokens.shape)}"
            )
        if token_budget is not None and self.tokens.shape[1] > token_budget:
            raise ValueError(
                f"tokens length T={self.tokens.shape[1]} exceeds token_budget={token_budget}"
            )

    def to(self, device: Union[str, torch.device]) -> ModuleOutput:
        """Move output tokens and auxiliary tensors to target device."""
        tokens = self.tokens.to(device)
        aux_moved = {}
        for k, v in self.aux.items():
            if isinstance(v, torch.Tensor):
                aux_moved[k] = v.to(device)
            else:
                aux_moved[k] = v
        return ModuleOutput(
            tokens=tokens,
            segment_spans=self.segment_spans,
            aux=aux_moved,
        )


class ContextModule(nn.Module, abc.ABC):
    """Abstract base class for all pluggable video-context modules.

    A ContextModule sits between the frozen vision encoder (or feature cache)
    and the frozen LLM. It receives a VideoBatch and outputs a ModuleOutput
    containing tokens that match the LLM hidden size.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the context module.

        Args:
            in_features: Feature dimension D of input frame features (from vision encoder/patch merger).
            out_features: Feature dimension D_llm of the LLM backbone hidden state.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    @abc.abstractmethod
    def forward(self, batch: VideoBatch) -> ModuleOutput:
        """Process a VideoBatch and produce ModuleOutput for the frozen LLM.

        Args:
            batch: VideoBatch containing frame features, timestamps, and budget.

        Returns:
            ModuleOutput containing tokens [B, T, D_llm] with T <= batch.token_budget.
        """
        raise NotImplementedError

    def save(self, path: Union[str, Path]) -> None:
        """Save module weights and configuration metadata to disk.

        Args:
            path: Destination file path (e.g. checkpoint.pt).
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "module_class": self.__class__.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, save_path)

    def load(self, path: Union[str, Path]) -> None:
        """Load module weights from disk.

        Args:
            path: Checkpoint file path.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {load_path}")
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint["state_dict"])

