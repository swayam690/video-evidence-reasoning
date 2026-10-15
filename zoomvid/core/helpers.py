"""Helper utilities for timestamp formatting, token budgets, and temporal metadata."""

from __future__ import annotations

from typing import List, Tuple, Union
import torch


def format_timestamp(seconds: float) -> str:
    """Format seconds into human-readable timestamp tag, e.g. 12.5s -> '<|time_12.5s|>'."""
    return f"<|time_{seconds:.1f}s|>"


def format_timestamp_span(start_s: float, end_s: float) -> str:
    """Format a temporal span into a tag, e.g. (1.2, 3.4) -> '<|span_1.2s_to_3.4s|>'."""
    return f"<|span_{start_s:.1f}s_to_{end_s:.1f}s|>"


def compute_frame_token_count(
    height: int,
    width: int,
    spatial_patch_size: int = 14,
    spatial_merge_size: int = 2,
) -> int:
    """Compute number of tokens produced by a single frame after patch merging.

    Args:
        height: Image/frame height in pixels (e.g. 336).
        width: Image/frame width in pixels (e.g. 336).
        spatial_patch_size: Vision transformer patch size (default 14).
        spatial_merge_size: 2D spatial merger factor (default 2 for 2x2 merger).

    Returns:
        Number of patch tokens per frame.
    """
    h_patches = height // spatial_patch_size
    w_patches = width // spatial_patch_size
    h_merged = h_patches // spatial_merge_size
    w_merged = w_patches // spatial_merge_size
    return h_merged * w_merged


def calculate_budget_allocation(
    total_budget: int,
    num_frames: int,
    tokens_per_frame: int,
) -> Tuple[int, int]:
    """Calculate the maximum number of frames that fit into a given token budget.

    Args:
        total_budget: Total allowed visual tokens.
        num_frames: Total available video frames.
        tokens_per_frame: Number of tokens each frame generates.

    Returns:
        (frames_to_keep, actual_tokens_used)
    """
    max_frames_fit = max(1, total_budget // tokens_per_frame)
    frames_to_keep = min(num_frames, max_frames_fit)
    actual_tokens = frames_to_keep * tokens_per_frame
    return frames_to_keep, actual_tokens


def verify_chronological_order(timestamps: Union[torch.Tensor, List[float]]) -> bool:
    """Check if timestamps are strictly non-decreasing."""
    if isinstance(timestamps, torch.Tensor):
        if timestamps.ndim == 1:
            return bool((timestamps[1:] >= timestamps[:-1] - 1e-4).all().item())
        elif timestamps.ndim == 2:
            return bool((timestamps[:, 1:] >= timestamps[:, :-1] - 1e-4).all().item())
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got shape {list(timestamps.shape)}")
    return all(b >= a - 1e-4 for a, b in zip(timestamps[:-1], timestamps[1:]))

