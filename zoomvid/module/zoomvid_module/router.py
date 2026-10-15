"""ZoomVid Router (Stub for Colab Implementation).

=============================================================================
INTENDED PURPOSE:
The Router acts as the temporal spotlight / policy network. Given the coarse
sketch tokens and (optionally) the query embedding, it computes importance
scores over temporal regions and decides which frames or temporal segments to
"zoom in" on, dynamically allocating the available token_budget.

INTENDED INPUTS:
----------------
- sketch_tokens: torch.Tensor of shape [B, N_sketch, D_sketch]
    Output from the SketchEncoder.
- question_emb: Optional[torch.Tensor] of shape [B, D_q]
    Embedding of the question/instruction.
- token_budget: int
    Total visual token budget allowed for the LLM.
- timestamps: torch.Tensor of shape [B, N]
    Original frame timestamps in seconds.

INTENDED OUTPUTS:
-----------------
- routing_weights: torch.Tensor of shape [B, N] or [B, N_sketch]
    Normalized frame importance scores / probabilities.
- selected_indices: torch.Tensor or List[List[int]]
    Indices of high-resolution frames selected for detailed patch extraction.
- segment_spans: List[List[Tuple[float, float]]]
    List of selected (start_seconds, end_seconds) temporal intervals per batch item.
- aux: Dict[str, Any]
    Routing entropy loss, sparsity regularization terms, or selection statistics.

SUGGESTED COLAB IMPLEMENTATION STEPS:
1. Compute relevance logits via MLP(sketch_tokens) or cross-attention with question_emb.
2. Select top-k frames or apply Gumbel-Softmax / differentiable top-k if training end-to-end.
3. Compute segment spans from contiguous selected frames.
=============================================================================
"""

import torch
import torch.nn as nn


class Router(nn.Module):
    """Stub class for ZoomVid Router."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # To be implemented on Google Colab

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "Router is a stub. Implement the forward pass in Colab as described in the docstring."
        )

