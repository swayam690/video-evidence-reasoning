"""ZoomVid Sketch Encoder (Stub for Colab Implementation).

=============================================================================
INTENDED PURPOSE:
The Sketch Encoder compresses the dense, fine-grained video features into a
compact, temporally coherent "sketch" representation. This coarse summary is
then passed to the Router to identify key events, timestamps, and regions of
interest without consuming the full token budget.

INTENDED INPUTS:
----------------
- frame_feats: torch.Tensor of shape [B, N, P, D]
    B: batch size
    N: number of input frames
    P: patches per frame from the frozen vision encoder (e.g. 14x14 or spatial merger)
    D: vision feature dimension (e.g. 2048 or 1024)
- timestamps: torch.Tensor of shape [B, N]
    Denoting frame timestamp in seconds.
- question_emb: Optional[torch.Tensor] of shape [B, D_q] or [B, L_q, D_q]
    Embedding of the input prompt/question for query-conditioned sketch encoding.

INTENDED OUTPUTS:
-----------------
- sketch_tokens: torch.Tensor of shape [B, N_sketch, D_sketch]
    Coarse temporal representations (e.g., temporally pooled frames, cross-frame
    attention tokens, or 1D temporal convolution summaries).
- sketch_timestamps: torch.Tensor of shape [B, N_sketch]
    Timestamps corresponding to the coarse sketch representations.
- aux: Dict[str, Any]
    Intermediate activations or attention maps for auxiliary loss/logging.

SUGGESTED COLAB IMPLEMENTATION STEPS:
1. Spatial pooling: average pool across patch dimension P to obtain [B, N, D].
2. Temporal modeling: apply a small 1D Conv, lightweight bidirectional GRU, or
   1-2 layer Transformer encoder across the temporal dimension N.
3. (Optional) Cross-attention with `question_emb` to guide coarse focus.
=============================================================================
"""

import torch
import torch.nn as nn


class SketchEncoder(nn.Module):
    """Stub class for ZoomVid Sketch Encoder."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # To be implemented on Google Colab

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "SketchEncoder is a stub. Implement the forward pass in Colab as described in the docstring."
        )

