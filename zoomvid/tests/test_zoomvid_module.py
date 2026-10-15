"""Unit tests for the ZoomVid architecture (SketchEncoder, Router, Assembler, ZoomVidModule)."""

import tempfile
from pathlib import Path
import pytest
import torch

from zoomvid.module.interface import VideoBatch
from zoomvid.module.zoomvid_module import Assembler, Router, SketchEncoder, ZoomVidModule


def test_sketch_encoder_shape() -> None:
    """Test that SketchEncoder produces temporally contextualized representations."""
    b, n, p, d = 2, 8, 144, 2048
    sketch_dim = 256
    encoder = SketchEncoder(in_features=d, sketch_dim=sketch_dim)

    frame_feats = torch.randn(b, n, p, d)
    timestamps = torch.linspace(0.0, 7.0, n).unsqueeze(0).expand(b, -1)

    sketch_tokens, aux = encoder(frame_feats, timestamps)

    assert sketch_tokens.shape == (b, n, sketch_dim)
    assert not torch.isnan(sketch_tokens).any()
    assert "spatial_summary" in aux


def test_router_selection_and_chronology() -> None:
    """Test that Router selects top-k frames within budget and orders them chronologically."""
    b, n, sketch_dim = 2, 10, 256
    router = Router(sketch_dim=sketch_dim)

    sketch_tokens = torch.randn(b, n, sketch_dim)
    timestamps = torch.linspace(0.0, 9.0, n).unsqueeze(0).expand(b, -1)
    budget = 432  # 432 // 144 = 3 frames max

    routing_out = router(
        sketch_tokens=sketch_tokens,
        token_budget=budget,
        timestamps=timestamps,
        patches_per_frame=144,
    )

    indices = routing_out["selected_indices"]
    assert indices.shape == (b, 3)

    # Verify chronological ordering
    for b_idx in range(b):
        idx_list = indices[b_idx].tolist()
        assert idx_list == sorted(idx_list), f"Indices not sorted: {idx_list}"
        assert len(idx_list) == len(set(idx_list)), f"Duplicate indices: {idx_list}"


def test_assembler_budget_and_shape() -> None:
    """Test that Assembler gathers patches, injects temporal encodings, and respects budget."""
    b, n, p, d = 1, 6, 144, 2048
    out_dim = 2048
    assembler = Assembler(in_features=d, out_features=out_dim)

    frame_feats = torch.randn(b, n, p, d)
    selected_indices = torch.tensor([[1, 3, 5]])  # 3 frames -> 3 * 144 = 432 tokens
    timestamps = torch.linspace(0.0, 5.0, n).unsqueeze(0)
    budget = 500

    tokens, aux = assembler(
        frame_feats=frame_feats,
        selected_indices=selected_indices,
        timestamps=timestamps,
        token_budget=budget,
    )

    assert tokens.shape == (b, 432, out_dim)
    assert not torch.isnan(tokens).any()
    assert tokens.shape[1] <= budget


def test_zoomvid_module_full_pipeline_and_gradients() -> None:
    """Test end-to-end forward pass, backpropagation, and save/load on ZoomVidModule."""
    b, n, p, d = 2, 6, 144, 2048
    module = ZoomVidModule(in_features=d, out_features=d, sketch_dim=256)

    frame_feats = torch.randn(b, n, p, d)
    timestamps = torch.linspace(0.0, 5.0, n).unsqueeze(0).expand(b, -1)
    batch = VideoBatch(
        frame_feats=frame_feats,
        timestamps=timestamps,
        mode="qa",
        token_budget=432,  # 3 frames
    )

    # Forward pass
    output = module(batch)
    output.validate(token_budget=432)

    assert output.tokens.shape == (b, 432, d)
    assert not torch.isnan(output.tokens).any()
    assert len(output.segment_spans) == b
    assert len(output.segment_spans[0]) == 3

    # Backpropagation: loss computes gradients across all module sub-networks
    loss = output.tokens.sum()
    loss.backward()

    # Verify gradients exist in each component
    has_sketch_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.sketch_encoder.parameters())
    has_assembler_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.assembler.parameters())
    assert has_sketch_grad, "SketchEncoder did not receive gradients!"
    assert has_assembler_grad, "Assembler did not receive gradients!"

    # Save and Load test
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "zoomvid_test.pt"
        module.save(save_path)
        assert save_path.exists()

        module.eval()
        loaded_module = ZoomVidModule.load(save_path)
        loaded_module.eval()
        with torch.no_grad():
            orig_out = module(batch)
            loaded_out = loaded_module(batch)
            diff = torch.abs(orig_out.tokens - loaded_out.tokens).max().item()
            assert diff < 1e-5, f"Saved and loaded outputs differ: {diff}"
