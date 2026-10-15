"""Unit tests for ZoomVid Module Interface and Reference Baselines."""

import tempfile
from pathlib import Path
import pytest
import torch

from zoomvid.module.interface import VideoBatch, ModuleOutput, ContextModule
from zoomvid.module.reference.uniform import UniformModule
from zoomvid.module.reference.dummy import DummyModule


def test_video_batch_valid() -> None:
    """Test valid VideoBatch creation and shape checking."""
    b, n, p, d = 2, 8, 16, 2048
    frame_feats = torch.randn(b, n, p, d)
    timestamps = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                               [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]])

    batch = VideoBatch(
        frame_feats=frame_feats,
        timestamps=timestamps,
        mode="qa",
        token_budget=1024,
    )
    assert batch.frame_feats.shape == (b, n, p, d)
    assert batch.timestamps.shape == (b, n)
    assert batch.token_budget == 1024


def test_video_batch_chronological_ordering() -> None:
    """Test that non-chronological timestamps raise a ValueError."""
    b, n, p, d = 1, 4, 4, 128
    frame_feats = torch.randn(b, n, p, d)
    # Timestamps going backwards: 0.0 -> 2.0 -> 1.0 -> 3.0
    bad_timestamps = torch.tensor([[0.0, 2.0, 1.0, 3.0]])

    with pytest.raises(ValueError, match="chronological"):
        VideoBatch(frame_feats=frame_feats, timestamps=bad_timestamps)


def test_video_batch_shape_mismatches() -> None:
    """Test error handling for mismatched dimensions."""
    frame_feats = torch.randn(2, 4, 16, 128)

    # Wrong batch size
    bad_timestamps_b = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="mismatch"):
        VideoBatch(frame_feats=frame_feats, timestamps=bad_timestamps_b)

    # Wrong frame count
    bad_timestamps_n = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="mismatch"):
        VideoBatch(frame_feats=frame_feats, timestamps=bad_timestamps_n)


def test_uniform_module_budget_enforcement() -> None:
    """Test UniformModule limits output tokens to token_budget and preserves order."""
    b, n, p, d = 2, 10, 16, 512
    out_dim = 1024
    frame_feats = torch.randn(b, n, p, d)
    timestamps = torch.linspace(0, 9, steps=n).unsqueeze(0).repeat(b, 1)

    # Budget allows for only 4 frames (4 * 16 = 64 tokens)
    budget = 70
    batch = VideoBatch(
        frame_feats=frame_feats,
        timestamps=timestamps,
        token_budget=budget,
    )

    module = UniformModule(in_features=d, out_features=out_dim)
    output = module(batch)

    assert output.tokens.ndim == 3
    assert output.tokens.shape[0] == b
    assert output.tokens.shape[1] <= budget
    assert output.tokens.shape[2] == out_dim
    assert output.aux["num_frames_selected"] == budget // p
    assert output.tokens.shape[1] == (budget // p) * p


def test_uniform_module_full_frames() -> None:
    """Test UniformModule when budget exceeds total video tokens."""
    b, n, p, d = 1, 5, 8, 256
    frame_feats = torch.randn(b, n, p, d)
    timestamps = torch.linspace(0, 4, steps=n).unsqueeze(0)

    # Total tokens = 5 * 8 = 40; budget is 100
    batch = VideoBatch(
        frame_feats=frame_feats,
        timestamps=timestamps,
        token_budget=100,
    )

    module = UniformModule(in_features=d, out_features=d)
    output = module(batch)

    assert output.tokens.shape == (b, 40, d)
    assert output.aux["num_frames_selected"] == 5


def test_dummy_module_gradients_and_save_load() -> None:
    """Test DummyModule backprop and save/load serialization."""
    b, n, p, d = 2, 6, 8, 128
    out_dim = 256
    frame_feats = torch.randn(b, n, p, d, requires_grad=True)
    timestamps = torch.linspace(0, 5, steps=n).unsqueeze(0).repeat(b, 1)

    batch = VideoBatch(
        frame_feats=frame_feats,
        timestamps=timestamps,
        token_budget=32,
    )

    module = DummyModule(in_features=d, out_features=out_dim)
    output = module(batch)

    # Test output dimensions
    assert output.tokens.shape == (b, min(n, 32), out_dim)

    # Test backpropagation
    loss = output.tokens.sum()
    loss.backward()
    assert module.proj.weight.grad is not None
    assert module.norm.weight.grad is not None

    # Test save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "dummy.pt"
        module.save(ckpt_path)

        new_module = DummyModule(in_features=d, out_features=out_dim)
        new_module.load(ckpt_path)

        # Outputs must match
        with torch.no_grad():
            out_orig = module(batch).tokens
            out_new = new_module(batch).tokens
            assert torch.allclose(out_orig, out_new)

