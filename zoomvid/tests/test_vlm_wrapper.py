"""Local unit tests for FrozenVLMBackbone logic and helper utilities."""

import pytest
import torch
import torch.nn as nn
from typing import Any

from zoomvid.core.helpers import (
    calculate_budget_allocation,
    compute_frame_token_count,
    format_timestamp,
    format_timestamp_span,
    verify_chronological_order,
)
from zoomvid.core.vlm_wrapper import parse_dtype


def test_parse_dtype() -> None:
    """Test dtype parsing for fp16, bf16, and fp32."""
    assert parse_dtype("fp16") == torch.float16
    assert parse_dtype("float16") == torch.float16
    assert parse_dtype("bf16") == torch.bfloat16
    assert parse_dtype("bfloat16") == torch.bfloat16
    assert parse_dtype("fp32") == torch.float32

    with pytest.raises(ValueError, match="Unsupported dtype"):
        parse_dtype("int8")


def test_helpers_formatting_and_budget() -> None:
    """Test helper functions for timestamp tags and token counting."""
    assert format_timestamp(4.25) == "<|time_4.2s|>"
    assert format_timestamp_span(1.0, 3.5) == "<|span_1.0s_to_3.5s|>"

    # 336x336 with 14x14 patches and 2x2 merger -> (24/2)*(24/2) = 144 tokens per frame
    tokens_per_frame = compute_frame_token_count(336, 336, spatial_patch_size=14, spatial_merge_size=2)
    assert tokens_per_frame == 144

    # Budget 500 tokens with 144 tokens/frame -> max 3 frames = 432 tokens
    frames, used_tokens = calculate_budget_allocation(500, num_frames=10, tokens_per_frame=tokens_per_frame)
    assert frames == 3
    assert used_tokens == 432

    # Chronological order verification
    assert verify_chronological_order([0.0, 1.0, 2.0, 2.5])
    assert not verify_chronological_order([0.0, 2.0, 1.0])
    assert verify_chronological_order(torch.tensor([[0.0, 1.0], [0.5, 1.5]]))
    assert not verify_chronological_order(torch.tensor([[0.0, 2.0], [1.5, 0.5]]))


class MockVLMModel(nn.Module):
    """Tiny mock model for local testing of freeze/unfreeze logic."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(50, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, 50, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(self, inputs_embeds: torch.Tensor, labels: torch.Tensor | None = None, **kwargs) -> Any:
        h = self.linear(inputs_embeds)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, 50), shift_labels.view(-1))

        class MockOutput:
            def __init__(self, logits, loss):
                self.logits = logits
                self.loss = loss

        return MockOutput(logits=logits, loss=loss)


def test_frozen_assertion_on_unfrozen_parameters() -> None:
    """Test that FrozenVLMBackbone assert_frozen detects unfrozen parameters."""
    from zoomvid.core.vlm_wrapper import FrozenVLMBackbone

    wrapper = FrozenVLMBackbone.__new__(FrozenVLMBackbone)
    nn.Module.__init__(wrapper)
    wrapper.model = MockVLMModel(hidden_size=16)

    # Initially parameters require grad
    with pytest.raises(RuntimeError, match="Backbone is NOT completely frozen"):
        wrapper.assert_frozen()

    # After freeze(), assertion must pass
    wrapper.freeze()
    wrapper.assert_frozen()
    for p in wrapper.model.parameters():
        assert not p.requires_grad
