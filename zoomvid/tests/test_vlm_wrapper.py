"""Local unit tests for FrozenVLMBackbone logic and helper utilities."""

import pytest
import torch
import torch.nn as nn
from typing import Any
from transformers import PretrainedConfig

from zoomvid.core.helpers import (
    calculate_budget_allocation,
    compute_frame_token_count,
    format_timestamp,
    format_timestamp_span,
    verify_chronological_order,
)
from zoomvid.core.vlm_wrapper import FrozenVLMBackbone, parse_dtype


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


class MockTokenizer:
    """Lightweight mock tokenizer for testing embedding assembly without Hub downloads."""

    def encode(self, text: str, add_special_tokens: bool = False, return_tensors: str = "pt") -> torch.Tensor:
        # Simple ASCII tokenization for mock testing
        tokens = [ord(c) % 50 for c in text] or [1]
        t = torch.tensor([tokens], dtype=torch.long)
        return t

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "mock answer"


class MockProcessor:
    """Lightweight mock processor."""

    def __init__(self) -> None:
        self.tokenizer = MockTokenizer()


class MockVLMModel(nn.Module):
    """Tiny mock model representing Qwen2.5-VL for injection testing."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(50, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, 50, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> Any:
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


def test_injection_sequence_assembly() -> None:
    """Test assemble_injection_sequence dimensions, ordering, and label masking."""
    hidden_size = 32
    wrapper = FrozenVLMBackbone.__new__(FrozenVLMBackbone)
    nn.Module.__init__(wrapper)
    wrapper.model = MockVLMModel(hidden_size=hidden_size)
    wrapper.freeze()
    wrapper.processor = MockProcessor()
    wrapper.llm_hidden_size = hidden_size
    wrapper.target_dtype = torch.float32

    batch_size = 2
    module_token_count = 10
    module_tokens = torch.randn(batch_size, module_token_count, hidden_size)

    prefix_text = "Prompt:"
    suffix_text = "What is this?"
    target_text = "A cat."

    full_embeds, labels, prompt_len, target_len = wrapper.assemble_injection_sequence(
        module_tokens=module_tokens,
        prefix_text=prefix_text,
        suffix_text=suffix_text,
        target_text=target_text,
    )

    prefix_len = len(wrapper.processor.tokenizer.encode(prefix_text)[0])
    suffix_len = len(wrapper.processor.tokenizer.encode(suffix_text)[0])
    expected_prompt_len = prefix_len + module_token_count + suffix_len
    expected_target_len = len(wrapper.processor.tokenizer.encode(target_text)[0])

    assert prompt_len == expected_prompt_len
    assert target_len == expected_target_len
    assert full_embeds.shape == (batch_size, prompt_len + target_len, hidden_size)

    # Assert labels mask prompt with -100
    assert (labels[:, :prompt_len] == -100).all()
    # Assert labels contain valid token ids for target
    assert (labels[:, prompt_len:] != -100).all()


def test_injection_path_numerical_equivalence() -> None:
    """Test that feeding visual embeddings through assemble_injection_sequence

    yields the exact same output as passing them as a pre-concatenated input.
    """
    hidden_size = 32
    wrapper = FrozenVLMBackbone.__new__(FrozenVLMBackbone)
    nn.Module.__init__(wrapper)
    wrapper.model = MockVLMModel(hidden_size=hidden_size)
    wrapper.freeze()
    wrapper.processor = MockProcessor()
    wrapper.llm_hidden_size = hidden_size
    wrapper.target_dtype = torch.float32

    b, t = 1, 8
    visual_tokens = torch.randn(b, t, hidden_size)

    # Forward via wrapper's forward_with_injected_tokens
    out1 = wrapper.forward_with_injected_tokens(
        module_tokens=visual_tokens,
        prefix_text="Prefix",
        suffix_text="Suffix",
        target_text="Answer",
    )

    # Reconstruct the exact inputs manually
    full_embeds, labels, _, _ = wrapper.assemble_injection_sequence(
        module_tokens=visual_tokens,
        prefix_text="Prefix",
        suffix_text="Suffix",
        target_text="Answer",
    )
    out2 = wrapper.model(inputs_embeds=full_embeds, labels=labels)

    assert torch.allclose(out1.logits, out2.logits, atol=1e-5)
    assert torch.allclose(out1.loss, out2.loss, atol=1e-5)
