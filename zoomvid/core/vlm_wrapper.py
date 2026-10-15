"""Frozen Vision-Language Model Backbone Wrapper for ZoomVid.

Wraps Qwen/Qwen2.5-VL-3B-Instruct in a strictly frozen configuration (requires_grad=False).
Provides utilities for per-frame vision feature extraction, custom embedding injection,
answer generation, and log-probability scoring of target answers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse string dtype representation to torch.dtype."""
    normalized = dtype_str.lower().strip()
    if normalized in ("fp16", "float16"):
        return torch.float16
    elif normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    elif normalized in ("fp32", "float32"):
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype string '{dtype_str}'. Choose from fp16, bf16, fp32.")


class FrozenVLMBackbone(nn.Module):
    """Frozen VLM Backbone wrapping Qwen2.5-VL models.

    Ensures all parameters are frozen and provides modular entry points for
    extracting vision tokens and injecting custom module representations
    into the LLM embedding stream.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype: str = "fp16",
        device: Optional[Union[str, torch.device]] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        attn_implementation: Optional[str] = None,
        load_pretrained: bool = True,
        config: Optional[Any] = None,
    ) -> None:
        """Initialize FrozenVLMBackbone.

        Args:
            model_id: HuggingFace model identifier.
            torch_dtype: Model precision flag ('fp16' default for Colab T4, or 'bf16').
            device: Target device (e.g. 'cuda' or 'cpu').
            device_map: Accelerate device map ('auto', None, etc.).
            attn_implementation: Attention backend ('sdpa', 'flash_attention_2', etc.).
            load_pretrained: If True, downloads weights from HuggingFace. If False,
                             initializes from config (used for lightweight testing).
            config: Optional pre-constructed config object.
        """
        super().__init__()
        self.model_id = model_id
        self.target_dtype = parse_dtype(torch_dtype)

        if config is None:
            self.config = AutoConfig.from_pretrained(model_id)
        else:
            self.config = config

        # Dynamically read hidden dimensions from model config (never hardcode)
        text_cfg = getattr(self.config, "text_config", self.config)
        self.llm_hidden_size: int = getattr(text_cfg, "hidden_size", 2048)

        vision_cfg = getattr(self.config, "vision_config", None)
        if vision_cfg is not None:
            self.vision_feature_size: int = getattr(
                vision_cfg, "out_hidden_size", self.llm_hidden_size
            )
            self.spatial_patch_size: int = getattr(vision_cfg, "spatial_patch_size", 14)
            self.spatial_merge_size: int = getattr(vision_cfg, "spatial_merge_size", 2)
        else:
            self.vision_feature_size = self.llm_hidden_size
            self.spatial_patch_size = 14
            self.spatial_merge_size = 2

        if load_pretrained:
            logger.info("Loading pretrained model %s with dtype %s", model_id, self.target_dtype)
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": self.target_dtype,
            }
            if device_map is not None:
                model_kwargs["device_map"] = device_map
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, **model_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            self.model = Qwen2_5_VLForConditionalGeneration(self.config)
            self.processor = None

        if device is not None and device_map is None:
            self.model.to(device)

        # CRITICAL: Freeze all VLM parameters
        self.freeze()
        self.assert_frozen()

    def freeze(self) -> None:
        """Strictly freeze all backbone parameters and set eval mode."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def assert_frozen(self) -> None:
        """Assert that every parameter in the backbone has requires_grad=False."""
        trainable = [name for name, p in self.model.named_parameters() if p.requires_grad]
        if trainable:
            raise RuntimeError(
                f"Backbone is NOT completely frozen! Trainable parameters found: {trainable[:5]}"
            )

    @property
    def visual(self) -> nn.Module:
        """Return the vision encoder module."""
        if hasattr(self.model, "visual"):
            return self.model.visual
        elif hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            return self.model.model.visual
        raise AttributeError("Could not locate visual encoder in Qwen2.5-VL model hierarchy.")

    @property
    def embedding_layer(self) -> nn.Module:
        """Return the input text embedding layer."""
        return self.model.get_input_embeddings()

    @property
    def device(self) -> torch.device:
        """Return model device."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return model parameter dtype."""
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def extract_vision_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-frame vision features post-patch-merger.

        Args:
            pixel_values: Vision pixel tensor of shape [Total_patches, patch_dim]
                          prepared by processor.
            grid_thw: Grid dimensions [N, 3] with (temporal, height, width) patch counts.

        Returns:
            Features post-patch-merger with dimension matching vision_feature_size.
        """
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(device=self.device)

        # Forward through vision encoder and spatial patch merger
        vision_outputs = self.visual(pixel_values, grid_thw=grid_thw)
        return vision_outputs

    def embed_text(self, text: str) -> torch.Tensor:
        """Tokenize and compute embedding vectors for a text string.

        Args:
            text: Input text prompt.

        Returns:
            Embedding tensor of shape [1, L, D_llm].
        """
        if self.processor is None or self.processor.tokenizer is None:
            raise RuntimeError("Processor/tokenizer is not initialized.")
        input_ids = self.processor.tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        return self.embedding_layer(input_ids)

    def assemble_injection_sequence(
        self,
        module_tokens: torch.Tensor,
        prefix_text: str,
        suffix_text: str = "",
        target_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        """Combine text prompt embeddings with injected module tokens.

        Args:
            module_tokens: Injected tokens from ContextModule of shape [B, T, D_llm].
            prefix_text: Text placed BEFORE the visual tokens (e.g. system instruction).
            suffix_text: Text placed AFTER the visual tokens (e.g. user question).
            target_text: Optional target answer string for scoring/training.

        Returns:
            Tuple of:
            - full_embeds: [B, L_total, D_llm]
            - labels: Optional [B, L_total] long tensor for cross-entropy loss (masked with -100)
            - prompt_length: Length of prompt preceding the target tokens
            - target_length: Number of target tokens (0 if target_text is None)
        """
        batch_size = module_tokens.shape[0]
        prefix_embeds = self.embed_text(prefix_text).repeat(batch_size, 1, 1)
        suffix_embeds = (
            self.embed_text(suffix_text).repeat(batch_size, 1, 1)
            if suffix_text
            else torch.empty(batch_size, 0, self.llm_hidden_size, device=self.device, dtype=self.dtype)
        )

        module_tokens = module_tokens.to(device=self.device, dtype=self.dtype)

        # Sequence: [Prefix, Visual_Module_Tokens, Suffix]
        prompt_embeds = torch.cat([prefix_embeds, module_tokens, suffix_embeds], dim=1)
        prompt_length = prompt_embeds.shape[1]

        if target_text is not None:
            target_ids = self.processor.tokenizer.encode(
                target_text, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            target_embeds = self.embedding_layer(target_ids).repeat(batch_size, 1, 1)
            target_length = target_ids.shape[1]

            full_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)

            # Build cross-entropy labels: mask out prompt tokens with -100
            labels = torch.full(
                (batch_size, prompt_length + target_length),
                -100,
                dtype=torch.long,
                device=self.device,
            )
            labels[:, prompt_length:] = target_ids.repeat(batch_size, 1)
        else:
            full_embeds = prompt_embeds
            labels = None
            target_length = 0

        return full_embeds, labels, prompt_length, target_length

    def forward_with_injected_tokens(
        self,
        module_tokens: torch.Tensor,
        prefix_text: str,
        suffix_text: str = "",
        target_text: Optional[str] = None,
    ) -> Any:
        """Run forward pass using injected context module tokens.

        Args:
            module_tokens: Injected tokens [B, T, D_llm].
            prefix_text: Text preceding visual tokens.
            suffix_text: Text following visual tokens.
            target_text: Optional target text for calculating loss.

        Returns:
            Model output containing logits, and loss if target_text is provided.
        """
        full_embeds, labels, _, _ = self.assemble_injection_sequence(
            module_tokens=module_tokens,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            target_text=target_text,
        )

        outputs = self.model(
            inputs_embeds=full_embeds,
            labels=labels,
        )
        return outputs

    def compute_target_log_prob(
        self,
        module_tokens: torch.Tensor,
        prefix_text: str,
        target_text: str,
        suffix_text: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Compute the log-probability of a target answer string.

        Args:
            module_tokens: Injected tokens [B, T, D_llm].
            prefix_text: Prompt prefix.
            target_text: Target answer to score.
            suffix_text: Prompt suffix.

        Returns:
            Dict containing:
            - 'loss': Cross-entropy loss (negative average log likelihood).
            - 'total_log_prob': Total log likelihood across all target tokens.
            - 'avg_log_prob': Average log likelihood per target token.
            - 'target_token_count': Number of target tokens.
        """
        full_embeds, labels, prompt_len, target_len = self.assemble_injection_sequence(
            module_tokens=module_tokens,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            target_text=target_text,
        )

        outputs = self.model(
            inputs_embeds=full_embeds,
            labels=labels,
        )

        loss = outputs.loss  # Mean cross-entropy over target tokens
        avg_log_prob = -loss
        total_log_prob = avg_log_prob * target_len

        return {
            "loss": loss,
            "total_log_prob": total_log_prob,
            "avg_log_prob": avg_log_prob,
            "target_token_count": torch.tensor(target_len, device=self.device),
        }

    @torch.no_grad()
    def generate_from_injected_tokens(
        self,
        module_tokens: torch.Tensor,
        prefix_text: str,
        suffix_text: str = "",
        max_new_tokens: int = 128,
        **generate_kwargs: Any,
    ) -> List[str]:
        """Generate text conditioned on injected module tokens and text prompts.

        Args:
            module_tokens: Injected tokens [B, T, D_llm].
            prefix_text: Prompt prefix.
            suffix_text: Prompt suffix.
            max_new_tokens: Maximum tokens to generate.
            **generate_kwargs: Additional arguments for model.generate.

        Returns:
            List of generated text strings per batch element.
        """
        full_embeds, _, _, _ = self.assemble_injection_sequence(
            module_tokens=module_tokens,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            target_text=None,
        )

        generated_ids = self.model.generate(
            inputs_embeds=full_embeds,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        results: List[str] = []
        for seq in generated_ids:
            decoded = self.processor.tokenizer.decode(seq, skip_special_tokens=True)
            results.append(decoded)
        return results

