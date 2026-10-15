"""Frozen Vision-Language Model Backbone Wrapper for ZoomVid.

Wraps Qwen/Qwen2.5-VL-3B-Instruct in a strictly frozen configuration (requires_grad=False).
Provides utilities for per-frame vision feature extraction, custom embedding injection
via mm_encoder_outputs (preserving M-RoPE 3D positional encoding), answer generation,
and log-probability scoring of target answers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling

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
    into the LLM embedding stream via mm_encoder_outputs (preserving M-RoPE).
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

        # Read special token IDs for building injection input_ids
        self.image_token_id: int = getattr(self.config, "image_token_id", 151655)
        self.video_token_id: int = getattr(self.config, "video_token_id", 151656)
        self.vision_start_token_id: int = getattr(self.config, "vision_start_token_id", 151652)
        self.vision_end_token_id: int = getattr(self.config, "vision_end_token_id", 151653)

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
            pixel_values: Vision pixel tensor prepared by processor.
            grid_thw: Grid dimensions [N, 3] with (temporal, height, width) patch counts.

        Returns:
            Features post-patch-merger of shape [Total_merged_patches, D_vision].
        """
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(device=self.device)

        # Forward through vision encoder and spatial patch merger
        vision_outputs = self.visual(pixel_values, grid_thw=grid_thw)

        # Handle BaseModelOutputWithPooling or raw tensor return types
        if isinstance(vision_outputs, torch.Tensor):
            return vision_outputs
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            out = vision_outputs.pooler_output
            if isinstance(out, (tuple, list)):
                return torch.cat(out, dim=0)
            return out
        if hasattr(vision_outputs, "last_hidden_state") and vision_outputs.last_hidden_state is not None:
            return vision_outputs.last_hidden_state
        return vision_outputs[0]

    def build_injection_input_ids(
        self,
        prefix_text: str,
        suffix_text: str,
        num_visual_tokens: int,
        target_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, int, int]:
        """Build input_ids with image placeholder tokens for module injection.

        Creates: [prefix_tokens, vision_start, image_pad * T, vision_end, suffix_tokens, target_tokens]
        This ensures proper M-RoPE 3D positional encoding for visual tokens.

        Args:
            prefix_text: Text before the visual tokens (e.g. system prompt).
            suffix_text: Text after the visual tokens (e.g. user question).
            num_visual_tokens: Number of visual tokens T produced by the context module.
            target_text: Optional target answer text for loss computation.

        Returns:
            Tuple of:
            - input_ids: [1, L_total] with image placeholders
            - labels: Optional [1, L_total] with -100 masking for prompt
            - mm_token_type_ids: [1, L_total] marking image token positions
            - prompt_length: Number of prompt tokens (before target)
            - target_length: Number of target tokens
        """
        tokenizer = self.processor.tokenizer

        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

        # Build: [prefix, vision_start, image_pad * T, vision_end, suffix]
        prompt_ids = (
            prefix_ids
            + [self.vision_start_token_id]
            + [self.image_token_id] * num_visual_tokens
            + [self.vision_end_token_id]
            + suffix_ids
        )
        prompt_length = len(prompt_ids)

        if target_text is not None:
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)
            target_length = len(target_ids)
            full_ids = prompt_ids + target_ids
        else:
            target_ids = []
            target_length = 0
            full_ids = prompt_ids

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)

        # Build mm_token_type_ids: 1 for image tokens, 0 for text
        mm_type_ids = torch.zeros_like(input_ids)
        img_start = len(prefix_ids) + 1  # after vision_start
        img_end = img_start + num_visual_tokens
        mm_type_ids[0, img_start:img_end] = 1

        # Build labels: -100 for prompt, actual ids for target
        if target_text is not None:
            labels = torch.full_like(input_ids, -100)
            labels[0, prompt_length:] = input_ids[0, prompt_length:]
        else:
            labels = None

        return input_ids, labels, mm_type_ids, prompt_length, target_length

    def forward_with_injected_tokens(
        self,
        module_tokens: torch.Tensor,
        prefix_text: str,
        suffix_text: str = "",
        target_text: Optional[str] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Any:
        """Run forward pass injecting context module tokens via mm_encoder_outputs.

        Uses the model's native masked_scatter + M-RoPE path to ensure correct
        3D positional encoding for the injected visual tokens.

        Args:
            module_tokens: Injected tokens [B, T, D_llm] from a ContextModule.
            prefix_text: Text preceding visual tokens.
            suffix_text: Text following visual tokens.
            target_text: Optional target text for calculating loss.
            image_grid_thw: Optional grid dimensions [N, 3]. If None, treats tokens
                            as a 1D temporal sequence.

        Returns:
            Model output containing logits, and loss if target_text is provided.
        """
        b, t, d = module_tokens.shape
        module_tokens = module_tokens.to(device=self.device, dtype=self.dtype)

        # Build input_ids with placeholder tokens
        input_ids, labels, mm_type_ids, _, _ = self.build_injection_input_ids(
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            num_visual_tokens=t,
            target_text=target_text,
        )

        # Repeat for batch
        if b > 1:
            input_ids = input_ids.repeat(b, 1)
            mm_type_ids = mm_type_ids.repeat(b, 1)
            if labels is not None:
                labels = labels.repeat(b, 1)

        # Build image_grid_thw if not provided: treat as 1D temporal sequence
        if image_grid_thw is None:
            image_grid_thw = torch.tensor([[t, 1, 1]], dtype=torch.long, device=self.device)

        # Wrap module tokens as mm_encoder_outputs (one list entry per batch element)
        per_sample_tokens = [module_tokens[i] for i in range(b)]
        mm_encoder_outputs = {
            "image": BaseModelOutputWithPooling(
                last_hidden_state=None,
                pooler_output=per_sample_tokens,
                hidden_states=None,
                attentions=None,
            )
        }

        attention_mask = torch.ones_like(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_type_ids,
            image_grid_thw=image_grid_thw,
            mm_encoder_outputs=mm_encoder_outputs,
            labels=labels,
        )
        return outputs

    def compute_target_log_prob(
        self,
        module_tokens: torch.Tensor,
        prefix_text: str,
        target_text: str,
        suffix_text: str = "",
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the log-probability of a target answer string.

        Args:
            module_tokens: Injected tokens [B, T, D_llm].
            prefix_text: Prompt prefix.
            target_text: Target answer to score.
            suffix_text: Prompt suffix.
            image_grid_thw: Optional grid dimensions.

        Returns:
            Dict containing:
            - 'loss': Cross-entropy loss (negative average log likelihood).
            - 'total_log_prob': Total log likelihood across all target tokens.
            - 'avg_log_prob': Average log likelihood per target token.
            - 'target_token_count': Number of target tokens.
        """
        _, _, _, _, target_len = self.build_injection_input_ids(
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            num_visual_tokens=module_tokens.shape[1],
            target_text=target_text,
        )

        outputs = self.forward_with_injected_tokens(
            module_tokens=module_tokens,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            target_text=target_text,
            image_grid_thw=image_grid_thw,
        )

        loss = outputs.loss
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
        image_grid_thw: Optional[torch.Tensor] = None,
        **generate_kwargs: Any,
    ) -> List[str]:
        """Generate text conditioned on injected module tokens and text prompts.

        Args:
            module_tokens: Injected tokens [B, T, D_llm].
            prefix_text: Prompt prefix.
            suffix_text: Prompt suffix.
            max_new_tokens: Maximum tokens to generate.
            image_grid_thw: Optional grid dimensions.
            **generate_kwargs: Additional arguments for model.generate.

        Returns:
            List of generated text strings per batch element.
        """
        b, t, d = module_tokens.shape
        module_tokens = module_tokens.to(device=self.device, dtype=self.dtype)

        input_ids, _, mm_type_ids, prompt_length, _ = self.build_injection_input_ids(
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            num_visual_tokens=t,
            target_text=None,
        )

        if b > 1:
            input_ids = input_ids.repeat(b, 1)
            mm_type_ids = mm_type_ids.repeat(b, 1)

        if image_grid_thw is None:
            image_grid_thw = torch.tensor([[t, 1, 1]], dtype=torch.long, device=self.device)

        per_sample_tokens = [module_tokens[i] for i in range(b)]
        mm_encoder_outputs = {
            "image": BaseModelOutputWithPooling(
                last_hidden_state=None,
                pooler_output=per_sample_tokens,
                hidden_states=None,
                attentions=None,
            )
        }

        attention_mask = torch.ones_like(input_ids)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_type_ids,
            image_grid_thw=image_grid_thw,
            mm_encoder_outputs=mm_encoder_outputs,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        # Trim prompt tokens from generated output
        results: List[str] = []
        for seq in generated_ids:
            output_ids = seq[prompt_length:]
            decoded = self.processor.tokenizer.decode(output_ids, skip_special_tokens=True)
            results.append(decoded)
        return results
