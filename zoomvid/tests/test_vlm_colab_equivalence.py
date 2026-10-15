"""Equivalence verification test for Qwen2.5-VL-3B-Instruct on Google Colab (T4).

Run this script on Google Colab with GPU enabled:
    python zoomvid/tests/test_vlm_colab_equivalence.py

Verifies:
1. All parameters in FrozenVLMBackbone have requires_grad=False.
2. Visual tokens extracted post-patch-merger and fed back via mm_encoder_outputs
   produce logits identical to the native VLM pipeline in fp16.
"""

import sys
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling

from zoomvid.core.vlm_wrapper import FrozenVLMBackbone


def run_colab_vlm_equivalence_test() -> None:
    print("=" * 70)
    print("Starting Qwen2.5-VL-3B Equivalence & Freeze Test (Colab T4 / GPU)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This test is intended for GPU (Colab T4).")
        device = "cpu"
        dtype_str = "fp32"
    else:
        device = "cuda"
        dtype_str = "fp16"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Device: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Step 1: Initialize FrozenVLMBackbone
    print(f"\n1. Initializing FrozenVLMBackbone ({model_id}) in {dtype_str}...")
    backbone = FrozenVLMBackbone(
        model_id=model_id,
        torch_dtype=dtype_str,
        device=device,
        load_pretrained=True,
    )

    # Step 2: Assert strictly frozen
    print("2. Verifying parameter freeze (requires_grad=False)...")
    backbone.assert_frozen()
    num_params = sum(p.numel() for p in backbone.model.parameters())
    num_trainable = sum(p.numel() for p in backbone.model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {num_trainable}")
    assert num_trainable == 0, "Expected 0 trainable parameters!"
    print("   [PASS] Backbone is 100% frozen.")

    # Step 3: Create synthetic test input
    print("\n3. Creating synthetic input image and prompt...")
    test_image = Image.new("RGB", (336, 336), color=(128, 64, 200))
    prompt_text = "Describe what you see in the image."

    processor = backbone.processor

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Step 4: Run native pipeline (pixel_values → vision encoder → masked_scatter → LLM)
    print("4. Running native VLM pipeline forward pass...")
    with torch.no_grad():
        native_outputs = backbone.model(**inputs)
        native_logits = native_outputs.logits

    # Step 5: Extract visual tokens post-patch-merger
    print("5. Extracting visual features post-patch-merger...")
    with torch.no_grad():
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        visual_tokens = backbone.extract_vision_features(
            pixel_values=pixel_values, grid_thw=image_grid_thw
        )
    print(f"   Extracted visual tokens shape: {list(visual_tokens.shape)}")

    # Step 6: Feed extracted visual tokens back via mm_encoder_outputs
    # This bypasses the vision encoder but preserves masked_scatter and M-RoPE
    print("6. Re-injecting visual tokens via mm_encoder_outputs...")
    mm_encoder_outputs = {
        "image": BaseModelOutputWithPooling(
            last_hidden_state=None,
            pooler_output=[visual_tokens],
            hidden_states=None,
            attentions=None,
        )
    }

    # Build inputs WITHOUT pixel_values (we supply pre-computed features instead)
    inject_inputs = {
        k: v for k, v in inputs.items()
        if k not in ("pixel_values", "pixel_values_videos")
    }
    inject_inputs["mm_encoder_outputs"] = mm_encoder_outputs

    with torch.no_grad():
        injected_outputs = backbone.model(**inject_inputs)
        injected_logits = injected_outputs.logits

    # Step 7: Compare logits
    print("7. Comparing native logits vs injected logits...")
    print(f"   Native logits shape:   {list(native_logits.shape)}")
    print(f"   Injected logits shape: {list(injected_logits.shape)}")

    assert native_logits.shape == injected_logits.shape, (
        f"Shape mismatch: native {native_logits.shape} vs injected {injected_logits.shape}"
    )

    max_diff = torch.abs(native_logits.float() - injected_logits.float()).max().item()
    mean_diff = torch.abs(native_logits.float() - injected_logits.float()).mean().item()
    cos_sim = torch.cosine_similarity(
        native_logits[0, -1, :].float(), injected_logits[0, -1, :].float(), dim=-1
    ).item()

    print(f"   Max absolute difference:  {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.8f}")
    print(f"   Cosine similarity (last token): {cos_sim:.8f}")

    # In fp16, exact match expected since we extracted and re-injected the same features
    assert max_diff < 0.01, f"Max logit difference too high: {max_diff}"
    assert cos_sim > 0.9999, f"Cosine similarity too low: {cos_sim}"
    print("   [PASS] Injection path logits match native pipeline!")

    # Step 8: Log VRAM usage
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\n8. Peak GPU VRAM usage: {peak_vram:.2f} GB")

    print("\n" + "=" * 70)
    print("ALL PHASE 0 EQUIVALENCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_colab_vlm_equivalence_test()
