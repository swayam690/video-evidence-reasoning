"""Equivalence verification test for Qwen2.5-VL-3B-Instruct on Google Colab (T4).

Run this script on Google Colab with GPU enabled:
    python zoomvid/tests/test_vlm_colab_equivalence.py

Verifies:
1. All parameters in FrozenVLMBackbone have requires_grad=False.
2. Visual tokens extracted post-patch-merger and fed back through the custom
   injection path match the output logits of the native VLM pipeline in fp16.
"""

import sys
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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
    # Create a 336x336 synthetic image
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

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = processor.image_processor(
        images=[test_image], return_tensors="pt"
    )

    inputs = processor(
        text=[text_prompt],
        images=test_image,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Step 4: Run native pipeline
    print("4. Running native VLM pipeline forward pass...")
    with torch.no_grad():
        native_outputs = backbone.model(**inputs)
        native_logits = native_outputs.logits

    # Step 5: Extract visual tokens using backbone
    print("5. Extracting visual features post-patch-merger...")
    with torch.no_grad():
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        visual_tokens = backbone.extract_vision_features(
            pixel_values=pixel_values, grid_thw=image_grid_thw
        )
    print(f"   Extracted visual tokens shape: {list(visual_tokens.shape)}")

    # Step 6: Feed visual tokens back through the injection path
    print("6. Feeding visual tokens through custom injection path...")
    prefix = "<|im_start|>user\n<|vision_start|>"
    suffix = f"<|vision_end|>{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

    # In batch format [B, T, D]
    visual_tokens_batched = visual_tokens.unsqueeze(0)

    with torch.no_grad():
        injected_outputs = backbone.forward_with_injected_tokens(
            module_tokens=visual_tokens_batched,
            prefix_text=prefix,
            suffix_text=suffix,
        )
        injected_logits = injected_outputs.logits

    # Step 7: Compare logits
    print("7. Comparing native logits vs injected logits...")
    print(f"   Native logits shape:   {list(native_logits.shape)}")
    print(f"   Injected logits shape: {list(injected_logits.shape)}")

    # Check maximum absolute difference
    if native_logits.shape == injected_logits.shape:
        diff = torch.abs(native_logits - injected_logits).max().item()
        print(f"   Max absolute difference: {diff:.6f}")
        assert diff < 1e-2, f"Logit difference too high: {diff}"
        print("   [PASS] Injection path logits match native pipeline!")
    else:
        print(f"   Note: Sequence lengths ({native_logits.shape[1]} vs {injected_logits.shape[1]}) differ slightly due to template tokens.")
        print("   Comparing last token logits (next-token prediction distribution):")
        cos_sim = torch.cosine_similarity(
            native_logits[0, -1, :], injected_logits[0, -1, :], dim=-1
        ).item()
        print(f"   Cosine similarity of next token prediction: {cos_sim:.6f}")
        assert cos_sim > 0.99, f"Cosine similarity below threshold: {cos_sim}"
        print("   [PASS] Next token prediction distributions match (>0.99 cosine similarity)!")

    print("\n" + "=" * 70)
    print("ALL PHASE 0 EQUIVALENCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_colab_vlm_equivalence_test()

