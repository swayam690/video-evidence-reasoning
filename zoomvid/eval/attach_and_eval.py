"""Attach a trained module checkpoint and run the standard evaluation pipeline.

Usage (on Colab):
    python -m zoomvid.eval.attach_and_eval \
        --checkpoint /path/to/checkpoint.pt \
        --cache-dir /path/to/cache \
        --eval-manifest /path/to/dev_manifest.json \
        --budgets 1024 2048 4096 \
        --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from zoomvid.cache.cache_reader import FeatureCacheReader
from zoomvid.core.vlm_wrapper import FrozenVLMBackbone
from zoomvid.eval.evaluate import (
    RESULTS_TABLE_HEADER,
    evaluate_module,
    save_eval_report,
)
from zoomvid.module.interface import ContextModule
from zoomvid.module.reference.dummy import DummyModule
from zoomvid.module.reference.similarity import SimilarityModule
from zoomvid.module.reference.uniform import UniformModule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODULE_REGISTRY = {
    "uniform": UniformModule,
    "dummy": DummyModule,
    "similarity": SimilarityModule,
}


def load_module_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> ContextModule:
    """Load a trained ContextModule from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        device: Target device.
        dtype: Optional torch dtype.

    Returns:
        Loaded ContextModule instance.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    module_class_name = ckpt.get("module_class", "DummyModule")
    in_features = ckpt["in_features"]
    out_features = ckpt["out_features"]

    # Look up class from registry or try importing
    if module_class_name.lower() in MODULE_REGISTRY:
        cls = MODULE_REGISTRY[module_class_name.lower()]
    else:
        cls = MODULE_REGISTRY.get("dummy", DummyModule)
        logger.warning(
            "Module class '%s' not in registry. Using DummyModule as fallback.", module_class_name
        )

    module = cls(in_features=in_features, out_features=out_features)
    module.load_state_dict(ckpt["state_dict"])
    if dtype is not None:
        module.to(device=device, dtype=dtype)
    else:
        module.to(device)
    module.eval()
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="ZoomVid: Attach Module & Evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Module checkpoint path")
    parser.add_argument("--module-type", type=str, default=None,
                        choices=list(MODULE_REGISTRY.keys()),
                        help="Use a reference module instead of a checkpoint")
    parser.add_argument("--cache-dir", type=str, required=True, help="Feature cache directory")
    parser.add_argument("--eval-manifest", type=str, required=True, help="Dev eval manifest JSON")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048, 4096],
                        help="Token budgets to evaluate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load backbone
    print("Loading frozen VLM backbone...")
    backbone = FrozenVLMBackbone(
        model_id=args.model_id,
        torch_dtype=args.dtype,
        device=device,
        load_pretrained=True,
    )

    # Load cache
    cache_reader = FeatureCacheReader(args.cache_dir)
    print(f"Cache loaded: {len(cache_reader)} videos")

    # Load eval manifest
    with open(args.eval_manifest, "r", encoding="utf-8") as f:
        eval_manifest = json.load(f)
    print(f"Eval manifest loaded: {len(eval_manifest)} samples")

    # Load or create module
    in_feat = backbone.vision_feature_size
    out_feat = backbone.llm_hidden_size

    if args.checkpoint:
        module = load_module_from_checkpoint(args.checkpoint, device=device, dtype=backbone.dtype)
        module_name = f"checkpoint:{Path(args.checkpoint).stem}"
    elif args.module_type:
        cls = MODULE_REGISTRY[args.module_type]
        module = cls(in_features=in_feat, out_features=out_feat)
        module_name = args.module_type
    else:
        raise ValueError("Must specify either --checkpoint or --module-type")

    module.to(device=device, dtype=backbone.dtype)
    module.eval()

    # Evaluate across budgets
    print(f"\n{RESULTS_TABLE_HEADER}")
    all_reports = []

    for budget in args.budgets:
        report = evaluate_module(
            module=module,
            backbone=backbone,
            cache_reader=cache_reader,
            eval_manifest=eval_manifest,
            token_budget=budget,
            max_samples=args.max_samples,
        )
        print(report.to_markdown_row())
        all_reports.append(report)

        save_eval_report(
            report,
            output_dir / f"{module_name}_budget{budget}.json",
        )

    # Save combined summary
    summary = {
        "module": module_name,
        "budgets": args.budgets,
        "reports": [r.to_dict() for r in all_reports],
    }
    with open(output_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

