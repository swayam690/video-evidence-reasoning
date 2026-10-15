"""Evaluation harness for ZoomVid baselines and modules.

Runs a ContextModule through the frozen VLM backbone on a dev set,
computing accuracy, token usage statistics, and logging results.
Designed to work with cached features (no video decoding at eval time).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from zoomvid.cache.cache_reader import FeatureCacheReader
from zoomvid.module.interface import ContextModule, VideoBatch

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for a single evaluation result."""

    video_id: str
    question: str
    predicted: str
    ground_truth: str
    correct: bool
    token_count: int
    latency_ms: float


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    module_name: str
    token_budget: int
    num_samples: int
    accuracy: float
    avg_token_count: float
    total_tokens: int
    avg_latency_ms: float
    results: List[EvalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "token_budget": self.token_budget,
            "num_samples": self.num_samples,
            "accuracy": round(self.accuracy, 4),
            "avg_token_count": round(self.avg_token_count, 1),
            "total_tokens": self.total_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }

    def to_markdown_row(self) -> str:
        return (
            f"| {self.module_name} | {self.token_budget} | {self.num_samples} | "
            f"{self.accuracy:.2%} | {self.avg_token_count:.0f} | {self.avg_latency_ms:.1f} ms |"
        )


def evaluate_module(
    module: ContextModule,
    backbone: Any,  # FrozenVLMBackbone (avoid circular import)
    cache_reader: FeatureCacheReader,
    eval_manifest: List[Dict[str, Any]],
    token_budget: int = 2048,
    max_samples: Optional[int] = None,
    prefix_text: str = "<|im_start|>user\n",
    suffix_template: str = "{question}<|im_end|>\n<|im_start|>assistant\n",
    max_new_tokens: int = 64,
) -> EvalReport:
    """Evaluate a ContextModule on a dev set via the frozen VLM.

    Args:
        module: Any ContextModule (UniformModule, SimilarityModule, trained module, etc.).
        backbone: FrozenVLMBackbone instance (loaded on GPU).
        cache_reader: FeatureCacheReader with cached vision features.
        eval_manifest: List of eval sample dicts with 'video_id', 'question', 'answer'.
        token_budget: Visual token budget for this evaluation run.
        max_samples: Limit evaluation to first N samples (for quick dev runs).
        prefix_text: Text template before visual tokens.
        suffix_template: Text template after visual tokens ({question} is replaced).
        max_new_tokens: Maximum generation tokens.

    Returns:
        EvalReport with accuracy and per-sample results.
    """
    module.to(device=backbone.device, dtype=backbone.dtype)
    module.eval()
    results: List[EvalResult] = []

    samples = eval_manifest[:max_samples] if max_samples else eval_manifest
    total_tokens = 0
    correct_count = 0

    for i, sample in enumerate(samples):
        video_id = sample.get("video_id", "")
        question = sample.get("question", "")
        ground_truth = str(sample.get("answer", "")).strip().lower()

        # Skip if video not in cache
        if not cache_reader.manifest.get(video_id):
            logger.warning("Video %s not found in cache. Skipping.", video_id)
            continue

        try:
            # Load cached features and build VideoBatch
            batch = cache_reader.to_video_batch(
                video_id=video_id,
                token_budget=token_budget,
                mode="qa",
            )
            batch.frame_feats = batch.frame_feats.to(device=backbone.device, dtype=backbone.dtype)
            batch.timestamps = batch.timestamps.to(device=backbone.device)
            if batch.question_emb is not None:
                batch.question_emb = batch.question_emb.to(device=backbone.device, dtype=backbone.dtype)

            # Forward through context module
            t_start = time.time()
            with torch.no_grad():
                output = module(batch)
            module_tokens = output.tokens  # [1, T, D]
            token_count = module_tokens.shape[1]
            total_tokens += token_count

            # Generate answer via backbone
            suffix = suffix_template.format(question=question)

            # Build grid_thw for the module output
            image_grid_thw = torch.tensor(
                [[token_count, 1, 1]], dtype=torch.long, device=backbone.device
            )

            generated = backbone.generate_from_injected_tokens(
                module_tokens=module_tokens.to(backbone.device),
                prefix_text=prefix_text,
                suffix_text=suffix,
                max_new_tokens=max_new_tokens,
                image_grid_thw=image_grid_thw,
            )

            predicted = generated[0].strip().lower() if generated else ""
            latency = (time.time() - t_start) * 1000

            # Simple exact/substring match
            is_correct = (
                ground_truth == predicted
                or ground_truth in predicted
                or predicted in ground_truth
            )

            if is_correct:
                correct_count += 1

            results.append(EvalResult(
                video_id=video_id,
                question=question,
                predicted=predicted,
                ground_truth=ground_truth,
                correct=is_correct,
                token_count=token_count,
                latency_ms=latency,
            ))

            if (i + 1) % 10 == 0:
                logger.info(
                    "Evaluated %d/%d samples. Running accuracy: %.2f%%",
                    i + 1, len(samples), correct_count / len(results) * 100,
                )

        except Exception as e:
            logger.error("Error evaluating video %s: %s", video_id, e)
            continue

    num_evaluated = len(results)
    accuracy = correct_count / num_evaluated if num_evaluated > 0 else 0.0
    avg_tokens = total_tokens / num_evaluated if num_evaluated > 0 else 0.0
    avg_latency = sum(r.latency_ms for r in results) / num_evaluated if num_evaluated > 0 else 0.0

    return EvalReport(
        module_name=module.__class__.__name__,
        token_budget=token_budget,
        num_samples=num_evaluated,
        accuracy=accuracy,
        avg_token_count=avg_tokens,
        total_tokens=total_tokens,
        avg_latency_ms=avg_latency,
        results=results,
    )


def save_eval_report(report: EvalReport, output_path: Path) -> None:
    """Save evaluation report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Saved eval report to %s", output_path)


RESULTS_TABLE_HEADER = (
    "| Module | Budget | Samples | Accuracy | Avg Tokens | Avg Latency |\n"
    "| :--- | :---: | :---: | :---: | :---: | :---: |"
)

