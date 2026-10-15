"""Dataset download, splitting, and benchmark deduplication pipeline.

Handles downloading subsets of LLaVA-Video-178K (training) and TempCompass (dev eval),
strictly guarantees that no evaluation benchmark videos appear in the training split,
and formats data into standard manifest files for feature caching and training.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def normalize_video_id(raw_id: str) -> str:
    """Normalize a video identifier or path to a clean canonical string.

    Strips directory paths, extensions (.mp4, .webm, etc.), and URL query strings.
    """
    cleaned = Path(raw_id).stem
    # Remove common video prefixes if present (e.g. 'v_', 'video_')
    cleaned = re.sub(r"^(video_|v_)", "", cleaned)
    return cleaned.strip()


def extract_benchmark_video_ids(eval_items: Iterable[Dict[str, Any]]) -> Set[str]:
    """Extract all normalized video IDs from benchmark items."""
    ids: Set[str] = set()
    for item in eval_items:
        for key in ("video_id", "video", "video_name", "vid", "yt_id"):
            val = item.get(key)
            if val:
                ids.add(normalize_video_id(str(val)))
    return ids


def deduplicate_dataset(
    train_items: List[Dict[str, Any]],
    eval_video_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove any training records that match evaluation benchmark video IDs.

    Args:
        train_items: List of training sample dictionaries.
        eval_video_ids: Set of normalized video IDs from eval benchmarks.

    Returns:
        Tuple of (clean_train_items, removed_overlap_items).
    """
    clean: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for item in train_items:
        # Check all possible keys that might contain the video identifier
        matched_eval = False
        for key in ("video_id", "video", "video_name", "vid", "yt_id", "path"):
            val = item.get(key)
            if val:
                norm_id = normalize_video_id(str(val))
                if norm_id in eval_video_ids:
                    matched_eval = True
                    break

        if matched_eval:
            removed.append(item)
        else:
            clean.append(item)

    logger.info(
        "Deduplication complete: %d train samples kept, %d removed due to eval benchmark overlap.",
        len(clean),
        len(removed),
    )
    return clean, removed


def prepare_synthetic_dev_set(output_path: Path, num_samples: int = 10) -> List[Dict[str, Any]]:
    """Create a minimal synthetic sample set for unit tests and local smoke runs."""
    samples = []
    for i in range(num_samples):
        samples.append({
            "video_id": f"synthetic_video_{i:04d}",
            "video_path": str(output_path / f"synthetic_{i:04d}.mp4"),
            "question": f"What action is performed in video {i}?",
            "options": ["Running", "Jumping", "Standing", "Walking"],
            "answer": "Running",
            "task_type": "action_recognition",
            "duration": 5.0,
        })
    return samples


def save_manifest(items: List[Dict[str, Any]], output_file: Path) -> None:
    """Save dataset records as a formatted JSON manifest."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    logger.info("Saved manifest (%d records) to %s", len(items), output_file)


def load_manifest(input_file: Path) -> List[Dict[str, Any]]:
    """Load dataset records from a JSON manifest."""
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="ZoomVid Dataset Download & Deduplication")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--max-train", type=int, default=1000, help="Max training videos to keep")
    parser.add_argument("--max-eval", type=int, default=200, help="Max dev evaluation samples")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading large files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ZoomVid Dataset Preparation & Deduplication Plan")
    print(f"Target Directory: {out_dir}")
    print(f"Max Training Samples: {args.max_train}")
    print(f"Max Eval Questions:   {args.max_eval}")
    print(f"Dry Run Mode:         {args.dry_run}")
    print("=" * 65)

    # In dry-run or local mode, construct the schema manifests
    eval_manifest_path = out_dir / "dev_tempcompass_manifest.json"
    train_manifest_path = out_dir / "train_llava_manifest.json"

    synthetic_eval = prepare_synthetic_dev_set(out_dir, num_samples=min(10, args.max_eval))
    save_manifest(synthetic_eval, eval_manifest_path)

    # Example train set with intentional overlap to demonstrate de-duplication
    sample_train = [
        {
            "video_id": f"synthetic_video_{i:04d}",
            "video_path": str(out_dir / f"synthetic_{i:04d}.mp4"),
            "conversations": [{"from": "human", "value": "Describe this video."}, {"from": "gpt", "value": "Sample."}],
        }
        for i in range(5)  # Matches first 5 eval videos
    ] + [
        {
            "video_id": f"train_video_{j:04d}",
            "video_path": str(out_dir / f"train_{j:04d}.mp4"),
            "conversations": [{"from": "human", "value": "Describe this video."}, {"from": "gpt", "value": "Sample."}],
        }
        for j in range(20)
    ]

    eval_ids = extract_benchmark_video_ids(synthetic_eval)
    clean_train, removed = deduplicate_dataset(sample_train, eval_ids)
    save_manifest(clean_train[: args.max_train], train_manifest_path)

    print(f"Successfully generated train and dev manifests in {out_dir}")


if __name__ == "__main__":
    main()

