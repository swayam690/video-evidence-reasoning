"""Unit tests for dataset deduplication, sharded feature caching, and cache reader."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from zoomvid.data.download_and_split import (
    deduplicate_dataset,
    extract_benchmark_video_ids,
    normalize_video_id,
)
from zoomvid.cache.feature_cache import FeatureCacheWriter
from zoomvid.cache.cache_reader import FeatureCacheReader
from zoomvid.module.reference.uniform import UniformModule


def test_normalize_video_id() -> None:
    """Test video ID normalization across formats and prefixes."""
    assert normalize_video_id("video_12345.mp4") == "12345"
    assert normalize_video_id("v_abcde.webm") == "abcde"
    assert normalize_video_id("/path/to/some/clip.mkv") == "clip"
    assert normalize_video_id("plain_id") == "plain_id"


def test_deduplicate_dataset() -> None:
    """Test that training videos overlapping with benchmark IDs are strictly removed."""
    eval_items = [
        {"video_id": "video_bench_01.mp4", "question": "Q1"},
        {"video_id": "bench_02.webm", "question": "Q2"},
        {"video_id": "shared_03.mkv", "question": "Q3"},
    ]
    eval_ids = extract_benchmark_video_ids(eval_items)
    assert "bench_01" in eval_ids
    assert "bench_02" in eval_ids
    assert "shared_03" in eval_ids

    train_items = [
        {"video_id": "train_01.mp4", "text": "Caption 1"},
        {"video_id": "shared_03.mp4", "text": "Overlap caption"},  # Should be removed!
        {"video_id": "train_02.mp4", "text": "Caption 2"},
        {"video_id": "video_bench_01.mp4", "text": "Overlap caption 2"},  # Should be removed!
    ]

    clean_train, removed = deduplicate_dataset(train_items, eval_ids)

    assert len(clean_train) == 2
    assert len(removed) == 2
    clean_ids = {normalize_video_id(item["video_id"]) for item in clean_train}
    assert clean_ids == {"train_01", "train_02"}
    removed_ids = {normalize_video_id(item["video_id"]) for item in removed}
    assert removed_ids == {"shared_03", "bench_01"}


def test_feature_cache_sharding_and_resume_safety() -> None:
    """Test feature caching: fp16 storage, shard limit rollover, and resume safety."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"

        # Set a small shard limit (e.g. 0.05 MB = 50 KB) to trigger sharding across multiple files
        writer = FeatureCacheWriter(cache_dir, shard_size_limit_mb=0.05)

        # Create 6 videos: 4 frames, 16 patches, feature_dim 64 in fp32
        # Each video in fp16 = 4 * 16 * 64 * 2 bytes = 8,192 bytes (~8 KB)
        # 6 videos = ~49 KB -> will trigger sharding
        for i in range(6):
            vid = f"vid_{i:04d}"
            feats = torch.randn(4, 16, 64, dtype=torch.float32)
            times = torch.linspace(0.0, 3.0, 4)
            writer.add_video(vid, feats, times, fps=1.0, resolution=(336, 336))

        writer.close()

        # Check manifest file exists
        manifest_file = cache_dir / "manifest.json"
        assert manifest_file.exists()

        summary = writer.get_summary()
        assert summary["total_videos"] == 6
        assert summary["avg_size_kb_per_video"] > 0

        # Verify sharding occurred
        shards = list(cache_dir.glob("shard_*.pt"))
        assert len(shards) >= 1

        # Test Resume Safety: Re-open writer and add existing videos again
        writer_resume = FeatureCacheWriter(cache_dir, shard_size_limit_mb=0.05)
        assert writer_resume.is_cached("vid_0000")
        assert writer_resume.is_cached("vid_0005")

        # Adding existing video should be a no-op
        writer_resume.add_video("vid_0000", torch.randn(4, 16, 64), torch.linspace(0, 3, 4), 1.0, (336, 336))
        writer_resume.close()

        # Manifest should still have exactly 6 videos
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        assert len(manifest_data) == 6


def test_feature_cache_reader_and_module_integration() -> None:
    """Test FeatureCacheReader streaming and feeding directly into a ContextModule."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        writer = FeatureCacheWriter(cache_dir, shard_size_limit_mb=1.0)

        # 3 frames, 144 patches, 2048 dim (standard Qwen2.5-VL feature dimensions)
        n, p, d = 3, 144, 2048
        feats = torch.randn(n, p, d, dtype=torch.float32)
        times = torch.tensor([0.0, 1.0, 2.0])
        writer.add_video("test_clip_01", feats, times, fps=1.0, resolution=(336, 336))
        writer.close()

        # Read back with FeatureCacheReader
        reader = FeatureCacheReader(cache_dir)
        assert len(reader) == 1
        assert reader.video_ids == ["test_clip_01"]

        item = reader[0]
        loaded_feats = item["frame_feats"]
        loaded_times = item["timestamps"]

        assert loaded_feats.dtype == torch.float16
        assert loaded_feats.shape == (n, p, d)
        assert loaded_times.shape == (n,)

        # Convert to VideoBatch
        batch = reader.to_video_batch("test_clip_01", token_budget=500)
        assert batch.frame_feats.shape == (1, n, p, d)
        assert batch.timestamps.shape == (1, n)

        # Feed to UniformModule
        module = UniformModule(in_features=d, out_features=d)
        output = module(batch)
        assert output.tokens.ndim == 3
        assert output.tokens.shape[1] <= 500

