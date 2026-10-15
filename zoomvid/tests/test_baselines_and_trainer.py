"""Unit tests for SimilarityModule baseline, TrainingDataset, and checkpointing."""

import json
import tempfile
from pathlib import Path
import pytest
import torch

from zoomvid.module.interface import VideoBatch
from zoomvid.module.reference.similarity import SimilarityModule
from zoomvid.module.reference.dummy import DummyModule
from zoomvid.cache.feature_cache import FeatureCacheWriter
from zoomvid.cache.cache_reader import FeatureCacheReader
from zoomvid.train.trainer import TrainingDataset, Trainer


def test_similarity_module_budget_and_chronology() -> None:
    """Test SimilarityModule respects token budget and sorts frames chronologically."""
    b, n, p, d = 2, 8, 16, 64
    out_dim = 128
    frame_feats = torch.randn(b, n, p, d)
    timestamps = torch.linspace(0.0, 7.0, n).unsqueeze(0).repeat(b, 1)

    # Question embedding matching dimension d
    question_emb = torch.randn(b, d)

    # Budget allows for only 3 frames (3 * 16 = 48 tokens)
    budget = 50
    batch = VideoBatch(
        frame_feats=frame_feats,
        timestamps=timestamps,
        question_emb=question_emb,
        token_budget=budget,
    )

    module = SimilarityModule(in_features=d, out_features=out_dim)
    output = module(batch)

    # 1. Output dimension checks
    assert output.tokens.ndim == 3
    assert output.tokens.shape[0] == b
    assert output.tokens.shape[1] <= budget
    assert output.tokens.shape[2] == out_dim
    assert output.aux["num_frames_selected"] == 3

    # 2. Assert selected frame indices are sorted chronologically
    selected_indices = output.aux["selected_indices"]
    for b_idx in range(b):
        indices = selected_indices[b_idx].tolist()
        assert indices == sorted(indices), "Selected indices must be sorted chronologically!"


def test_similarity_module_prefers_matching_frames() -> None:
    """Test SimilarityModule selects frame with highest cosine similarity."""
    b, n, p, d = 1, 4, 4, 32
    frame_feats = torch.zeros(b, n, p, d)
    timestamps = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    # Make frame index 2 closely match question_emb
    target_vector = torch.randn(d)
    frame_feats[0, 2] = target_vector.unsqueeze(0).repeat(p, 1)
    question_emb = target_vector.unsqueeze(0)  # [1, d]

    # Budget allows only 1 frame (1 * 4 = 4 tokens)
    batch = VideoBatch(
        frame_feats=frame_feats,
        timestamps=timestamps,
        question_emb=question_emb,
        token_budget=4,
    )

    module = SimilarityModule(in_features=d, out_features=d)
    output = module(batch)

    assert output.aux["num_frames_selected"] == 1
    assert output.aux["selected_indices"][0].item() == 2


def test_training_dataset_and_checkpointing() -> None:
    """Test TrainingDataset loads cached samples and Trainer checkpoint save/reload works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cache_dir = tmp_path / "cache"

        # Create 2 cached videos
        writer = FeatureCacheWriter(cache_dir, shard_size_limit_mb=1.0)
        writer.add_video("vid_1", torch.randn(2, 4, 16), torch.tensor([0.0, 1.0]), 1.0, (336, 336))
        writer.add_video("vid_2", torch.randn(2, 4, 16), torch.tensor([0.0, 1.0]), 1.0, (336, 336))
        writer.close()

        cache_reader = FeatureCacheReader(cache_dir)

        # Manifest with 1 matching and 1 missing video
        manifest = [
            {"video_id": "vid_1", "question": "What is here?", "answer": "A cat."},
            {"video_id": "vid_missing", "question": "Unknown?", "answer": "Nothing."},
            {"video_id": "vid_2", "conversations": [{"from": "human", "value": "Action?"}, {"from": "gpt", "value": "Jump."}]},
        ]

        dataset = TrainingDataset(cache_reader, manifest)
        assert len(dataset) == 2  # Only vid_1 and vid_2 should be kept

        item0 = dataset[0]
        assert item0["video_id"] == "vid_1"
        assert item0["question"] == "What is here?"
        assert item0["answer"] == "A cat."

        item1 = dataset[1]
        assert item1["video_id"] == "vid_2"
        assert item1["question"] == "Action?"
        assert item1["answer"] == "Jump."

        # Test module save and reload
        module = DummyModule(in_features=16, out_features=32)
        ckpt_path = tmp_path / "checkpoint.pt"
        module.save(ckpt_path)

        new_module = DummyModule(in_features=16, out_features=32)
        new_module.load(ckpt_path)

        for p1, p2 in zip(module.parameters(), new_module.parameters()):
            assert torch.allclose(p1, p2)


def test_eval_report_formatting() -> None:
    """Test format_markdown_table and load_all_reports from zoomvid.eval.report."""
    from zoomvid.eval.report import format_markdown_table, load_all_reports

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        report_data = {
            "module_name": "TestModule",
            "token_budget": 1024,
            "num_samples": 10,
            "accuracy": 0.85,
            "avg_token_count": 720.0,
            "avg_latency_ms": 150.0,
        }
        with open(tmp_path / "report1.json", "w") as f:
            json.dump(report_data, f)

        reports = load_all_reports(tmp_path)
        assert len(reports) == 1
        table = format_markdown_table(reports)
        assert "TestModule" in table
        assert "85.00%" in table
        assert "1024" in table


