"""Streamable feature cache reader and PyTorch Dataset for ZoomVid.

Reads sharded video features and timestamps with LRU shard caching, allowing
efficient streaming on Google Colab or local disk.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from zoomvid.module.interface import VideoBatch


class FeatureCacheReader(Dataset):
    """Dataset and streamable reader for sharded ZoomVid feature cache."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_open_shards: int = 4,
    ) -> None:
        """Initialize FeatureCacheReader.

        Args:
            cache_dir: Directory containing manifest.json and shard_*.pt files.
            max_open_shards: Maximum number of open shards to keep in memory (LRU).
        """
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Cache manifest not found at {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest: Dict[str, Dict[str, Any]] = json.load(f)

        self.video_ids: List[str] = list(self.manifest.keys())
        self.max_open_shards = max_open_shards
        self._shard_cache: OrderedDict[str, Dict[str, Dict[str, torch.Tensor]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.video_ids)

    def _get_shard_data(self, shard_file: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """Retrieve shard dictionary, loading from disk with LRU caching."""
        if shard_file in self._shard_cache:
            self._shard_cache.move_to_end(shard_file)
            return self._shard_cache[shard_file]

        shard_path = self.cache_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        data = torch.load(shard_path, map_location="cpu", weights_only=True)
        self._shard_cache[shard_file] = data

        if len(self._shard_cache) > self.max_open_shards:
            self._shard_cache.popitem(last=False)

        return data

    def get_features(self, video_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch frame features and timestamps for a specific video ID.

        Returns:
            Tuple of (frame_feats [N, P, D] in fp16, timestamps [N]).
        """
        if video_id not in self.manifest:
            raise KeyError(f"Video ID '{video_id}' not found in cache manifest.")

        meta = self.manifest[video_id]
        shard_file = meta["shard_file"]
        shard_data = self._get_shard_data(shard_file)

        item = shard_data[video_id]
        return item["frame_feats"], item["timestamps"]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Dataset item retriever.

        Returns:
            Dict with 'video_id', 'frame_feats' [N, P, D], 'timestamps' [N], and metadata.
        """
        vid = self.video_ids[idx]
        feats, times = self.get_features(vid)
        return {
            "video_id": vid,
            "frame_feats": feats,
            "timestamps": times,
            "meta": self.manifest[vid],
        }

    def to_video_batch(
        self,
        video_id: str,
        token_budget: int = 2048,
        mode: str = "qa",
        question_emb: Optional[torch.Tensor] = None,
    ) -> VideoBatch:
        """Helper to create a single-video VideoBatch [1, N, P, D] ready for ContextModule."""
        feats, times = self.get_features(video_id)
        return VideoBatch(
            frame_feats=feats.unsqueeze(0),
            timestamps=times.unsqueeze(0),
            question_emb=question_emb,
            mode=mode,
            token_budget=token_budget,
        )

