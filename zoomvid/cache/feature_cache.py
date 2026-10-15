"""Feature extraction and sharded caching pipeline for ZoomVid.

Decodes video at 1-2 fps and fixed resolution (336x336), passes frames through
the frozen vision encoder post-patch-merger, and saves features + timestamps
in fp16 across ~500 MB shards with a JSON manifest. Fully resume-safe.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def decode_video_frames(
    video_path: Union[str, Path],
    target_fps: float = 1.0,
    target_resolution: Tuple[int, int] = (336, 336),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode video frames at target_fps and resize to target_resolution.

    Args:
        video_path: Path to video file.
        target_fps: Desired sampling rate (e.g. 1.0 or 2.0 frames per second).
        target_resolution: (height, width) tuple (default 336x336).

    Returns:
        Tuple of:
        - frames: torch.Tensor of shape [N, 3, H, W] in range [0, 1]
        - timestamps: torch.Tensor of shape [N] in seconds
    """
    vpath = str(video_path)
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file at {vpath}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 25.0

    frame_interval = max(1, int(round(native_fps / target_fps)))

    frames: List[np.ndarray] = []
    timestamps: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # OpenCV reads BGR -> convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to target resolution (width, height)
            resized = cv2.resize(
                frame_rgb,
                (target_resolution[1], target_resolution[0]),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(resized)
            timestamps.append(frame_idx / native_fps)

        frame_idx += 1

    cap.release()

    if not frames:
        # Fallback: empty tensor
        return torch.empty(0, 3, *target_resolution), torch.empty(0)

    # Convert to [N, 3, H, W] float32 tensor in range [0, 1]
    frames_np = np.stack(frames, axis=0)  # [N, H, W, 3]
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
    timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)

    return frames_tensor, timestamps_tensor


class FeatureCacheWriter:
    """Manages writing feature cache into ~500 MB shards with a manifest.

    Resume-safe: loads existing manifest.json on startup and skips previously cached videos.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        shard_size_limit_mb: float = 500.0,
    ) -> None:
        """Initialize FeatureCacheWriter.

        Args:
            cache_dir: Root directory where shards and manifest are stored.
            shard_size_limit_mb: Target size limit per shard in Megabytes (default 500 MB).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size_limit_bytes = int(shard_size_limit_mb * 1024 * 1024)

        self.manifest_file = self.cache_dir / "manifest.json"
        self.manifest: Dict[str, Dict[str, Any]] = {}
        self.cached_video_ids: set[str] = set()

        self._load_manifest()

        # Shard management state
        self.current_shard_idx = len(list(self.cache_dir.glob("shard_*.pt")))
        self.current_shard_data: Dict[str, Dict[str, torch.Tensor]] = {}
        self.current_shard_bytes = 0

    def _load_manifest(self) -> None:
        """Load manifest if present and index already cached video IDs."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                    self.cached_video_ids = set(self.manifest.keys())
                logger.info("Loaded existing manifest with %d cached videos.", len(self.cached_video_ids))
            except Exception as e:
                logger.warning("Could not read existing manifest: %s. Starting fresh.", e)
                self.manifest = {}
                self.cached_video_ids = set()

    def is_cached(self, video_id: str) -> bool:
        """Check if video has already been cached (resume-safe)."""
        return video_id in self.cached_video_ids

    def _flush_current_shard(self) -> None:
        """Save the in-memory shard data to disk and reset buffer."""
        if not self.current_shard_data:
            return

        shard_name = f"shard_{self.current_shard_idx:04d}.pt"
        shard_path = self.cache_dir / shard_name

        logger.info(
            "Flushing shard %s (%d videos, %.2f MB) to disk...",
            shard_name,
            len(self.current_shard_data),
            self.current_shard_bytes / (1024 * 1024),
        )

        torch.save(self.current_shard_data, shard_path)

        # Update manifest file on disk
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

        self.current_shard_idx += 1
        self.current_shard_data = {}
        self.current_shard_bytes = 0

    def add_video(
        self,
        video_id: str,
        frame_feats: torch.Tensor,
        timestamps: torch.Tensor,
        fps: float,
        resolution: Tuple[int, int],
    ) -> None:
        """Add extracted video features to the cache.

        Args:
            video_id: Unique video identifier.
            frame_feats: Features tensor of shape [N, P, D] in fp16.
            timestamps: Timestamp tensor of shape [N].
            fps: Video extraction fps.
            resolution: (height, width) frame resolution.
        """
        if self.is_cached(video_id):
            logger.debug("Video %s already cached. Skipping.", video_id)
            return

        # Ensure features are saved in fp16
        frame_feats = frame_feats.detach().cpu().to(torch.float16)
        timestamps = timestamps.detach().cpu().to(torch.float32)

        feat_bytes = frame_feats.nelement() * frame_feats.element_size()
        time_bytes = timestamps.nelement() * timestamps.element_size()
        video_bytes = feat_bytes + time_bytes

        # If adding this video exceeds the shard size limit, flush current shard first
        if self.current_shard_bytes + video_bytes > self.shard_size_limit_bytes and self.current_shard_data:
            self._flush_current_shard()

        shard_name = f"shard_{self.current_shard_idx:04d}.pt"

        self.current_shard_data[video_id] = {
            "frame_feats": frame_feats,
            "timestamps": timestamps,
        }
        self.current_shard_bytes += video_bytes

        n, p, d = frame_feats.shape
        self.manifest[video_id] = {
            "video_id": video_id,
            "shard_file": shard_name,
            "shape": list(frame_feats.shape),
            "num_frames": n,
            "patches_per_frame": p,
            "feature_dim": d,
            "dtype": "float16",
            "fps": fps,
            "resolution": list(resolution),
            "duration_sec": float(timestamps[-1].item()) if len(timestamps) > 0 else 0.0,
            "size_bytes": video_bytes,
        }
        self.cached_video_ids.add(video_id)

    def close(self) -> None:
        """Flush remaining items and persist final manifest."""
        self._flush_current_shard()
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        logger.info("FeatureCacheWriter closed. Total videos cached: %d", len(self.manifest))

    def get_summary(self) -> Dict[str, Any]:
        """Return cache statistics and size per video."""
        total_videos = len(self.manifest)
        if total_videos == 0:
            return {"total_videos": 0, "total_size_mb": 0.0, "avg_size_kb_per_video": 0.0}

        total_bytes = sum(item["size_bytes"] for item in self.manifest.values())
        avg_bytes = total_bytes / total_videos

        return {
            "total_videos": total_videos,
            "total_shards": self.current_shard_idx + (1 if self.current_shard_data else 0),
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "total_size_gb": round(total_bytes / (1024**3), 3),
            "avg_size_kb_per_video": round(avg_bytes / 1024, 2),
            "avg_size_mb_per_video": round(avg_bytes / (1024 * 1024), 3),
        }

