"""
segmenter.py
------------
PURPOSE: Split an input video into fixed-length temporal segments and extract
         multiple representative frames per segment for event-level analysis.

HOW IT WORKS:
  - We use moviepy to load the video and read its total duration.
  - We slide a window of `segment_duration` seconds across the video.
  - For each window we record start/end times.
  - We extract `num_frames_per_segment` uniformly spaced frames across the interval.
  - Frames are saved as JPEG images and returned as a list of file paths.

OUTPUT: A list of dicts, one per segment:
  [
    {
      "start": 0,
      "end": 3,
      "frame_paths": ["/tmp/.../seg_0_f0.jpg", "/tmp/.../seg_0_f1.jpg", ...],
      "frame_path": "/tmp/.../seg_0_f1.jpg"  # midpoint frame for backward compatibility
    },
    ...
  ]
"""

import os
import tempfile
import numpy as np
from PIL import Image
from moviepy import VideoFileClip


def segment_video(
    video_path: str,
    segment_duration: int = 3,
    num_frames_per_segment: int = 3
) -> list[dict]:
    """
    Split `video_path` into segments of `segment_duration` seconds each
    and extract `num_frames_per_segment` frames per segment.

    Args:
        video_path:             Path to the input video file.
        segment_duration:       Duration of each segment in seconds (default 3).
        num_frames_per_segment: Number of frames to sample per segment (default 3).

    Returns:
        A list of segment dicts with keys: start, end, frame_paths, frame_path.
    """
    video = VideoFileClip(video_path)
    total_duration = video.duration

    print(f"[segmenter] Video duration: {total_duration:.1f}s")
    print(f"[segmenter] Segment size:   {segment_duration}s")
    print(f"[segmenter] Frames/segment: {num_frames_per_segment}")

    frames_dir = tempfile.mkdtemp(prefix="videomodelling_frames_")
    print(f"[segmenter] Saving frames to: {frames_dir}")

    segments = []
    segment_index = 0
    start = 0.0

    while start < total_duration:
        end = min(start + segment_duration, total_duration)

        # Generate sampling timestamps across [start, end]
        if num_frames_per_segment <= 1:
            timestamps = [(start + end) / 2.0]
        else:
            # Uniformly space points between start and slightly before end
            duration_span = max(0.1, end - start)
            timestamps = [
                start + (i + 0.5) * (duration_span / num_frames_per_segment)
                for i in range(num_frames_per_segment)
            ]

        # Clamp timestamps so they are strictly < total_duration
        timestamps = [min(t, max(0.0, total_duration - 0.05)) for t in timestamps]

        frame_paths = []
        for frame_idx, t in enumerate(timestamps):
            frame_array = video.get_frame(t)
            frame_image = Image.fromarray(frame_array.astype("uint8"))

            frame_filename = f"seg_{segment_index}_f{frame_idx}_{t:.1f}s.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            frame_image.save(frame_path, format="JPEG")
            frame_paths.append(frame_path)

        mid_path = frame_paths[len(frame_paths) // 2]

        segments.append({
            "start": int(start),
            "end": int(end),
            "frame_paths": frame_paths,
            "frame_path": mid_path,
        })

        print(
            f"[segmenter] Segment {segment_index}: {int(start)}s–{int(end)}s "
            f"→ Extracted {len(frame_paths)} frames"
        )

        segment_index += 1
        start = end

    video.close()
    print(f"[segmenter] Total segments: {len(segments)}")
    return segments
