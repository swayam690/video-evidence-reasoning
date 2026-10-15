"""
main.py
-------
ENTRY POINT for the Video Evidence-State Reasoning Prototype.

This script ties together all four components:
  1. segmenter  → split video into temporal segments, extract multiple frames per segment
  2. describer  → generate event/action descriptions via Qwen2.5-VL over multi-frame sequences
  3. classifier → classify each segment as S/C/U against the claim based on observable action
  4. reasoner   → use temporal evidence context to answer the claim

USAGE:
  python main.py --video path/to/video.mp4 --claim "The man is opening a door." --num-frames 3
"""

import argparse
import json
import os
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

from src.segmenter  import segment_video
from src.describer  import describe_segments
from src.classifier import classify_segments
from src.reasoner   import reason


def print_banner():
    print("=" * 60)
    print("  Video Evidence-State Reasoning Prototype")
    print("  Event-Level Multi-Frame Temporal Reasoning")
    print("  S=Supported  C=Contradicted  U=Unobserved")
    print("=" * 60)


def print_evidence_table(segments: list[dict], claim: str):
    """Pretty-print the final temporal evidence context."""
    print("\n" + "─" * 60)
    print(f"  CLAIM: \"{claim}\"")
    print("─" * 60)
    for seg in segments:
        time_label = f"{seg['start']}s–{seg['end']}s"
        state = seg.get("state", "?")
        desc  = seg.get("description", "")
        emoji = {"S": "✅", "C": "❌", "U": "❓"}.get(state, "?")
        print(f"  [{time_label}]  STATE: {state} {emoji}")
        # Print description indented
        desc_lines = desc.splitlines()
        for line in desc_lines:
            print(f"    {line}")
        print("─" * 60)


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Video Evidence-State Reasoning Prototype"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file (e.g., video.mp4)"
    )
    parser.add_argument(
        "--claim",
        required=True,
        help="The claim or question to evaluate against the video"
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=3,
        help="Duration of each temporal segment in seconds (default: 3)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=3,
        help="Number of frames to extract per segment (default: 3)"
    )
    parser.add_argument(
        "--output",
        default="output.json",
        help="Path to save the JSON result (default: output.json)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        exit(1)

    print(f"\n  Video:      {args.video}")
    print(f"  Claim:      \"{args.claim}\"")
    print(f"  Segment:    {args.segment_duration}s each")
    print(f"  Frames/Seg: {args.num_frames}\n")

    # Step 1: Segment video and extract multi-frame sequences
    print("STEP 1: Segmenting video and sampling frames...")
    segments = segment_video(
        args.video,
        segment_duration=args.segment_duration,
        num_frames_per_segment=args.num_frames
    )

    # Step 2: Describe event/action per segment
    print("\nSTEP 2: Generating event descriptions via Qwen2.5-VL...")
    segments = describe_segments(segments)

    # Step 3: Classify each segment as S/C/U
    print("\nSTEP 3: Classifying evidence states (S/C/U)...")
    segments = classify_segments(segments, claim=args.claim)

    # Print evidence summary
    print_evidence_table(segments, args.claim)

    # Step 4: Final reasoning
    print("\nSTEP 4: Final temporal reasoning...")
    final_answer = reason(segments, claim=args.claim)

    print("\n" + "=" * 60)
    print("  FINAL ANSWER")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)

    output_data = {
        "claim": args.claim,
        "video": args.video,
        "num_frames_per_segment": args.num_frames,
        "temporal_evidence_context": [
            {
                "start":       seg["start"],
                "end":         seg["end"],
                "description": seg.get("description", ""),
                "state":       seg.get("state", "U"),
            }
            for seg in segments
        ],
        "final_answer": final_answer,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
