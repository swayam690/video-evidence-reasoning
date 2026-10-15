"""
describer.py
------------
PURPOSE: Generate a temporal event description for each video segment
         by sending a sequence of representative frames to Qwen2.5-VL
         via OpenRouter API.
"""

import os
import time
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _load_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        raw_bytes = f.read()
    b64_string = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_string}"


def describe_segments(segments: list[dict]) -> list[dict]:
    client = _load_client()
    model_name = os.environ.get("MODEL_NAME", "qwen/qwen-2.5-vl-7b-instruct:free")

    print(f"\n[describer] Using model: {model_name}")
    print(f"[describer] Describing {len(segments)} segments using multi-frame context...\n")

    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        frame_paths = seg.get("frame_paths", [seg["frame_path"]])

        print(f"[describer] Segment {start}s–{end}s: sending {len(frame_paths)} frames to VLM...")

        content_parts = [
            {
                "type": "text",
                "text": (
                    f"You are analyzing a chronological sequence of {len(frame_paths)} frames "
                    f"from a video segment spanning from {start} seconds to {end} seconds."
                ),
            }
        ]

        for idx, fp in enumerate(frame_paths):
            content_parts.append({
                "type": "text",
                "text": f"Frame {idx + 1} of {len(frame_paths)}:"
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": _image_to_base64(fp)}
            })

        prompt_instructions = (
            "\nTask: Describe the EVENT or ACTION occurring across this segment over time.\n"
            "Do NOT simply describe static appearances of a single frame.\n\n"
            "Provide your response in EXACTLY this structured format:\n"
            f"Interval: {start}s–{end}s\n"
            "Summary: <Concise description of what happens during the interval>\n"
            "Visible Entities: <People, objects, or environmental elements visible>\n"
            "Observable Action: <Specific action observed taking place across frames, or 'None / State Only' if no active action is visible>\n\n"
            "STRICT OBSERVATION RULES:\n"
            "1. Do NOT infer unobserved actions. For example, an open door ('state') does NOT imply 'someone opened the door'.\n"
            "2. Distinguish clearly between:\n"
            "   - Object state: e.g., 'door is open'\n"
            "   - Active action: e.g., 'person approaches and pushes door open'\n"
            "   - No observable evidence: e.g., 'no person or interaction visible'"
        )

        content_parts.append({"type": "text", "text": prompt_instructions})

        messages = [{"role": "user", "content": content_parts}]

        description = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=250,
                    temperature=0.2,
                )
                description = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"[describer] Warning: attempt {attempt+1} failed: {e}")
                time.sleep(3 * (attempt + 1))

        if not description:
            description = (
                f"Interval: {start}s–{end}s\n"
                f"Summary: Segment from {start}s to {end}s.\n"
                f"Visible Entities: Unknown.\n"
                f"Observable Action: None / State Only."
            )

        seg["description"] = description
        print(f"[describer] Output:\n{description}\n" + "─" * 40)
        time.sleep(1.0)

    return segments
