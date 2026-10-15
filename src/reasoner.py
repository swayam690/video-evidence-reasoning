"""
reasoner.py
-----------
PURPOSE: Given the full temporal event context (all segments with event descriptions
         and S/C/U states), generate a final reasoned answer to the user's claim.

OUTPUT: Structured final verdict and explanation.
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _load_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _format_context(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        state = seg.get("state", "U")
        desc  = seg.get("description", "No description.")
        start = seg["start"]
        end   = seg["end"]
        lines.append(f"── Segment [{start}s–{end}s] | State: {state} ──\n{desc}\n")
    return "\n".join(lines)


def reason(segments: list[dict], claim: str) -> str:
    client = _load_client()
    model_name = os.environ.get("MODEL_NAME", "qwen/qwen-2.5-vl-7b-instruct:free")

    evidence_context = _format_context(segments)

    system_prompt = (
        "You are a video temporal reasoning assistant.\n"
        "You have analyzed a video across temporal segments. For each segment you have an event-level observation and an evidence state label:\n"
        "  S = segment contains observable evidence supporting the claim/action\n"
        "  C = segment contains observable evidence contradicting the claim/action\n"
        "  U = segment is unobserved or lacks observable evidence for the action\n\n"
        "Read all segment observations in temporal order and provide a final verdict:\n"
        "  VERDICT: [TRUE / FALSE / UNCERTAIN]\n"
        "  REASONING: [2–4 concise sentences explaining your decision based strictly on observable actions across the sequence]"
    )

    user_message = (
        f"CLAIM: \"{claim}\"\n\n"
        f"TEMPORAL EVIDENCE CONTEXT:\n{evidence_context}\n"
        "Provide your final verdict and reasoning:"
    )

    print(f"\n[reasoner] Sending temporal evidence context to LLM for final reasoning...")

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=300,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[reasoner] Warning: attempt {attempt+1} failed: {e}")
            time.sleep(3 * (attempt + 1))

    return "VERDICT: UNCERTAIN\nREASONING: Model call failed."
