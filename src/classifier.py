"""
classifier.py
-------------
PURPOSE: Given a claim/question and event-level temporal segment descriptions,
         classify each segment's evidence state as S, C, or U.
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


def _classify_one(client: OpenAI, model_name: str, claim: str, description: str) -> str:
    system_prompt = (
        "You are an evidence state classifier for video reasoning.\n"
        "You evaluate temporal event descriptions against a target CLAIM.\n\n"
        "CLASSIFICATION RULES:\n"
        "  S = Supported    — The description provides clear, observable evidence that the claimed ACTION or EVENT is taking place.\n"
        "                     CRITICAL: For action claims (e.g., 'The man is opening a door'), S requires direct evidence of the person performing the action. An object's static state (e.g. 'the door is open') is NOT sufficient for S.\n\n"
        "  C = Contradicted — The description provides clear observable evidence that directly contradicts the claimed action.\n\n"
        "  U = Unobserved   — The segment does NOT provide sufficient observable evidence to confirm or deny whether the action occurred.\n"
        "                     This includes static object states without visible actions, missing subjects, or unrelated scenes.\n\n"
        "RULES:\n"
        "- Reply with ONLY one letter: S, C, or U.\n"
        "- Do not explain.\n"
        "- When in doubt or if evidence is static/state-only, choose U."
    )

    user_message = (
        f"CLAIM: {claim}\n\n"
        f"TEMPORAL EVENT DESCRIPTION:\n{description}\n\n"
        "Evidence State (S, C, or U):"
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().upper()
            if raw and raw[0] in ("S", "C", "U"):
                return raw[0]
            return "U"
        except Exception as e:
            print(f"[classifier] Warning: attempt {attempt+1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    return "U"


def classify_segments(segments: list[dict], claim: str) -> list[dict]:
    client = _load_client()
    model_name = os.environ.get("MODEL_NAME", "qwen/qwen-2.5-vl-7b-instruct:free")

    print(f"\n[classifier] Claim: \"{claim}\"")
    print(f"[classifier] Classifying {len(segments)} segments...\n")

    for seg in segments:
        state = _classify_one(client, model_name, claim, seg["description"])
        seg["state"] = state
        state_emoji = {"S": "✅", "C": "❌", "U": "❓"}.get(state, "?")
        print(f"[classifier] {seg['start']}s–{seg['end']}s → {state} {state_emoji}")
        time.sleep(0.5)

    return segments
