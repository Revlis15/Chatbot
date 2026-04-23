from __future__ import annotations

import os
from typing import Optional

import requests


def call_openrouter(prompt: str) -> Optional[str]:
    """
    OpenRouter chat completion.
    - Returns None on any failure or if OPENROUTER_API_KEY is missing.
    - Timeout: 15s (demo-safe).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[LLM - OpenRouter]", "missing OPENROUTER_API_KEY")
        return None

    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")
    url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"

    print("[LLM - OpenRouter]", f"model={model}")
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a professional AI Research Assistant. Answer in Vietnamese. Be technical, precise, and structured."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        return content or None
    except Exception as e:
        print("[LLM - OpenRouter]", "failed:", type(e).__name__)
        return None

