from __future__ import annotations

from typing import Iterable, List


def chunk_text(text: str, *, max_chars: int = 320, overlap_chars: int = 40) -> List[str]:
    """
    Simple, deterministic chunker (rule-based).
    Keeps chunks small for embeddings + retrieval stability.
    """
    s = (text or "").strip()
    if not s:
        return []
    if max_chars <= 0:
        return [s]

    chunks: List[str] = []
    i = 0
    n = len(s)
    step = max(1, max_chars - max(0, overlap_chars))
    while i < n:
        j = min(n, i + max_chars)
        chunk = s[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks
