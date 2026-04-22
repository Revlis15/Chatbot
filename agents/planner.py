from __future__ import annotations

from typing import List


def build_plan(query: str) -> List[str]:
    """
    Production-lite planner: returns a structured tool plan.
    """
    _ = query  # reserved for future routing
    return ["search_web", "search_paper", "retrieve", "synthesize"]


def build_plan_memory_aware(
    query: str,
    *,
    history: List[dict] | None = None,
    memory_hits: List[dict] | None = None,
    memory_quality: float = 0.0,
) -> List[str]:
    """
    Deterministic memory-aware planning.
    - If this is a short follow-up and memory is sufficient, skip external search.
    - Otherwise, fall back to the default plan.
    """
    q = str(query or "").strip().lower()
    history = history or []
    memory_hits = memory_hits or []

    follow_markers = [
        "tiếp",
        "tiếp theo",
        "như trên",
        "như đã nói",
        "cái đó",
        "ý bạn",
        "continue",
        "as above",
        "same as before",
        "what about",
    ]
    is_followup = (len(q.split()) <= 6) or any(m in q for m in follow_markers)

    if (history or memory_hits) and is_followup and (float(memory_quality or 0.0) >= 0.75):
        # Prefer reuse: rely on RAG + memory injection, then synthesize.
        return ["retrieve", "synthesize"]

    return build_plan(query)

