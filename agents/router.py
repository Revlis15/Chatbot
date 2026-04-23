from __future__ import annotations

from typing import Any, Dict


def route_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic router (no LLM).
    Returns {"route": "fast_path"|"research_path"}.
    """
    query = str(state.get("query") or "").strip()
    q = query.lower()
    session_id = str(state.get("session_id") or "").strip()
    history = state.get("history") or []
    memory_hits = state.get("memory_hits") or []
    mem_quality = float(state.get("memory_quality") or 0.0)

    # Follow-up heuristic: if we have session context and the query is short / referential,
    # prefer fast_path to reuse memory + RAG instead of external research.
    follow_markers = [
        "tiếp",
        "tiếp theo",
        "như trên",
        "như đã nói",
        "cái đó",
        "ý bạn",
        "what about",
        "continue",
        "as above",
        "same as before",
    ]
    is_followup = (len(q.split()) <= 6) or any(m in q for m in follow_markers)
    if session_id and (history or memory_hits) and is_followup and mem_quality >= 0.75:
        return {"query": query, "route": "fast_path"}

    return {"query": query, "route": "research_path"}


def fast_path_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast path: skip web/paper live calls.
    Keep schema stable for downstream nodes.
    """
    query = str(state.get("query") or "").strip()
    return {"query": query, "web_results": [], "papers": []}

