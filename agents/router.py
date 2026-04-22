from __future__ import annotations

from typing import Any, Dict


def route_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic router (no LLM).
    Returns {"route": "fast_path"|"research_path"}.
    """
    query = str(state.get("query") or "").strip()
    q = query.lower()

    # Research-heavy intents.
    heavy_markers = [
        "latest",
        "202",
        "benchmark",
        "paper",
        "arxiv",
        "survey",
        "state of the art",
        "sota",
        "compare",
        "so sánh",
        "đánh giá",
        "bài báo",
        "paper",
    ]
    if any(m in q for m in heavy_markers) or len(q.split()) >= 10:
        return {"query": query, "route": "research_path"}

    return {"query": query, "route": "fast_path"}


def fast_path_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast path: skip web/paper live calls.
    Keep schema stable for downstream nodes.
    """
    query = str(state.get("query") or "").strip()
    return {"query": query, "web_results": [], "papers": [], "db_papers": []}

