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
    mem_conflict = bool(state.get("memory_conflict") or False)

    # Follow-up heuristic: if we have session context and the query is short / referential,
    # prefer fast_path to reuse memory + RAG instead of external research.
    follow_markers = [
        "tiếp",
        "tiếp theo",
        "như trên",
        "như đã nói",
        "đó",
        "cái đó",
        "ý bạn",
        "what about",
        "continue",
        "as above",
        "same as before",
        "it",
        "that",
        "those",
    ]
    is_followup = (len(q.split()) <= 6) or any(m in q for m in follow_markers)
    if session_id and (history or memory_hits) and is_followup and mem_quality >= 0.75 and not mem_conflict:
        return {"query": query, "route": "fast_path"}

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
        # If it's explicitly research-heavy, keep research_path unless it's a clear follow-up
        # with existing session context to reuse.
        if mem_conflict:
            return {"query": query, "route": "research_path"}
        if session_id and (history or memory_hits) and is_followup and mem_quality >= 0.75:
            return {"query": query, "route": "fast_path"}
        if session_id and (history or memory_hits) and mem_quality >= 0.45:
            return {"query": query, "route": "hybrid_path"}
        return {"query": query, "route": "research_path"}

    # Non-heavy queries: prefer memory if decent, otherwise hybrid.
    if mem_conflict:
        return {"query": query, "route": "research_path"}
    if session_id and (history or memory_hits) and mem_quality >= 0.45:
        return {"query": query, "route": "hybrid_path"}
    return {"query": query, "route": "fast_path"}


def fast_path_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast path: skip web/paper live calls.
    Keep schema stable for downstream nodes.
    """
    query = str(state.get("query") or "").strip()
    return {"query": query, "web_results": [], "papers": [], "db_papers": []}


def hybrid_path_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hybrid path: use memory + selective search (bounded).
    We don't change `query`; instead we provide:
    - search_query: refined query for tools
    - research_policy: bounded tool selection hints
    """
    query = str(state.get("query") or "").strip()
    memory_hits = state.get("memory_hits") or []

    # Query refinement (best-effort): remove tokens heavily represented in memory hits.
    # This encourages search to focus on "unknown/new" info.
    def _tok(s: str) -> List[str]:
        s2 = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(s or ""))
        return [t for t in s2.split() if len(t) >= 4]

    q_toks = _tok(query)
    mem_text = " ".join(str(h.get("text") or "") for h in memory_hits[:3] if isinstance(h, dict))
    mem_toks = set(_tok(mem_text))
    refined = " ".join([t for t in q_toks if t not in mem_toks]).strip()
    search_query = refined if len(refined) >= 8 else query

    # Selective search: start with web only; allow bounded steps.
    policy = {"max_steps": 2, "tools": ["search_web"]}
    print("[Router] hybrid_path search_query:", search_query)
    return {"query": query, "search_query": search_query, "research_policy": policy}

