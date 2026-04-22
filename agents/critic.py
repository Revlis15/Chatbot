from __future__ import annotations

from typing import Any, Dict, List


def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic critic (no LLM) + bounded retry signal.
    - If answer is empty, force a fallback answer.
    - If answer exists but no sources/docs, request one more RAG pass (bounded).
    """
    query = str(state.get("query") or "").strip()
    answer = str(state.get("answer") or "").strip()
    docs: List[Dict[str, Any]] = state.get("docs", []) or []
    web_results: List[Dict[str, Any]] = state.get("web_results", []) or []
    papers: List[Dict[str, Any]] = state.get("papers", []) or []

    errors: List[Dict[str, Any]] = list(state.get("errors") or [])
    observations: List[Dict[str, Any]] = list(state.get("observations") or [])

    retry_count = int(state.get("critic_retry") or 0)

    if not answer:
        # demo-safe hard fallback
        answer = "I couldn't produce a confident answer. Please try a more specific query."
        errors.append({"where": "critic", "error": "empty_answer_fallback"})
        observations.append({"step": "critic", "ok": False, "note": "fallback_answer"})
        return {"query": query, "answer": answer, "critic_route": "end", "errors": errors, "observations": observations}

    has_sources = bool(docs) or bool(web_results) or bool(papers)
    if (not has_sources) and retry_count < 1:
        observations.append({"step": "critic", "ok": True, "note": "retry_rag"})
        return {"query": query, "critic_retry": retry_count + 1, "critic_route": "retry_rag", "errors": errors, "observations": observations}

    observations.append({"step": "critic", "ok": True, "note": "accept"})
    return {"query": query, "critic_route": "end", "errors": errors, "observations": observations}

