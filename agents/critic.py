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
    mem_quality = float(state.get("memory_quality") or 0.0)
    mem_conflict = bool(state.get("memory_conflict") or False)
    memory_hits: List[Dict[str, Any]] = state.get("memory_hits") or []

    if not answer:
        # demo-safe hard fallback
        answer = "I couldn't produce a confident answer. Please try a more specific query."
        errors.append({"where": "critic", "error": "empty_answer_fallback"})
        observations.append({"step": "critic", "ok": False, "note": "fallback_answer"})
        return {"query": query, "answer": answer, "critic_route": "end", "errors": errors, "observations": observations}

    # If memory is contradictory, force research once (bounded).
    if mem_conflict and retry_count < 1:
        errors.append({"where": "critic", "error": "memory_conflict_force_research"})
        observations.append({"step": "critic", "ok": True, "note": "retry_research_due_to_conflict"})
        return {"query": query, "critic_retry": retry_count + 1, "critic_route": "retry_research", "errors": errors, "observations": observations}

    # Memory-aware consistency guardrail (bounded):
    # If we have confident memory but the answer looks like a refusal/low-signal, retry once.
    low_signal_markers = [
        "i don't know",
        "i do not know",
        "not sure",
        "can't",
        "cannot",
        "không biết",
        "khó trả lời",
        "không chắc",
    ]
    a_low = answer.lower()
    looks_low_signal = (len(answer) < 80) or any(m in a_low for m in low_signal_markers)
    if memory_hits and looks_low_signal and retry_count < 1:
        # Low confidence: prefer fetching more evidence (research).
        if mem_quality < 0.45:
            errors.append({"where": "critic", "error": "low_confidence_retry_research"})
            observations.append({"step": "critic", "ok": True, "note": "retry_research_low_conf"})
            return {"query": query, "critic_retry": retry_count + 1, "critic_route": "retry_research", "errors": errors, "observations": observations}
        # Medium/high: re-run RAG once to enrich context (cheaper than full research).
        errors.append({"where": "critic", "error": "memory_inconsistency_retry"})
        observations.append({"step": "critic", "ok": True, "note": "retry_rag_due_to_memory"})
        return {"query": query, "critic_retry": retry_count + 1, "critic_route": "retry_rag", "errors": errors, "observations": observations}

    has_sources = bool(docs) or bool(web_results) or bool(papers)
    if (not has_sources) and retry_count < 1:
        observations.append({"step": "critic", "ok": True, "note": "retry_research_no_sources"})
        return {"query": query, "critic_retry": retry_count + 1, "critic_route": "retry_research", "errors": errors, "observations": observations}

    observations.append({"step": "critic", "ok": True, "note": "accept"})
    return {"query": query, "critic_route": "end", "errors": errors, "observations": observations}

