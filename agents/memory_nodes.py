from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import memory_store
import session_manager


def _truncate_messages(history: List[Dict[str, Any]], *, max_items: int = 5, max_chars_each: int = 400) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in (history or [])[-max_items:]:
        role = str(m.get("role") or "").strip() or "user"
        content = str(m.get("content") or "").strip()
        if len(content) > max_chars_each:
            content = content[: max_chars_each - 1].rstrip() + "…"
        if content:
            out.append({"role": role, "content": content})
    return out


def _build_search_query(query: str, history: List[Dict[str, Any]]) -> str:
    q = str(query or "").strip()
    if not q:
        return ""
    last_user_messages = [
        str(m.get("content") or "").strip() for m in (history or []) if str(m.get("role") or "").strip() == "user"
    ]
    last_user_messages = [m for m in last_user_messages if m][-3:]
    suffix = " ".join(last_user_messages).strip()
    return f"{q} {suffix}".strip()


def _dedupe_hits(hits: List[Dict[str, Any]], *, limit: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for h in hits or []:
        if not isinstance(h, dict):
            continue
        txt = str(h.get("text") or "").strip()
        key = " ".join(txt.lower().split())[:240]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= limit:
            break
    return out


def _avg_score(hits: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for h in hits or []:
        if not isinstance(h, dict):
            continue
        s = h.get("score")
        if isinstance(s, (int, float)):
            vals.append(float(s))
    if not vals:
        return 0.0
    return sum(vals) / float(len(vals))


def _score_variance(hits: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for h in hits or []:
        if not isinstance(h, dict):
            continue
        s = h.get("score")
        if isinstance(s, (int, float)):
            vals.append(float(s))
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / float(len(vals))
    return sum((x - mean) ** 2 for x in vals) / float(len(vals))


def _recency_weight(hits: List[Dict[str, Any]]) -> float:
    """
    recency_weight = exp(-time_decay)
    time_decay is derived from average age relative to a time constant.
    """
    now = float(time.time())
    ages: List[float] = []
    for h in hits or []:
        if not isinstance(h, dict):
            continue
        md = h.get("metadata") or {}
        if not isinstance(md, dict):
            continue
        created_at = md.get("created_at")
        # Expect UNIX seconds if present.
        if isinstance(created_at, (int, float)) and float(created_at) > 0:
            ages.append(max(0.0, now - float(created_at)))
    if not ages:
        return 1.0
    avg_age = sum(ages) / float(len(ages))
    tau_s = float(7 * 24 * 3600)
    time_decay = float(avg_age) / float(tau_s)
    return float(math.exp(-time_decay))


def _detect_conflict(hits: List[Dict[str, Any]]) -> bool:
    # simple heuristic:
    # - contradictory keywords across hits, OR
    # - low similarity variance with multiple hits (near-duplicate / unstable signal)
    texts = [str(h.get("text") or "").lower() for h in (hits or [])[:5] if isinstance(h, dict)]
    if len(texts) < 2:
        return False
    if len(hits) >= 2 and _score_variance(hits) < 0.0005:
        return True
    neg_markers = ["không", "not ", "never", "no "]
    a = texts[0]
    for b in texts[1:]:
        a_neg = any(m in a for m in neg_markers)
        b_neg = any(m in b for m in neg_markers)
        if a_neg != b_neg:
            # overlap check (very lightweight)
            shared = set(a.split()) & set(b.split())
            if len(shared) >= 5:
                return True
    return False


def _derive_quality(score: float) -> str:
    if score > 0.75:
        return "high"
    if score > 0.5:
        return "medium"
    return "low"


def _memory_sufficient(quality: str, coverage: int) -> bool:
    if quality == "high":
        return True
    if quality == "medium" and coverage >= 3:
        return True
    return False


def format_memory_context(
    *,
    history: List[Dict[str, Any]],
    memory_hits: List[Dict[str, Any]],
    docs: List[Dict[str, Any]],
) -> str:
    """
    Structured memory-aware context formatting for LLM.
    """
    lines: List[str] = []

    lines.append("[Conversation History]")
    if history:
        for m in history[-5:]:
            lines.append(f"- {m.get('role')}: {m.get('content')}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("[Relevant Past Knowledge]")
    if memory_hits:
        for i, h in enumerate(memory_hits[:5], start=1):
            txt = str(h.get("text") or "").strip()
            score = h.get("score")
            score_s = f"{float(score):.3f}" if isinstance(score, (int, float)) else ""
            if txt:
                lines.append(f"- M{i} (score={score_s}): {txt}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("[Retrieved Documents]")
    doc_texts = [str(d.get("text") or "").strip() for d in (docs or [])[:3] if isinstance(d, dict)]
    if doc_texts:
        for i, t in enumerate(doc_texts, start=1):
            if t:
                lines.append(f"- D{i}: {t}")
    else:
        lines.append("(none)")

    return "\n".join(lines).strip()


def _first_two_sentences(text: str) -> str:
    s = " ".join(str(text or "").split()).strip()
    if not s:
        return ""
    sentences: List[str] = []
    buf = ""
    for ch in s:
        buf += ch
        if ch in ".!?":
            seg = buf.strip()
            if seg:
                sentences.append(seg)
            buf = ""
            if len(sentences) >= 2:
                break
    if len(sentences) >= 2:
        return " ".join(sentences).strip()
    if sentences:
        return sentences[0].strip()
    return s


def _is_greeting(query: str) -> bool:
    q = " ".join(str(query or "").lower().split())
    return q in {"hi", "hello", "hey", "xin chào", "chào"}


def _importance(text: str) -> float:
    return float(memory_store.compute_importance(text))


def _classify(text: str) -> str:
    return str(memory_store.classify_memory(text))


def _usage_score(usage_count: Any) -> float:
    try:
        n = int(usage_count) if usage_count is not None else 0
    except Exception:
        n = 0
    return float(math.log(1.0 + max(0, n)))


def _recency(now_s: float, last_used_s: Any, *, tau_s: float) -> float:
    try:
        ts = float(last_used_s) if last_used_s is not None else 0.0
    except Exception:
        ts = 0.0
    if ts <= 0:
        return 0.0
    dt = max(0.0, now_s - ts)
    return float(math.exp(-dt / float(tau_s)))


def _final_score(*, similarity: float, recency: float, importance: float, usage_score: float) -> float:
    return float((0.4 * similarity) + (0.2 * recency) + (0.3 * importance) + (0.1 * usage_score))


def load_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load short-term history + long-term memory hits.
    Demo-safe: failures go to state["errors"] (no raise).
    """
    query = str(state.get("query") or "").strip()
    session_id = str(state.get("session_id") or "").strip()

    # Normalize to required contract types, but stay backward-compatible with existing state shapes.
    errors_raw = list(state.get("errors") or [])
    errors: List[str] = [str(e) for e in errors_raw]
    observations: List[Dict[str, Any]] = list(state.get("observations") or [])

    if not session_id:
        return {
            "query": query,
            "session_id": None,
            "history": [],
            "memory_hits": [],
            "memory_topk": 0,
            "memory_coverage": 0,
            "memory_quality": 0.0,
            "memory_conflict": False,
            "memory_sufficient": False,
            "errors": errors,
            "observations": observations,
        }

    # Short-term memory (chat history)
    try:
        hist = session_manager.load_recent_history(session_id, k=5)
        history = _truncate_messages(hist, max_items=5, max_chars_each=400)
        print("[Memory] Loaded history:", len(history))
        observations.append({"step": "memory_load_history", "ok": True, "note": f"n={len(history)}"})
    except Exception as e:
        history = []
        errors.append(f"history_failed:{type(e).__name__}")
        observations.append({"step": "memory_load_history", "ok": False, "note": "failed"})

    # Long-term memory (vector search)
    try:
        now_s = float(time.time())
        tau_s = float(7 * 24 * 3600)
        search_q = _build_search_query(query, history)
        raw_hits = memory_store.search_memory(session_id=session_id, query=search_q, k=5)
        hits0 = _dedupe_hits(raw_hits, limit=5)[:5]

        scored: List[Dict[str, Any]] = []
        for h in hits0:
            md = h.get("metadata") or {}
            if not isinstance(md, dict):
                md = {}
            similarity = float(h.get("score") or 0.0)
            last_used = md.get("last_used") if "last_used" in md else md.get("created_at")
            rec = _recency(now_s, last_used, tau_s=tau_s)
            imp = float(md.get("importance")) if isinstance(md.get("importance"), (int, float)) else _importance(str(h.get("text") or ""))
            use = _usage_score(md.get("usage_count"))
            fs = _final_score(similarity=similarity, recency=rec, importance=imp, usage_score=use)

            md2 = dict(md)
            md2.setdefault("importance", imp)
            md2.setdefault("usage_count", int(md2.get("usage_count") or 0) if str(md2.get("usage_count") or "").strip() else 0)
            md2.setdefault("created_at", int(md2.get("created_at") or now_s))
            md2.setdefault("last_used", int(md2.get("last_used") or md2.get("created_at") or now_s))
            md2.setdefault("type", _classify(str(h.get("text") or "")))

            scored.append({"text": str(h.get("text") or ""), "score": similarity, "final_score": fs, "metadata": md2})

        scored.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
        topk = int(min(5, max(3, len(scored)))) if scored else 0
        hits = scored[:topk] if topk > 0 else []

        coverage = int(len(hits))
        mem_quality = float(_avg_score([{"score": float(h.get("final_score") or 0.0)} for h in hits])) if hits else 0.0
        conflict = _detect_conflict([{"text": h.get("text"), "score": h.get("final_score")} for h in hits])
        sufficient = bool(mem_quality > 0.6)

        print("[Memory] Retrieved hits:", len(hits), "quality=", f"{mem_quality:.3f}", "coverage=", coverage, "conflict=", conflict)
        observations.append(
            {
                "step": "memory_search",
                "ok": True,
                "note": f"hits={len(hits)} quality={mem_quality:.3f} coverage={coverage} conflict={conflict} sufficient={sufficient}",
            }
        )
    except Exception as e:
        hits = []
        mem_quality = 0.0
        coverage = 0
        conflict = False
        sufficient = False
        errors.append(f"memory_search_failed:{type(e).__name__}")
        observations.append({"step": "memory_search", "ok": False, "note": "failed"})

    return {
        "query": query,
        "session_id": session_id,
        "history": history,
        "memory_hits": hits,
        "memory_topk": int(len(hits)),
        "memory_coverage": int(coverage),
        "memory_quality": float(mem_quality),
        "memory_conflict": bool(conflict),
        "memory_sufficient": bool(sufficient),
        "errors": errors,
        "observations": observations,
    }


def memory_rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine memory_context + retrieved docs into unified `context`.
    Keeps output deterministic (rule-based).
    """
    query = str(state.get("query") or "").strip()
    history: List[Dict[str, Any]] = state.get("history") or []
    memory_hits: List[Dict[str, Any]] = state.get("memory_hits") or []
    docs: List[Dict[str, Any]] = state.get("docs", []) or []
    errors_raw = list(state.get("errors") or [])
    errors: List[str] = [str(e) for e in errors_raw]
    observations: List[Dict[str, Any]] = list(state.get("observations") or [])

    context = format_memory_context(history=history, memory_hits=memory_hits[:5], docs=docs)
    print("[Memory] Used in context:", bool(history or memory_hits))
    observations.append({"step": "memory_rag", "ok": True, "note": f"ctx_chars={len(context)}"})
    return {"query": query, "context": context, "errors": errors, "observations": observations}


def store_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store meaningful messages:
    - append user query + assistant answer into SQLite session history
    - add long-term memory (vector DB) for session_id
    """
    query = str(state.get("query") or "").strip()
    session_id = str(state.get("session_id") or "").strip()
    answer = str(state.get("answer") or "").strip()
    pattern = str(state.get("pattern") or "").strip()

    errors_raw = list(state.get("errors") or [])
    errors: List[str] = [str(e) for e in errors_raw]
    observations: List[Dict[str, Any]] = list(state.get("observations") or [])

    if not session_id:
        observations.append({"step": "memory_store", "ok": True, "note": "no_session"})
        return {"query": query, "errors": errors, "observations": observations}

    # Save short-term messages (SQLite)
    try:
        session_manager.save_message(session_id, "user", query)
        if answer:
            session_manager.save_message(session_id, "assistant", answer)
        observations.append({"step": "memory_write_history", "ok": True, "note": f"pattern={pattern or 'n/a'}"})
    except Exception as e:
        errors.append(f"history_write_failed:{type(e).__name__}")
        observations.append({"step": "memory_write_history", "ok": False, "note": "failed"})

    # Save long-term memory (Chroma) ONLY if:
    # - len(answer) > 100 chars
    # - not greeting
    # - not empty
    try:
        now_i = int(time.time())
        should_store = bool(answer) and (len(answer) > 100) and (not _is_greeting(query))
        if not should_store:
            observations.append({"step": "memory_write_vector", "ok": True, "note": "skipped"})
        else:
            summary = _first_two_sentences(answer)
            payload = f"{answer}\n\nSUMMARY: {summary}".strip()
            md = {
                "summary": summary,
                "pattern": pattern or "planner",
                "importance": _importance(payload),
                "usage_count": 0,
                "created_at": now_i,
                "last_used": now_i,
                "type": _classify(payload),
            }
            memory_store.add_memory(session_id=session_id, text=payload, metadata=md)
            observations.append({"step": "memory_write_vector", "ok": True, "note": "stored"})

        # Usage tracking: touch memories that were used (top-k injected into context).
        try:
            hits = state.get("memory_hits") or []
            if isinstance(hits, list) and hits:
                ms = memory_store.MemoryStore()
                used_any = 0
                for h in hits[:5]:
                    if not isinstance(h, dict):
                        continue
                    mdh = h.get("metadata") or {}
                    if not isinstance(mdh, dict):
                        continue
                    mid = str(mdh.get("_id") or "").strip()
                    if not mid:
                        continue
                    if ms.update_memory_usage(session_id=session_id, memory_id=mid, used_at=now_i):
                        used_any += 1
                observations.append({"step": "memory_usage_update", "ok": True, "note": f"touched={used_any}"})
        except Exception:
            observations.append({"step": "memory_usage_update", "ok": False, "note": "failed"})
    except Exception as e:
        errors.append(f"vector_write_failed:{type(e).__name__}")
        observations.append({"step": "memory_write_vector", "ok": False, "note": "failed"})

    return {"query": query, "errors": errors, "observations": observations}

