from __future__ import annotations

from typing import Any, Dict, List


def _compress_texts(texts: List[str], *, max_chars: int = 900) -> str:
    buf: List[str] = []
    used = 0
    for t in texts:
        s = " ".join(str(t).split())
        if not s:
            continue
        if used + len(s) + 2 > max_chars:
            remaining = max(0, max_chars - used - 3)
            if remaining > 0:
                buf.append(s[:remaining].rstrip() + "…")
            break
        buf.append(s)
        used += len(s) + 2
    return "\n\n".join(buf).strip()


def context_compress_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rule-based context compression to reduce prompt bloat.
    Produces `context: str` derived from top docs.
    """
    docs: List[Dict[str, Any]] = state.get("docs", []) or []
    texts = [str(d.get("text") or "") for d in docs[:3]]
    context = _compress_texts(texts, max_chars=900)
    return {"context": context}

