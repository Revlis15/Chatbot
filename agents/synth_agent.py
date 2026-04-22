from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from llm import call_openrouter


def _as_text_block(items: List[Dict[str, Any]], *, fields: List[str], prefix: str) -> str:
    lines: List[str] = []
    for i, it in enumerate(items[:3], start=1):
        parts = []
        for f in fields:
            raw = it.get(f)
            v = "" if raw is None else str(raw).strip()
            if v:
                parts.append(f"{f}={v}")
        line = f"{prefix}{i}. " + ("; ".join(parts) if parts else str(it))
        lines.append(line)
    return "\n".join(lines) if lines else f"{prefix}(none)"


def _mock_summarize(query: str, web_results: List[Dict[str, Any]], papers: List[Dict[str, Any]], docs: List[Dict[str, Any]]) -> str:
    # Simple Vietnamese-oriented template, but still usable for other languages.
    web_titles = [w.get("title", "") for w in web_results[:3] if w.get("title")]
    paper_titles = [p.get("title", "") for p in papers[:3] if p.get("title")]
    key_docs = [d.get("text", "")[:160] for d in docs[:3] if d.get("text")]

    lines: List[str] = []
    lines.append(f"## Tóm tắt cho truy vấn\n{query}\n")
    lines.append("## So sánh nhanh")
    lines.append("- **YOLOv8**: thường ưu tiên tốc độ/triển khai real-time; phù hợp edge/streaming; trade-off mAP theo size model.")
    lines.append("- **Faster R-CNN**: 2-stage (RPN + head); hay dùng khi cần baseline mạnh/độ chính xác cao hơn trong một số bài toán; thường chậm hơn.")
    lines.append("")
    lines.append("## Khi nào chọn cái nào?")
    lines.append("- **Chọn YOLOv8** nếu ưu tiên latency/FPS, triển khai nhanh, GPU/CPU hạn chế, pipeline real-time.")
    lines.append("- **Chọn Faster R-CNN** nếu ưu tiên chất lượng (đặc biệt với vật thể nhỏ/khó), chấp nhận inference chậm hơn, xử lý offline/batch.")
    lines.append("")
    if web_titles:
        lines.append("## Nguồn web (top)")
        lines.extend([f"- {t}" for t in web_titles])
        lines.append("")
    if paper_titles:
        lines.append("## Paper liên quan (top)")
        lines.extend([f"- {t}" for t in paper_titles])
        lines.append("")
    if key_docs:
        lines.append("## Gợi ý từ kho tri thức (RAG)")
        lines.extend([f"- {t}..." for t in key_docs])
        lines.append("")
    return "\n".join(lines).strip()


def _fallback_answer() -> str:
    print("[LLM - Fallback]")
    return "Summary: YOLOv8 nhanh hơn, Faster R-CNN chính xác hơn."


def synth_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query: str = str(state.get("query") or "").strip()
    web_results: List[Dict[str, Any]] = state.get("web_results", []) or []
    papers: List[Dict[str, Any]] = state.get("papers", []) or []
    docs: List[Dict[str, Any]] = state.get("docs", []) or []
    context: str = str(state.get("context") or "").strip()

    print("[Synthesizer]")
    if not query:
        return {"answer": _fallback_answer()}
    prompt = "\n".join(
        [
            f"User query: {query}",
            "",
            "Context (compressed):",
            (context or "(none)"),
            "",
            "Web results:",
            _as_text_block(web_results, fields=["title", "content", "url"], prefix="W"),
            "",
            "Paper results:",
            _as_text_block(papers, fields=["title", "abstract", "url"], prefix="P"),
            "",
            "Retrieved docs:",
            _as_text_block(docs, fields=["id", "text", "score"], prefix="D"),
            "",
            "Task: Provide a clean summary and a comparison table-like section (no markdown tables), and practical recommendation.",
            "Language: Prefer Vietnamese if the query is Vietnamese.",
        ]
    )

    llm_answer = call_openrouter(prompt)
    if llm_answer:
        answer = llm_answer.strip()
    else:
        # Demo-safe fallback (required).
        answer = _fallback_answer()
    return {"answer": answer}

