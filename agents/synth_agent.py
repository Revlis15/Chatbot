from __future__ import annotations

from typing import Any, Dict, List

from llm import call_openrouter


def _fallback_answer() -> str:
    return "Unable to generate a response."


def synth_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip()

    memory_context = str(state.get("memory_context") or "")
    collected_knowledge = str(state.get("collected_knowledge") or "")
    raw_docs = state.get("raw_documents") or []

    print("[Synth] processing final synthesis")

    if not query:
        return {"answer": _fallback_answer(), "synth_failed": True}

    evidence_str = ""
    for i, doc in enumerate(raw_docs[:8], 1): # Lấy top 8 tài liệu chất lượng nhất
        title = doc.get("title") or doc.get("url") or f"Source {i}"
        # Phân biệt Web (content) và Paper (abstract)
        content = doc.get("content") or doc.get("abstract") or doc.get("text") or ""
        evidence_str += f"--- Source {i}: {title} ---\n{content[:1500]}\n\n"

    prompt = f"""
        You are a Senior AI Research Scientist. Generate a professional technical report.

        QUERY:
        {query}

        EXECUTIVE SUMMARY:
        {collected_knowledge}

        INTERNAL CONTEXT (Memory & Local RAG):
        {memory_context}

        DETAILED EVIDENCE (Web & Academic Papers):
        {evidence_str}

        RULES:
        1. Use ONLY provided data to answer.
        2. Prioritize technical benchmarks (mAP, FPS, parameters) for algorithms like YOLO or PD-SORT.
        3. Compare different versions if relevant (e.g., YOLOv10 vs YOLOv11).
        4. If there is a CONFLICT between Web Research and Internal Memory, highlight it clearly.
        5. Cite sources as [Source X].
        6. If a conflict was detected during research (see INTERNAL CONTEXT), explicitly mention the differing values and cite the sources for each.
        7. Provide a 'Confidence Score' for each conflicting data point based on source recency.

        OUTPUT:
        - Comprehensive technical report with structured headings.
        - Final conclusion or recommendation.
        """

    llm_answer = call_openrouter(prompt)
    print(f"DEBUG LLM RAW: {llm_answer}")

    if llm_answer:
        return {
            "answer": llm_answer.strip(),
            "synth_failed": False
        }

    return {
        "answer": _fallback_answer(),
        "synth_failed": True
    }