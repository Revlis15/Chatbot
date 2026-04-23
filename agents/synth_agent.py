from __future__ import annotations

from typing import Any, Dict, List

from llm import call_openrouter


def _fallback_answer() -> str:
    return "Unable to generate a response."


def synth_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip()

    tool_results = state.get("tool_results") or {}
    observations = state.get("observations") or []
    memory_context = str(state.get("memory_context") or "")

    print("[Synth] processing final synthesis")

    if not query:
        return {"answer": _fallback_answer(), "synth_failed": True}

    web_results = tool_results.get("web", [])
    paper_results = tool_results.get("paper", [])

    prompt = f"""
You are a synthesis engine for an AI research system.

TASK:
Generate a final high-quality answer based on provided evidence.

QUERY:
{query}

MEMORY CONTEXT:
{memory_context or "(none)"}

WEB RESULTS:
{web_results[:3]}

PAPER RESULTS:
{paper_results[:3]}

RULES:
- Use ONLY provided data
- Do not invent facts
- Combine memory + tool results intelligently
- If conflicting info exists, mention uncertainty
- Prefer concise structured explanation
- Include comparison if relevant

OUTPUT:
- Clear answer
- Optional bullet insights
"""

    llm_answer = call_openrouter(prompt)

    if llm_answer:
        return {
            "answer": llm_answer.strip(),
            "synth_failed": False
        }

    return {
        "answer": _fallback_answer(),
        "synth_failed": True
    }