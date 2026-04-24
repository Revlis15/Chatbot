from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from llm import call_openrouter
from graph.state import GraphState

ALLOWED_STEPS = [
    "search_web",    
    "search_paper",  
    "summarize",     
    "rag_agent",     
    "synth_agent"    
]

STEP_DESCRIPTIONS = {
    "search_web": "Use for latest news, GitHub repositories, official documentation, and general benchmarks (e.g., YOLOv12 release dates).",
    "search_paper": "Use for academic papers, ArXiv, deep mathematical explanations, and formal peer-reviewed metrics (e.g., original PD-SORT or OC-SORT papers).",
    "summarize": "Integrate findings from web or papers into knowledge base. ALWAYS follow any search step.",
    "rag_agent": "Check local vector database for previously saved research or private documents.",
    "synth_agent": "Final report generation."
}

descriptions_str = "\n".join([f"- {k}: {v}" for k, v in STEP_DESCRIPTIONS.items()])


def _parse_planner_output(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")

        data = json.loads(match.group())

        plan = data.get("plan", [])
        plan = [s for s in plan if s in ALLOWED_STEPS]

        if not plan:
            plan = ALLOWED_STEPS

        sub_queries = data.get("sub_queries", [])[:3]

        return {
            "plan": plan,
            "sub_queries": sub_queries
        }

    except Exception:
        return {
            "plan": ALLOWED_STEPS,
            "sub_queries": []
        }


def planner_node(state: GraphState) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip()
    memory_quality = float(state.get("memory_quality") or 0.0)

    print(f"[Planner] Processing: {query}")

    prompt = f"""
        You are a Senior Research Architect. Decompose the query into a strategic execution plan.

        USER QUERY: "{query}"
        MEMORY QUALITY: {memory_quality:.2f} (1.0 means we already have the answer in history)

        AVAILABLE STEPS:
        {descriptions_str}

        STRICT PLANNING RULES:
        1. Use 'search_paper' specifically for academic theory, formulas, and official benchmarks (e.g., ArXiv papers for PD-SORT).
        2. Use 'search_web' for latest news, GitHub repos, and recent blog posts (e.g., YOLOv12 release).
        3. Every 'search_web' or 'search_paper' MUST be followed by 'summarize' to process raw data.
        4. If MEMORY QUALITY > 0.8, favor 'rag_agent' -> 'synth_agent' to save time.
        5. The final step MUST ALWAYS be 'synth_agent'.

        OUTPUT JSON FORMAT:
        {{
        "sub_queries": ["query 1", "query 2"],
        "plan": ["step1", "step2", "synth_agent"]
        }}
    """

    llm_response = call_openrouter(prompt)
    print(f"DEBUG LLM RAW: {llm_response}")
    data = _parse_planner_output(llm_response or "")

    return {
        "query": query,
        "plan": data["plan"],
        "sub_queries": data["sub_queries"],
        "observations": [
            {
                "step": "planner",
                "plan": data["plan"],
                "sub_queries": data["sub_queries"]
            }
        ]
    }