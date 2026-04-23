from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from llm import call_openrouter
from graph.state import GraphState

ALLOWED_STEPS = [
    "load_memory",
    "research",
    "rag_agent",
    "synth_agent"
]


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
You are a deterministic planning engine for an AI research system.

TASK:
1. Break the user query into 1–3 sub-queries for information retrieval.
2. Create an execution plan using ONLY allowed steps.

QUERY:
{query}

MEMORY QUALITY:
{memory_quality:.2f}

ALLOWED STEPS:
{ALLOWED_STEPS}

RULES:
- Always include "research" for technical/scientific topics
- If memory_quality > 0.8, you may simplify plan
- Do NOT add unknown steps
- Output MUST be valid JSON only

OUTPUT FORMAT:
{{
  "sub_queries": ["..."],
  "plan": ["load_memory", "research", "rag_agent", "synth_agent"]
}}
"""

    llm_response = call_openrouter(prompt)
    data = _parse_planner_output(llm_response or "")

    return {
        "query": query,
        "plan": data["plan"],
        "sub_queries": data["sub_queries"],
        "observations": [
            {
                "step": "planner",
                "sub_queries": data["sub_queries"]
            }
        ]
    }