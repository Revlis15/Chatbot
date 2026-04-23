from __future__ import annotations

from typing import Any, Dict
import json
from llm import call_openrouter
from graph.state import GraphState


def replanner_node(state: GraphState) -> Dict[str, Any]:
    query = state.get("query")
    plan = state.get("plan") or []
    observations = state.get("observations") or []

    print(f"[Replanner] Evaluating: {query}")

    safe_observations = [
        str(obs) for obs in observations[-2:]
    ]

    prompt = f"""
You are a planning controller for an AI agent system.

TASK:
Evaluate whether the current execution plan is sufficient.

INPUT:
- query: {query}
- plan: {plan}
- observations: {safe_observations}

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "status": "continue" | "done",
  "updated_plan": ["step1", "step2"],
  "reason": "short explanation"
}}

RULES:
- If enough information → status = "done"
- If not enough → status = "continue"
- Always return valid JSON only
- Do not include extra text
"""

    response = call_openrouter(prompt)

    try:
        data = json.loads(response)
    except Exception:
        return {
            "replan_status": "done",
            "plan": plan
        }

    status = data.get("status", "done")

    if status == "done":
        return {
            "replan_status": "done"
        }

    return {
        "replan_status": "continue",
        "plan": data.get("updated_plan", plan)
    }