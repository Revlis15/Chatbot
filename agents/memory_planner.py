from __future__ import annotations

from typing import Any, Dict


def memory_planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-memory planner:
    - uses history + memory hits + memory_sufficient to adjust plan
    - avoids repeating expensive steps for follow-ups
    """
    query = str(state.get("query") or "").strip()
    mem_quality = float(state.get("memory_quality") or 0.0)
    mem_conflict = bool(state.get("memory_conflict") or False)

    # Strict planning logic (per spec)
    if mem_conflict is True:
        plan = ["research", "rag", "synthesize"]
    elif mem_quality >= 0.75:
        plan = ["rag", "synthesize"]
    elif mem_quality >= 0.6:
        plan = ["hybrid", "rag", "synthesize"]
    else:
        # keep original planner plan
        return {"query": query}

    print("[Planner - SessionMemory]")
    print("Plan:", plan)
    return {"query": query, "plan": plan}

