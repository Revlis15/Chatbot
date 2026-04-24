from __future__ import annotations

import re
from typing import Any, Dict
import json
from agents.planner import STEP_DESCRIPTIONS
from llm import call_openrouter
from graph.state import GraphState

def replanner_node(state: GraphState) -> Dict[str, Any]:
    current_iter = state.get("iterations", 0) + 1
    query = state.get("query")
    knowledge = state.get("collected_knowledge", "")
    past_steps = state.get("past_steps") or []
    memory_context = state.get("memory_context", "")

    # Sử dụng trực tiếp STEP_DESCRIPTIONS đã import để tránh redundancy
    from agents.planner import STEP_DESCRIPTIONS
    descriptions_str = "\n".join([f"- {k}: {v}" for k, v in STEP_DESCRIPTIONS.items()])

    prompt = f"""
    You are a Strategic Research Planner. Based on the CURRENT KNOWLEDGE, adjust the research strategy.

    GOAL: "{query}"
    CURRENT KNOWLEDGE: {knowledge}
    PAST STEPS COMPLETED: {past_steps}
    AVAILABLE STEPS: {descriptions_str}
    MEMORY & RAG CONTEXT (Past Research & Local Docs): {memory_context}

    STRATEGIC REPLANNING RULES:
    1. If current data is too general, PIVOT to 'search_paper' for deep technical details.
    2. If you found a specific GitHub repo or tool, add a step to 'search_web' specifically for its documentation.
    3. If there is a conflict in data, add 'rag_agent' to verify with internal logs.
    4. If the goal is fully addressed (the "how" and "what" are clear), set status to "done" and move to 'synth_agent'.

    OUTPUT JSON:
    
    {{
      "status": "continue" | "done",
      "strategy_update": "Why are we changing or maintaining the plan?",
      "new_remaining_plan": ["step1", "step2"],
      "new_sub_queries": ["specific targeted query"]
    }}
    """

    response = call_openrouter(prompt)
    clean_text = response or "" 
    match = re.search(r"\{.*\}", clean_text, re.DOTALL)
    
    if not match:
        return {"replan_status": "done", "iterations": current_iter} 

    try:
        data = json.loads(match.group())
    except Exception:
        return {"replan_status": "done", "iterations": current_iter}

    status = data.get("status", "done")

    # Safety exit for iterations
    if status == "done" or current_iter >= 3:
        return {"replan_status": "done", "iterations": current_iter}

    # Cập nhật past_steps dựa trên bước đầu tiên của plan cũ vừa thực hiện
    last_plan = state.get("plan") or []
    step_just_finished = [last_plan[0]] if last_plan else []

    return {
        "replan_status": status,
        "iterations": current_iter,
        "plan": data.get("new_remaining_plan", []),
        "sub_queries": data.get("new_sub_queries", []), # QUAN TRỌNG: Để research_node vòng sau sử dụng
        "past_steps": step_just_finished # Sẽ được operator.add cộng dồn vào state
    }