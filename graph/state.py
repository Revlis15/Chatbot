from __future__ import annotations

from typing import Annotated, TypedDict, Dict, Any, List
import operator


class GraphState(TypedDict, total=False):
    query: str
    session_id: str | None              

    memory_conflict: bool
    conflict_notes: str

    plan: list[str] 
    sub_queries: list[str]
    past_steps: Annotated[List[str], operator.add]
    iterations: int
    replan_status: str # "continue" | "done"
    
    search_history: Annotated[List[Dict[str, str]], operator.add]
    collected_knowledge: str
    raw_documents: Annotated[List[Dict[str, Any]], operator.add]

    history: list[dict]
    memory_hits: list[dict]
    memory_context: str
    memory_quality: float

    tool_policy: dict
    tool_results: dict

    answer: str
    synth_failed: bool

    observations: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]