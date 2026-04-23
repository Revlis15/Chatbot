from __future__ import annotations

from typing import Any, Dict, List

try:
    # LangGraph canonical import (most versions)
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    # Fallback for older/newer layouts in some environments
    from langgraph.graph.graph import END, StateGraph  # type: ignore

from agents.planner import build_plan
from agents.memory_nodes import load_memory_node, memory_rag_node, store_memory_node
from agents.rag_agent import rag_node
from agents.research_agent import research_node
from agents.router import fast_path_node, route_node
from agents.synth_agent import synth_node
from graph.state import GraphState


def planner_node(state: GraphState) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip()
    plan = build_plan(query)
    print("[Planner]")
    print("Plan:", plan)
    return {"query": query, "plan": plan}


def build_production_pipeline():
    g = StateGraph(GraphState)
    g.add_node("planner", planner_node)
    g.add_node("load_memory", load_memory_node)
    g.add_node("router", route_node)
    g.add_node("fast_path", fast_path_node)
    g.add_node("research", research_node)
    g.add_node("rag_agent", rag_node)
    g.add_node("memory_rag", memory_rag_node)
    g.add_node("synth_agent", synth_node)
    g.add_node("store_memory", store_memory_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "load_memory")
    g.add_edge("load_memory", "router")
    g.add_conditional_edges(
        "router",
        lambda s: str(s.get("route") or "research_path"),
        {
            "fast_path": "fast_path",
            "research_path": "research",
        },
    )
    g.add_edge("fast_path", "rag_agent")
    g.add_edge("research", "rag_agent")
    g.add_edge("rag_agent", "memory_rag")
    g.add_edge("memory_rag", "synth_agent")
    g.add_edge("synth_agent", "store_memory")
    g.add_edge("store_memory", END)

    # NOTE: Some LangGraph versions don't accept compile(recursion_limit=...).
    # Recursion limits (if needed) should be set via invoke/stream config instead.
    return g.compile()


# Backward compatible alias (older callers)
def build_research_graph():
    return build_production_pipeline()

