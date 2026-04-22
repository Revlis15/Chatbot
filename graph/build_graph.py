from __future__ import annotations

from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from agents.planner import build_plan
from agents.critic import critic_node
from agents.memory_planner import memory_planner_node
from agents.memory_nodes import load_memory_node, memory_rag_node, store_memory_node
from agents.rag_agent import rag_node
from agents.research_agent import research_node
from agents.router import fast_path_node, hybrid_path_node, route_node
from agents.synth_agent import synth_node
from graph.state import GraphState


def planner_node(state: GraphState) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip()
    plan = build_plan(query)
    print("[Planner]")
    print("Plan:", plan)
    return {"query": query, "plan": plan}


def build_research_graph():
    g = StateGraph(GraphState)
    g.add_node("planner", planner_node)
    g.add_node("load_memory", load_memory_node)
    g.add_node("memory_planner", memory_planner_node)
    g.add_node("router", route_node)
    g.add_node("fast_path", fast_path_node)
    g.add_node("hybrid_path", hybrid_path_node)
    g.add_node("research_agent", research_node)
    g.add_node("rag_agent", rag_node)
    g.add_node("memory_rag", memory_rag_node)
    g.add_node("synth_agent", synth_node)
    g.add_node("critic", critic_node)
    g.add_node("store_memory", store_memory_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "load_memory")
    g.add_edge("load_memory", "memory_planner")
    g.add_edge("memory_planner", "router")
    g.add_conditional_edges(
        "router",
        lambda s: str(s.get("route") or "research_path"),
        {
            "fast_path": "fast_path",
            "hybrid_path": "hybrid_path",
            "research_path": "research_agent",
        },
    )
    g.add_edge("fast_path", "rag_agent")
    g.add_edge("hybrid_path", "research_agent")
    g.add_edge("research_agent", "rag_agent")
    g.add_edge("rag_agent", "memory_rag")
    g.add_edge("memory_rag", "synth_agent")
    g.add_edge("synth_agent", "critic")
    g.add_conditional_edges(
        "critic",
        lambda s: str(s.get("critic_route") or "end"),
        {
            "retry_rag": "rag_agent",
            "retry_research": "research_agent",
            "end": "store_memory",
        },
    )
    g.add_edge("store_memory", END)

    return g.compile()

