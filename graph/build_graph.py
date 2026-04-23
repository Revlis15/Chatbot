from __future__ import annotations
from typing import Any, Dict

from langgraph.graph import END, StateGraph

# Import các node đã được cập nhật theo pattern P&E
from agents.planner import planner_node
from agents.replan_agent import replanner_node
from agents.memory_nodes import load_memory_node, memory_rag_node, store_memory_node
from agents.rag_agent import rag_node
from agents.research_agent import research_node
from agents.synth_agent import synth_node
from graph.state import GraphState

def build_production_pipeline():
    g = StateGraph(GraphState)

    g.add_node("planner", planner_node)
    g.add_node("load_memory", load_memory_node)
    g.add_node("research", research_node)
    g.add_node("rag_agent", rag_node)
    g.add_node("replanner", replanner_node)
    g.add_node("synth_agent", synth_node)
    g.add_node("store_memory", store_memory_node)

    g.set_entry_point("load_memory")
    
    g.add_edge("load_memory", "planner")
    g.add_edge("planner", "research")
    g.add_edge("research", "rag_agent")
    g.add_edge("rag_agent", "replanner")

    g.add_conditional_edges(
        "replanner",
        lambda s: s.get("replan_status"),
        {
            "continue": "research",
            "done": "synth_agent"
        }
    )

    g.add_edge("synth_agent", "store_memory")
    g.add_edge("store_memory", END)

    return g.compile()

# Alias cho các caller cũ nếu cần
def build_research_graph():
    return build_production_pipeline()