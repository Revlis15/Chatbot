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
from agents.summarize_agent import summarize_node

def build_production_pipeline():
    g = StateGraph(GraphState)

    # Đăng ký toàn bộ Node
    g.add_node("load_memory", load_memory_node)
    g.add_node("planner", planner_node)
    g.add_node("research", research_node)
    g.add_node("summarize", summarize_node)
    g.add_node("rag_agent", rag_node)
    g.add_node("memory_rag", memory_rag_node)
    g.add_node("replanner", replanner_node)
    g.add_node("synth_agent", synth_node)
    g.add_node("store_memory", store_memory_node)

    # Thiết lập luồng chạy (Edges)
    g.set_entry_point("load_memory")
    
    g.add_edge("load_memory", "planner")
    
    g.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "research": "research",
            "rag_agent": "rag_agent",
            "summarize": "memory_rag"
        }
    )

    g.add_edge("research", "summarize")

    g.add_conditional_edges(
        "summarize",
        route_after_summarize,
        {
            "rag_agent": "rag_agent",
            "memory_rag": "memory_rag"
        }
    )

    g.add_edge("rag_agent", "memory_rag")
    g.add_edge("memory_rag", "replanner")

    g.add_conditional_edges(
        "replanner",
        route_after_replan,
        {
            "research": "research",
            "rag_agent": "rag_agent",
            "summarize": "summarize",
            "synth_agent": "synth_agent"
        }
    )

    g.add_edge("synth_agent", "store_memory")
    g.add_edge("store_memory", END)

    return g.compile()

def route_after_planner(state: GraphState):
    """Quyết định đi vào Research, RAG hay nhảy thẳng tới Summarize."""
    plan = state.get("plan") or []
    # Nếu kế hoạch có các bước tìm kiếm
    if any(s in plan for s in ["research", "search_web", "search_paper"]):
        return "research"
    # Nếu chỉ cần RAG nội bộ
    if "rag_agent" in plan:
        return "rag_agent"
    return "summarize"

def route_after_summarize(state: GraphState):
    """Quyết định có cần đối soát RAG sau khi đã có dữ liệu Web/Paper không."""
    plan = state.get("plan") or []
    if "rag_agent" in plan:
        return "rag_agent"
    return "memory_rag" # Nhảy thẳng tới trạm gác bộ nhớ

def route_after_replan(state: GraphState):
    status = state.get("replan_status")
    
    # Nếu xong thì đi viết báo cáo
    if status == "done":
        return "synth_agent"

    # Nếu cần tiếp tục, kiểm tra xem bước tiếp theo là gì
    plan = state.get("plan") or []
    if not plan:
        return "synth_agent"

    next_step = plan[0]
    
    # Điều hướng linh hoạt dựa trên kế hoạch mới
    if next_step in ["research", "search_web", "search_paper"]:
        return "research"
    if next_step == "rag_agent":
        return "rag_agent"
    if next_step == "summarize":
        return "summarize"
        
    return "research" # Fallback mặc định