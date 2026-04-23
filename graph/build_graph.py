# build_graph.py
from langgraph.graph import END, StateGraph
from agents.memory_nodes import load_memory_node, memory_rag_node, store_memory_node
from agents.rag_agent import rag_node
from agents.research_agent import research_node
from agents.router import route_node
from agents.synth_agent import synth_node
from graph.state import GraphState

def fast_path_node(state: GraphState):
    # Nút trung gian để log lại việc bỏ qua research
    return {"observations": [{"step": "fast_path", "note": "skipped research"}]}

def build_production_pipeline():
    g = StateGraph(GraphState)
    
    # Định nghĩa các nút
    g.add_node("load_memory", load_memory_node)
    g.add_node("router", route_node)
    g.add_node("fast_path", fast_path_node)
    g.add_node("research", research_node)
    g.add_node("rag_agent", rag_node)
    g.add_node("memory_rag", memory_rag_node)
    g.add_node("synth_agent", synth_node)
    g.add_node("store_memory", store_memory_node)

    # Thiết lập luồng chạy
    g.set_entry_point("load_memory") # Bắt đầu bằng việc load bộ nhớ
    g.add_edge("load_memory", "router")
    
    g.add_conditional_edges(
        "router",
        lambda s: s.get("route"),
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

    return g.compile()