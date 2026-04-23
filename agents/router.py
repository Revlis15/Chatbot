# router.py
from __future__ import annotations
from typing import Any, Dict

def route_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip().lower()
    mem_quality = float(state.get("memory_quality") or 0.0)
    history = state.get("history") or []
    memory_hits = state.get("memory_hits") or []

    # Nhận diện đặc điểm của query
    academic_keywords = ["paper", "nghiên cứu", "thuật toán", "yolo", "mot", "ocr", "saccernet"]
    is_academic = any(k in query for k in academic_keywords)
    
    follow_markers = ["tiếp", "như trên", "cái đó", "tại sao", "how", "continue"]
    is_followup = len(query.split()) <= 6 or any(m in query for m in follow_markers)
    has_memory = bool(history or memory_hits)

    # 1. FAST PATH: Ưu tiên bộ nhớ nếu thông tin đã có sẵn và chất lượng cao
    if has_memory and is_followup and mem_quality >= 0.75:
        return {
            "route": "fast_path",
            "plan": ["load_memory", "fast_path", "rag_agent", "synth_agent"],
            "tool_policy": {
                "use_web": False,
                "use_paper": False,
                "use_rag": True,
                "depth": "light"
            },
            "observations": [{"step": "router", "decision": "fast_path"}]
        }

    # 2. RESEARCH PATH: Cần tìm kiếm sâu (ưu tiên Paper nếu là truy vấn kỹ thuật)
    policy = {
        "use_web": True,
        "use_paper": is_academic, # Chỉ tìm paper nếu liên quan đến kỹ thuật/học thuật
        "use_rag": True,
        "depth": "full"
    }

    return {
        "route": "research_path",
        "plan": ["load_memory", "research", "rag_agent", "synth_agent"],
        "tool_policy": policy,
        "observations": [{"step": "router", "decision": "research_path", "is_academic": is_academic}]
    }