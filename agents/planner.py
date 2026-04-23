from __future__ import annotations
import json
import re
from typing import Any, Dict, List

from llm import call_openrouter
from graph.state import GraphState

VALID_STEPS = ["load_memory", "research", "rag_agent", "synth_agent"]

def _parse_planner_output(text: str) -> Dict[str, Any]:
    """Trích xuất JSON an toàn từ phản hồi của LLM."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            # Chuẩn hóa plan
            plan = [s for s in data.get("plan", []) if s in VALID_STEPS]
            if not plan: plan = ["load_memory", "research", "rag_agent", "synth_agent"]
            return {
                "plan": plan,
                "sub_queries": data.get("sub_queries", [])[:3] # Giới hạn 3 query để tối ưu tốc độ
            }
    except Exception:
        pass
    return {
        "plan": ["load_memory", "research", "rag_agent", "synth_agent"],
        "sub_queries": []
    }

def planner_node(state: GraphState) -> Dict[str, Any]:
    """LLM chia nhỏ bài toán thành các bước thực thi và các câu hỏi phụ chuyên sâu."""
    query = str(state.get("query") or "").strip()
    mem_quality = float(state.get("memory_quality") or 0.0)

    print(f"[Planner] Analyzing technical query: {query}")

    prompt = f"""
    Bạn là một chuyên gia điều phối AI Research Agent. 
    Dựa trên câu hỏi: "{query}"
    
    Nhiệm vụ:
    1. Chia nhỏ câu hỏi thành 2-3 câu hỏi phụ (sub-queries) bằng tiếng Anh để tìm kiếm dữ liệu kỹ thuật chính xác nhất (ví dụ về kiến trúc model, benchmark, dataset).
    2. Lập danh sách các bước thực thi từ: {VALID_STEPS}.

    YÊU CẦU ĐẶC BIỆT:
    - Nếu câu hỏi liên quan đến thuật toán/paper (như YOLO, MOT, PD-SORT), hãy đảm bảo có bước 'research'.
    - Nếu chất lượng bộ nhớ hiện tại ({mem_quality:.2f}) > 0.8, có thể tối giản plan.

    TRẢ VỀ JSON THEO ĐỊNH DẠNG:
    {{
      "sub_queries": ["query 1", "query 2"],
      "plan": ["step 1", "step 2"]
    }}
    """

    llm_response = call_openrouter(prompt)
    data = _parse_planner_output(llm_response or "")
    
    print(f"[Planner] Sub-queries created: {data['sub_queries']}")
    return {
        "plan": data["plan"],
        "sub_queries": data["sub_queries"],
        "observations": [{"step": "planner", "sub_queries": data["sub_queries"]}]
    }