from __future__ import annotations
from typing import Any, Dict, List
import json
from llm import call_openrouter
from graph.state import GraphState

def replanner_node(state: GraphState) -> Dict[str, Any]:
    """
    LLM đánh giá kết quả thực thi và quyết định: 
    Cập nhật plan mới hay đi đến bước tổng hợp (Synth).
    """
    query = state.get("query")
    plan = state.get("plan") or []
    observations = state.get("observations") or []
    
    print(f"[Re-planner] Evaluating progress for: {query}")

    prompt = f"""
    Bạn là một người giám sát AI Agent. 
    MỤC TIÊU GỐC: "{query}"
    KẾ HOẠCH ĐÃ CHẠY: {plan}
    KẾT QUẢ ĐÃ THU THẬP: {observations[-2:]} # Lấy 2 quan sát gần nhất

    NHIỆM VỤ:
    Dựa trên kết quả thu được, hãy quyết định xem:
    1. Đã đủ thông tin chưa? Nếu đủ, hãy trả về 'final_answer'.
    2. Nếu thiếu hoặc lỗi, hãy tạo một kế hoạch mới (List các bước) để tiếp tục nghiên cứu.

    TRẢ VỀ JSON:
    {{
      "decision": "continue" hoặc "final_answer",
      "new_plan": ["step_1", "step_2"] (chỉ cần nếu decision là continue)
    }}
    """

    response = call_openrouter(prompt)
    # Giả định có hàm parse JSON như ở file planner.py
    try:
        data = json.loads(response) # Nên dùng Regex như file cũ để an toàn
    except:
        data = {"decision": "final_answer"}

    if data.get("decision") == "final_answer":
        return {"route": "synth_agent"}
    
    return {
        "plan": data.get("new_plan", []),
        "route": "research" # Quay lại bước thực thi
    }