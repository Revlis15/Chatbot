from __future__ import annotations
from typing import Any, Dict, List
from mcp_client.tools import ToolClient

def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    RESEARCH EXECUTION NODE (Updated):
    - Thực thi tool với tham số 'include_raw_content' để lấy dữ liệu sâu.
    - Đóng gói 'search_history' (Query + Snippet) cho Replanner.
    - Đóng gói 'raw_documents' (Full content) cho Synth Agent.
    """

    # Ưu tiên sub_queries từ Replanner (vòng lặp) hoặc Planner (vòng đầu)
    sub_queries = state.get("sub_queries") or [state.get("query")]
    plan = state.get("plan") or []

    tc = ToolClient()

    web_results: List[Dict[str, Any]] = []
    papers: List[Dict[str, Any]] = []
    search_logs: List[Dict[str, str]] = [] # Nhật ký: {query, result}
    errors: List[Dict[str, Any]] = []

    print(f"[Research] Executing {len(sub_queries)} sub-queries with Raw Content focus")

    # Xác định tool dựa trên plan
    use_web = any(x in plan for x in ["search_web", "web_search"])
    use_paper = any(x in plan for x in ["search_paper", "paper_research"])
    
    if "research" in plan:
        use_web = True
        use_paper = True

    for sq in sub_queries:
        current_query_results = []

        # =====================
        # WEB TOOL (Tavily focus)
        # =====================
        if use_web:
            try:
                # Kích hoạt lấy nội dung thô (raw_content)
                tr = tc.search_web(sq, include_raw_content=True) 
                if tr.ok:
                    data = (tr.data.get("results") or [])[:2] # Lấy top 2 kết quả chất lượng nhất
                    web_results.extend(data)
                    # Lưu snippet vào nhật ký để Replanner đọc
                    snippets = "\n".join([f"- {r.get('content')}" for r in data])
                    current_query_results.append(snippets)
                else:
                    errors.append({"tool": "web", "query": sq, "error": tr.error})
            except Exception as e:
                errors.append({"tool": "web", "query": sq, "error": str(e)})

        # =====================
        # PAPER TOOL (Arxiv focus)
        # =====================
        if use_paper:
            try:
                tr = tc.search_paper(sq)
                if tr.ok:
                    data = (tr.data.get("results") or [])[:2]
                    papers.extend(data)
                    # Lưu abstract vào nhật ký
                    abstracts = "\n".join([f"- {r.get('title')}: {r.get('abstract')}" for r in data])
                    current_query_results.append(abstracts)
                else:
                    errors.append({"tool": "paper", "query": sq, "error": tr.error})
            except Exception as e:
                errors.append({"tool": "paper", "query": sq, "error": str(e)})

        # Lưu vào nhật ký truy vấn để chống lặp (search_history)
        if current_query_results:
            search_logs.append({
                "query": sq,
                "result": "\n".join(current_query_results)
            })

    # Đóng gói quan sát (observations)
    obs = [{
        "step": "research_execution",
        "sub_queries": len(sub_queries),
        "web_count": len(web_results),
        "paper_count": len(papers),
    }]

    return {
        "search_history": search_logs, 
        "raw_documents": web_results + papers,
        "observations": obs,
        "errors": errors
    }