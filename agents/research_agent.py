from __future__ import annotations
from typing import Any, Dict, List
from mcp_client.tools import ToolClient

def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Thực thi tìm kiếm tuần tự cho từng sub-query được Planner đề xuất."""
    plan = state.get("plan") or []
    if "research" not in plan:
        return {}

    # Nếu Planner không tạo sub_queries, dùng query gốc làm fallback
    sub_queries = state.get("sub_queries") or [state.get("query")]
    tc = ToolClient()
    
    all_web_results = []
    all_papers = []
    observations = []

    print(f"[Research] Processing {len(sub_queries)} sub-queries...")

    for i, sq in enumerate(sub_queries, start=1):
        print(f" -> Execution {i}/{len(sub_queries)}: {sq}")
        
        # 1. Tìm kiếm Web (cho tin tức, GitHub, blog kỹ thuật)
        try:
            web_tr = tc.search_web(sq)
            if web_tr.ok:
                results = (web_tr.data.get("results") or [])[:2] # Lấy top 2 mỗi query
                all_web_results.extend(results)
        except Exception as e:
            print(f" [!] Web search failed for '{sq}': {e}")

        # 2. Tìm kiếm Paper (ArXiv/Scholar cho các dự án Computer Vision)
        # Chỉ chạy nếu Planner yêu cầu nghiên cứu sâu
        if "search_paper" in plan or i == 1: # Luôn ưu tiên paper cho query đầu tiên
            try:
                paper_tr = tc.search_paper(sq)
                if paper_tr.ok:
                    papers = (paper_tr.data.get("results") or [])[:2]
                    all_papers.extend(papers)
            except Exception as e:
                print(f" [!] Paper search failed for '{sq}': {e}")

    # Loại bỏ trùng lặp nếu cần (Deduplication đơn giản qua URL/Title)
    seen_urls = set()
    unique_web = []
    for r in all_web_results:
        url = r.get("url")
        if url not in seen_urls:
            unique_web.append(r)
            seen_urls.add(url)

    observations.append({
        "step": "multi_query_research",
        "queries_count": len(sub_queries),
        "total_web": len(unique_web),
        "total_papers": len(all_papers)
    })

    return {
        "web_results": unique_web[:5], # Giới hạn tổng số kết quả để tránh tràn context
        "papers": all_papers[:5],
        "observations": observations
    }