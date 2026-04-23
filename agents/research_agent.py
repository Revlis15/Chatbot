from __future__ import annotations

from typing import Any, Dict, List
from mcp_client.tools import ToolClient


def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    PURE EXECUTION NODE:
    - chỉ thực thi tool theo sub_queries + tool_policy
    - KHÔNG tự quyết định logic
    """

    sub_queries = state.get("sub_queries") or [state.get("query")]
    plan = state.get("plan") or []

    tc = ToolClient()

    web_results: List[Dict[str, Any]] = []
    papers: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    print(f"[Research] executing {len(sub_queries)} sub-queries")

    use_web = "research" in plan or "search_web" in plan
    use_paper = "search_paper" in plan

    for sq in sub_queries:

        # =====================
        # WEB TOOL
        # =====================
        if use_web:
            try:
                tr = tc.search_web(sq)
                if tr.ok:
                    web_results.extend((tr.data.get("results") or [])[:2])
                else:
                    errors.append({"tool": "web", "query": sq, "error": tr.error})
            except Exception as e:
                errors.append({"tool": "web", "query": sq, "error": str(e)})

        # =====================
        # PAPER TOOL
        # =====================
        if use_paper:
            try:
                tr = tc.search_paper(sq)
                if tr.ok:
                    papers.extend((tr.data.get("results") or [])[:2])
                else:
                    errors.append({"tool": "paper", "query": sq, "error": tr.error})
            except Exception as e:
                errors.append({"tool": "paper", "query": sq, "error": str(e)})

    # =====================
    # SIMPLE OBSERVATION
    # =====================
    observations.append({
        "step": "research_execution",
        "sub_queries": len(sub_queries),
        "web_count": len(web_results),
        "paper_count": len(papers),
    })

    return {
        "tool_results": {
            "web": web_results[:5],
            "paper": papers[:5]
        },
        "observations": observations,
        "errors": errors
    }