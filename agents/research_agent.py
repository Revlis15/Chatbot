from __future__ import annotations

from typing import Any, Dict, List
from mcp_client.tools import ToolClient


def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip()
    if not query:
        return {
            "query": query,
            "web_results": [],
            "papers": [],
            "errors": [],
            "observations": [],
        }

    tc = ToolClient()
    policy = state.get("tool_policy") or {}

    errors: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []

    web_results: List[Dict[str, Any]] = []
    papers: List[Dict[str, Any]] = []

    print(f"[Research] policy={policy}")

    # =========================
    # TOOL EXECUTION (WEB)
    # =========================
    if policy.get("use_web", False):
        tr = tc.search_web(query)
        data = tr.data or {}

        web_results = (data.get("results") or [])[:3]

        observations.append({
            "tool": "search_web",
            "output_size": len(web_results),
            "ok": bool(tr.ok),
        })

        if not tr.ok:
            errors.append({
                "tool": "search_web",
                "error": tr.error,
            })

    # =========================
    # TOOL EXECUTION (PAPER)
    # =========================
    if policy.get("use_paper", False):
        tr = tc.search_paper(query)
        data = tr.data or {}

        papers = (data.get("results") or [])[:3]

        observations.append({
            "tool": "search_paper",
            "output_size": len(papers),
            "ok": bool(tr.ok),
        })

        if not tr.ok:
            errors.append({
                "tool": "search_paper",
                "error": tr.error,
            })

    return {
        "query": query,
        "web_results": web_results,
        "papers": papers,
        "errors": errors,
        "observations": observations,
    }