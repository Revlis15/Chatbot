from __future__ import annotations

from typing import Any, Dict, List

from mcp_client.tools import ToolClient


def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip()
    if not query:
        return {"query": query, "web_results": [], "papers": [], "errors": [], "observations": []}
    tc = ToolClient()
    errors: List[Dict[str, Any]] = list(state.get("errors") or [])
    observations: List[Dict[str, Any]] = list(state.get("observations") or [])

    print("[Research]")
    print("[Research] search_web")
    web_tr = tc.search_web(query)
    web_obs = web_tr.data or {}
    web_results: List[Dict[str, Any]] = (web_obs.get("results", []) or [])[:3]
    observations.append({"step": "search_web", "tool": "search_web", "input": query, "output_size": len(web_results), "ok": bool(web_tr.ok)})
    if not web_tr.ok and web_tr.error:
        errors.append({"where": "research_agent", "tool": "search_web", "error": web_tr.error})

    print("[Research] search_paper")
    paper_tr = tc.search_paper(query)
    paper_obs = paper_tr.data or {}
    papers: List[Dict[str, Any]] = (paper_obs.get("results", []) or [])[:3]
    observations.append(
        {"step": "search_paper", "tool": "search_paper", "input": query, "output_size": len(papers), "ok": bool(paper_tr.ok)}
    )
    if not paper_tr.ok and paper_tr.error:
        errors.append({"where": "research_agent", "tool": "search_paper", "error": paper_tr.error})

    return {"query": query, "web_results": web_results, "papers": papers, "errors": errors, "observations": observations}

