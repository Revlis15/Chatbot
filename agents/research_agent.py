from __future__ import annotations

from typing import Any, Dict, List, Tuple

from mcp_client.tools import ToolClient


def run_react_research(
    query: str,
    *,
    max_steps: int = 4,
    tools: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Bounded ReAct-style research loop (no LLM policy yet).
    Policy (simple heuristics):
    - Prefer web first, then papers.
    - If both are empty after first pass, retry each once (bounded).
    Returns: (web_results, paper_results, errors, observations)
    """
    web_results: List[Dict[str, Any]] = []
    paper_results: List[Dict[str, Any]] = []
    tc = ToolClient()
    errors: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []

    # Bounded attempts: default web then paper, optionally repeat if empty.
    if tools:
        candidates = list(tools)[:max_steps]
    else:
        candidates = ["search_web", "search_paper", "search_web", "search_paper"][:max_steps]
    for step_idx, tool_name in enumerate(candidates, start=1):
        # stop early if we already have something from both channels
        if web_results and paper_results:
            break
        if tool_name == "search_web" and web_results:
            continue
        if tool_name == "search_paper" and paper_results:
            continue

        print("[Research Agent - ReAct]")
        print("Thought:", f"I should call {tool_name} to gather evidence.")
        print("Action:", f"{tool_name}({query})")

        if tool_name == "search_web":
            tr = tc.search_web(query)
        else:
            tr = tc.search_paper(query)

        obs = tr.data or {}
        size = len(obs.get("results", []) or [])
        print("Observation:", f"received {size} results")
        observations.append(
            {
                "step": f"react_{step_idx}",
                "tool": tool_name,
                "input": query,
                "output_size": size,
                "ok": bool(tr.ok),
            }
        )
        if not tr.ok and tr.error:
            errors.append({"where": "research_agent", "tool": tool_name, "error": tr.error})

        if tool_name == "search_web":
            web_results = (obs.get("results", []) or [])[:3]
        elif tool_name == "search_paper":
            paper_results = (obs.get("results", []) or [])[:3]

    return web_results, paper_results, errors, observations


def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip()
    if not query:
        return {"query": query, "web_results": [], "papers": [], "db_papers": [], "errors": [], "observations": []}
    # Optional refined query for tool calls (hybrid path).
    search_query = str(state.get("search_query") or "").strip() or query
    policy = state.get("research_policy") or {}
    max_steps = int(policy.get("max_steps") or 4) if isinstance(policy, dict) else 4
    tools = policy.get("tools") if isinstance(policy, dict) else None
    if tools is not None and not isinstance(tools, list):
        tools = None
    tc = ToolClient()
    errors: List[Dict[str, Any]] = list(state.get("errors") or [])
    observations: List[Dict[str, Any]] = list(state.get("observations") or [])
    # 1) Query cached papers first (demo-safe).
    db_papers: List[Dict[str, Any]] = []
    print("[DB Query]")
    cached_tr = tc.query_papers(query)
    cached = cached_tr.data
    if isinstance(cached, list):
        # Endpoint returns: [{title, content}]
        for it in cached[:3]:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title") or "").strip()
            abstract = str(it.get("content") or "").strip()
            if title:
                db_papers.append({"title": title, "abstract": abstract, "source": "db"})
    if not cached_tr.ok and cached_tr.error:
        errors.append({"where": "research_agent", "tool": "query_papers", "error": cached_tr.error})
    observations.append(
        {
            "step": "db_query_papers",
            "tool": "query_papers",
            "input": query,
            "output_size": len(db_papers),
            "ok": bool(cached_tr.ok),
        }
    )

    # 2) Continue normal research flow.
    web_results, papers, react_errors, react_obs = run_react_research(search_query, max_steps=max_steps, tools=tools)
    errors.extend(react_errors)
    observations.extend(react_obs)

    # 3) Save newly found papers into DB (best-effort).
    if papers:
        print("[DB Save]")
        payload_papers = [
            {
                "title": str(p.get("title") or "").strip(),
                "abstract": str(p.get("abstract") or "").strip(),
                "source": str(p.get("source") or "semantic_scholar").strip(),
            }
            for p in papers
            if isinstance(p, dict) and (p.get("title") or "")
        ]
        save_tr = tc.save_papers(payload_papers)
        observations.append(
            {
                "step": "db_save_papers",
                "tool": "save_papers",
                "input": query,
                "output_size": len(payload_papers),
                "ok": bool(save_tr.ok),
            }
        )
        if not save_tr.ok and save_tr.error:
            errors.append({"where": "research_agent", "tool": "save_papers", "error": save_tr.error})

    # 4) Combine cached + fresh papers, de-dupe by title.
    combined: List[Dict[str, Any]] = []
    seen = set()
    for p in (db_papers + (papers or []))[:10]:
        if not isinstance(p, dict):
            continue
        title = str(p.get("title") or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append(p)

    return {
        "query": query,
        "web_results": web_results,
        "papers": combined,
        "db_papers": db_papers,
        "errors": errors,
        "observations": observations,
    }

