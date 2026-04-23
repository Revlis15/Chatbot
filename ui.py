from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st

import ui_graph


def _mcp_url() -> str:
    return os.getenv("MCP_URL", "http://localhost:8001").rstrip("/")


def _post_run(query: str) -> Dict[str, Any]:
    url = f"{_mcp_url()}/run"
    resp = requests.post(url, json={"q": query}, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _post_run_with_session(query: str, session_id: str | None) -> Dict[str, Any]:
    url = f"{_mcp_url()}/run"
    payload: Dict[str, Any] = {"q": query}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _post_run_with_session_and_pattern(query: str, session_id: str | None, pattern: str) -> Dict[str, Any]:
    url = f"{_mcp_url()}/run"
    payload: Dict[str, Any] = {"q": query, "pattern": str(pattern or "planner").strip().lower()}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _post_run_compare(query: str, patterns: List[str]) -> Dict[str, Any]:
    url = f"{_mcp_url()}/run_compare"
    resp = requests.post(url, json={"query": query, "patterns": patterns}, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _post_run_compare_with_session(query: str, patterns: List[str], session_id: str | None) -> Dict[str, Any]:
    url = f"{_mcp_url()}/run_compare"
    payload: Dict[str, Any] = {"query": query, "patterns": patterns}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _get_history() -> List[Dict[str, Any]]:
    url = f"{_mcp_url()}/history"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def _truncate(text: str, n: int = 300) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"


st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")

tab_compare, tab_graph = st.tabs(["Compare", "Graph Viewer"])

with tab_compare:
    query = st.text_input("Query", value="So sánh YOLOv8 và Faster R-CNN mới nhất")
    session_id = st.text_input("session_id (optional)", value="demo-session-1")
    patterns = st.multiselect(
        "Patterns",
        options=["react", "planner", "rewoo"],
        default=["planner", "react", "rewoo"],
    )
    run = st.button("Run", type="primary")

    if run:
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            # If user selects nothing, default to planner.
            patterns = patterns if patterns else ["planner"]
            with st.spinner("Running patterns..."):
                try:
                    sid = session_id.strip() or None
                    if len(patterns) == 1:
                        # Fast path: single pattern uses /run (graph pattern is handled server-side)
                        single = _post_run_with_session_and_pattern(query.strip(), sid, patterns[0])
                        data = {"query": query.strip(), "results": {patterns[0]: single}}
                    else:
                        data = _post_run_compare_with_session(query.strip(), patterns, sid)
                except Exception as e:
                    st.error(f"API call failed: {type(e).__name__}: {e}")
                    st.stop()

            if not isinstance(data, dict) or not data:
                st.warning("Empty response from backend.")
                st.stop()

            results: Dict[str, Any] = data.get("results") or {}
            if not isinstance(results, dict) or not results:
                st.warning("No results returned.")
                st.stop()

            st.divider()

            st.subheader("🔎 Comparison")
            cols = st.columns(len(patterns))
            for idx, pattern in enumerate(patterns):
                r = results.get(pattern) or {}
                with cols[idx]:
                    st.markdown(f"### {pattern}")
                    st.subheader("🧠 Plan")
                    plan = r.get("plan") or []
                    st.write(plan if plan else "(none)")

                    st.subheader("✅ Answer")
                    ans = str(r.get("answer") or "").strip()
                    if ans:
                        st.success(ans)
                    else:
                        st.warning("No answer returned.")

                    st.subheader("🧾 Logs")
                    logs = str(r.get("logs") or "")
                    if logs.strip():
                        st.code(logs, language="text")
                    else:
                        st.info("No logs.")

            st.divider()
            st.subheader("🕘 History (last 20)")
            try:
                hist = _get_history()
                if hist:
                    for item in hist[:20]:
                        q = str(item.get("query") or "")
                        p = str(item.get("pattern") or "")
                        t = str(item.get("time") or "")
                        a = _truncate(str(item.get("answer") or ""), 180)
                        st.write(f"- [{t}] **{p}** — {q} — {a}")
                else:
                    st.info("No history yet.")
            except Exception as e:
                st.error(f"Failed to load history: {type(e).__name__}: {e}")

with tab_graph:
    st.subheader("Graph visualization + execution viewer")
    mode = st.radio("Mode", ["Live", "Replay"], horizontal=True)
    q = st.text_input("Execution query", value="Continue based on our last answer", key="gv_query")
    session_id = st.text_input("session_id (optional)", value="demo-session-1", key="gv_sid")

    app = None
    if mode == "Live":
        with st.expander("Live mode notes", expanded=False):
            st.write("- Live mode runs a local LangGraph `app.stream()` inside Streamlit.")
            st.write("- It does not call MCP `/run` endpoints.")
        try:
            from graph.build_graph import build_research_graph

            app = build_research_graph()
        except Exception as e:
            st.error(f"Failed to build local graph app: {type(e).__name__}: {e}")
            app = None

    inputs: Dict[str, Any] = {"query": q, "session_id": session_id.strip() or None}
    trace = ui_graph.mock_trace_example()
    ui_graph.render_execution_viewer(app=app, inputs=inputs, mode=mode, replay_trace=trace)

