from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st


def _mcp_url() -> str:
    return os.getenv("MCP_URL", "http://localhost:8001").rstrip("/")


def _post_run(query: str) -> Dict[str, Any]:
    url = f"{_mcp_url()}/run"
    resp = requests.post(url, json={"q": query}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _post_run_compare(query: str, patterns: List[str]) -> Dict[str, Any]:
    url = f"{_mcp_url()}/run_compare"
    resp = requests.post(url, json={"query": query, "patterns": patterns}, timeout=180)
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

query = st.text_input("Query", value="So sánh YOLOv8 và Faster R-CNN mới nhất")
patterns = st.multiselect(
    "Patterns",
    options=["react", "planner", "rewoo"],
    default=["planner", "react", "rewoo"],
)
run = st.button("Run", type="primary")

if run:
    if not query.strip():
        st.warning("Please enter a query.")
    elif not patterns:
        st.warning("Please select at least one pattern.")
    else:
        with st.spinner("Running patterns..."):
            try:
                data = _post_run_compare(query.strip(), patterns)
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
                    st.success(_truncate(ans, 1200))
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

