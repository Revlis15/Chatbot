from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st


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
    resp = requests.post(url, json=payload, timeout=360)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")

with st.container():
    query = st.text_input("Query", placeholder="Câu hỏi nghiên cứu của bạn là gì?")
    session_id = st.text_input("session_id (optional)", value="test")
    run = st.button("Run", type="primary")

    if run:
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Running pipeline..."):
                try:
                    sid = session_id.strip() or None
                    data = _post_run_with_session(query.strip(), sid)
                except Exception as e:
                    st.error(f"API call failed: {type(e).__name__}: {e}")
                    st.stop()

            st.divider()

            # st.subheader("🧠 Plan")
            # plan = data.get("plan") or []
            # st.write(plan if plan else "(none)")

            st.subheader("✅ Answer")
            ans = str(data.get("answer") or "").strip()
            if ans:
                st.success(ans)
            else:
                st.warning("No answer returned.")

            st.subheader("🧾 Logs")
            logs = str(data.get("logs") or "")
            if logs.strip():
                st.code(logs, language="text")
            else:
                st.info("No logs.")
