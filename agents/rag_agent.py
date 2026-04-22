from __future__ import annotations

from typing import Any, Dict, List

from mcp_client.tools import ToolClient


def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = str(state.get("query") or "").strip()
    print("[RAG]")
    if not query:
        # Demo-safe: keep pipeline alive even if state is malformed.
        return {"query": query, "docs": []}
    tc = ToolClient()
    tr = tc.retrieve(query, k=3)
    docs: List[Dict[str, Any]] = (tr.data or {}).get("docs", [])[:3]
    print("Retrieved:", f"{len(docs)} docs")
    return {"query": query, "docs": docs}

