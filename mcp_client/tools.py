from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp_client.client import McpError, call_mcp, call_mcp_post


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any
    error: Optional[str] = None


class ToolClient:
    """
    Thin MCP tool abstraction layer.
    Keeps endpoint paths/payload mapping out of agent logic.
    """

    def search_web(self, q: str) -> ToolResult:
        try:
            res = call_mcp("/search_web", q, timeout_s=20)
            return ToolResult(ok=True, data=res)
        except McpError as e:
            return ToolResult(ok=False, data={"query": q, "results": []}, error=str(e))

    def search_paper(self, q: str) -> ToolResult:
        try:
            res = call_mcp("/search_paper", q, timeout_s=25)
            return ToolResult(ok=True, data=res)
        except McpError as e:
            return ToolResult(ok=False, data={"query": q, "results": []}, error=str(e))

    def retrieve(self, q: str, k: int = 3) -> ToolResult:
        try:
            res = call_mcp("/retrieve", q, timeout_s=25, extra_params={"k": k})
            return ToolResult(ok=True, data=res)
        except McpError as e:
            return ToolResult(ok=False, data={"query": q, "k": k, "docs": []}, error=str(e))

