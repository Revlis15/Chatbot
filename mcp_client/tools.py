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
    Thin MCP tool abstraction layer (Updated).
    Hỗ trợ truyền tham số để lấy raw content cho nghiên cứu chuyên sâu.
    """

    def search_web(self, q: str, include_raw_content: bool = False) -> ToolResult:
        """
        Gửi yêu cầu tìm kiếm web. 
        Nếu include_raw_content=True, Tavily sẽ trả về toàn bộ nội dung text của trang web.
        """
        try:
            # Truyền include_raw_content vào extra_params để MCP Server xử lý
            res = call_mcp(
                "/search_web", 
                q, 
                timeout_s=30, # Tăng timeout vì lấy raw content sẽ chậm hơn snippet
                extra_params={"include_raw_content": include_raw_content}
            )
            return ToolResult(ok=True, data=res)
        except McpError as e:
            return ToolResult(ok=False, data={"query": q, "results": []}, error=str(e))

    def search_paper(self, q: str) -> ToolResult:
        try:
            res = call_mcp("/search_paper", q, timeout_s=30)
            return ToolResult(ok=True, data=res)
        except McpError as e:
            return ToolResult(ok=False, data={"query": q, "results": []}, error=str(e))

    def retrieve(self, q: str, k: int = 3) -> ToolResult:
        try:
            res = call_mcp("/retrieve", q, timeout_s=25, extra_params={"k": k})
            return ToolResult(ok=True, data=res)
        except McpError as e:
            return ToolResult(ok=False, data={"query": q, "k": k, "docs": []}, error=str(e))