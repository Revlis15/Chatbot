from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


class McpError(Exception):
    pass


def _base_url() -> str:
    # Primary config for Docker + local usage (requested):
    # - Docker: MCP_URL=http://mcp:8000
    # - Local:  MCP_URL=http://localhost:8000
    #
    # Backward compatibility: fall back to MCP_BASE_URL if set.
    return os.getenv("MCP_URL") or os.getenv("MCP_BASE_URL", "http://localhost:8000").rstrip("/")


def call_mcp(endpoint: str, query: str, *, timeout_s: int = 20, extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Simple MCP client wrapper.

    Args:
        endpoint: e.g. '/search_web', '/search_paper', '/retrieve'
        query: query string
    """
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    url = f"{_base_url()}{endpoint}"
    params: Dict[str, Any] = {"q": query}
    if extra_params:
        params.update(extra_params)

    try:
        resp = requests.get(url, params=params, timeout=timeout_s)
    except Exception as e:
        raise McpError(f"MCP request failed: {type(e).__name__}: {e}") from e

    if resp.status_code >= 400:
        raise McpError(f"MCP error {resp.status_code}: {resp.text}")

    try:
        return resp.json()
    except Exception as e:
        raise McpError(f"Invalid JSON from MCP: {type(e).__name__}: {e}") from e


def call_mcp_post(endpoint: str, payload: Dict[str, Any], *, timeout_s: int = 20) -> Dict[str, Any]:
    """
    Simple MCP POST client wrapper (JSON body).
    """
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    url = f"{_base_url()}{endpoint}"

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except Exception as e:
        raise McpError(f"MCP request failed: {type(e).__name__}: {e}") from e

    if resp.status_code >= 400:
        raise McpError(f"MCP error {resp.status_code}: {resp.text}")

    try:
        return resp.json()
    except Exception as e:
        raise McpError(f"Invalid JSON from MCP: {type(e).__name__}: {e}") from e

