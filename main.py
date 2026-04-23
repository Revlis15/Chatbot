from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Process
from typing import Any, Dict

import requests

from graph.build_graph import build_production_pipeline


def _init_langsmith_tracing() -> None:
    """
    Optional LangSmith tracing.
    - Enabled only if LANGCHAIN_API_KEY exists.
    - Safe no-op otherwise.
    """
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "research-assistant-demo")
        print("[LangSmith] ENABLED")
    else:
        print("[LangSmith] DISABLED")


def _langgraph_config(mode: str) -> Dict[str, Any]:
    return {
        "tags": ["demo", "langgraph", "research-assistant"],
        "metadata": {
            "mode": mode,
            "agents": ["planner", "research_agent", "rag_agent", "synth_agent"],
        },
    }


def _force_utf8_stdio() -> None:
    # Windows consoles may default to legacy encodings; ensure logs can print Vietnamese.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _run_uvicorn() -> None:
    import uvicorn

    from mcp_server.server import app

    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config=config)
    server.run()


def _wait_for_server(base_url: str, timeout_s: int = 20) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url.rstrip('/')}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_planner_mode(query: str) -> Dict[str, Any]:
    app = build_production_pipeline()
    return app.invoke({"query": query}, config=_langgraph_config(mode="planner"))


def run_pipeline_mode(query: str) -> Dict[str, Any]:
    app = build_production_pipeline()
    return app.invoke({"query": query}, config=_langgraph_config(mode="production"))


def main() -> int:
    _force_utf8_stdio()
    _init_langsmith_tracing()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default=os.getenv("MODE", "planner"),
        choices=["production"],
        help="production=single pipeline",
    )
    parser.add_argument(
        "--query",
        default="So sánh YOLOv8 và Faster R-CNN mới nhất",
        help="Research query",
    )
    parser.add_argument(
        "--start-server",
        default=os.getenv("START_MCP_SERVER", "1"),
        choices=["0", "1"],
        help="Start MCP server automatically (1) or assume already running (0)",
    )
    args = parser.parse_args()

    base_url = os.getenv("MCP_URL") or os.getenv("MCP_BASE_URL", "http://localhost:8000")
    server_proc: Process | None = None

    if args.start_server == "1":
        server_proc = Process(target=_run_uvicorn, daemon=True)
        server_proc.start()
        if not _wait_for_server(base_url, timeout_s=25):
            print("Failed to start MCP server or reach /health.", file=sys.stderr)
            return 2

    try:
        if args.mode == "production":
            state = run_pipeline_mode(args.query)
            print("\n[Final Answer]\n")
            print(state.get("answer", ""))
            return 0
        return 1
    finally:
        if server_proc is not None and server_proc.is_alive():
            server_proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())

