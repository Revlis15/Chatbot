from __future__ import annotations

import contextlib
import io
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import Body, FastAPI, HTTPException, Query

from rag.vector_store import ChromaVectorStore


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_items: int = 256) -> None:
        self._ttl = ttl_seconds
        self._max_items = max_items
        self._store: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        ent = self._store.get(key)
        if not ent:
            return None
        if time.time() >= ent.expires_at:
            self._store.pop(key, None)
            return None
        return ent.value

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self._max_items:
            # Simple eviction: drop oldest expiry (good enough for production-lite).
            oldest = min(self._store.items(), key=lambda kv: kv[1].expires_at)[0]
            self._store.pop(oldest, None)
        self._store[key] = CacheEntry(value=value, expires_at=time.time() + self._ttl)


app = FastAPI(title="MCP Tool Server (Research Assistant)", version="0.1.0")

_cache = TTLCache(ttl_seconds=int(os.getenv("MCP_CACHE_TTL_SECONDS", "300")))
_vector_store = ChromaVectorStore()
_graph = None
_graphs: Dict[str, Any] = {}


def _cache_key(endpoint: str, q: str, extra: Tuple[Tuple[str, str], ...] = ()) -> str:
    parts = [endpoint, q.strip()]
    for k, v in extra:
        parts.append(f"{k}={v}")
    return "|".join(parts)


def _normalize_web_results(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for it in items[:3]:
        title = str(it.get("title") or "").strip()
        content = str(it.get("content") or "").strip()
        raw_content = str(it.get("raw_content") or "").strip()
        if title or content or raw_content:
            out.append({"title": title, "content": content, "raw_content": raw_content})
    return out[:3]


def _search_web_tavily(q: str, include_raw_content: bool = False) -> List[Dict[str, str]]:
    """
    PRIMARY web search: Tavily if TAVILY_API_KEY exists.
    Returns [] on any failure (demo-safe).
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    print("[Web Search - Tavily]")
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        
        # Gửi cờ include_raw_content cho Tavily
        res = client.search(query=q, max_results=3, include_raw_content=include_raw_content)
        items = res.get("results") or []
        
        mapped: List[Dict[str, Any]] = []
        for r in items[:3]:
            mapped.append({
                "title": r.get("title") or "", 
                "content": r.get("content") or "",
                "raw_content": r.get("raw_content") if include_raw_content else None 
            })
        return mapped
    except Exception as e:
        print("[Web Search - Tavily]", "failed:", type(e).__name__)
        return []


def _search_web_ddg(q: str) -> List[Dict[str, str]]:
    """
    FALLBACK web search: DuckDuckGo (no API key).
    Returns [] on any failure (demo-safe).
    """
    print("[Web Search - DDG Fallback]")
    try:
        try:
            from ddgs import DDGS  # new package name
        except Exception:
            from duckduckgo_search import DDGS  # backward compatibility

        # duckduckgo-search can rate-limit; we must never crash.
        mapped: List[Dict[str, Any]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=3):
                mapped.append({"title": r.get("title") or "", "content": r.get("body") or ""})
        return _normalize_web_results(mapped)
    except Exception as e:
        print("[Web Search - DDG Fallback]", "failed:", type(e).__name__)
        return []


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _get_pipeline():
    if "production_pipeline" in _graphs:
        return _graphs["production_pipeline"]
    from graph.build_graph import build_production_pipeline

    _graphs["production_pipeline"] = build_production_pipeline()
    return _graphs["production_pipeline"]


@app.post("/run")
def run_pipeline(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Thin orchestration endpoint for demos/UI.
    Does NOT change agent logic; it simply invokes the existing LangGraph pipeline
    and captures stdout logs for display.
    """
    q = str((payload or {}).get("q") or "").strip()
    session_id = str((payload or {}).get("session_id") or "").strip()
    if not q:
        return {
            "query": q,
            "plan": [],
            "web_results": [],
            "papers": [],
            "context": [],
            "answer": "",
            "logs": "Empty query.",
        }

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            graph = _get_pipeline()
            state = graph.invoke(
                {"query": q, "session_id": session_id},
                config={
                    "tags": ["demo", "ui", "langgraph", "research-assistant", "production_pipeline"],
                    "metadata": {"step_type": "full_run", "pipeline": "production_pipeline", "session_id": session_id},
                },
            )

        docs = state.get("docs") or []
        context = [str(d.get("text") or "") for d in docs[:3] if isinstance(d, dict)]

        return {
            "query": q,
            "plan": state.get("plan") or [],
            "web_results": state.get("web_results") or [],
            "papers": state.get("papers") or [],
            "context": context,
            "answer": state.get("answer") or "",
            "errors": state.get("errors") or [],
            "observations": state.get("observations") or [],
            "logs": buf.getvalue(),
        }
    except Exception as e:
        logs = buf.getvalue()
        logs += f"\n[Run] ERROR: {type(e).__name__}\n"
        # Demo-safe: return valid JSON instead of crashing.
        return {
            "query": q,
            "plan": [],
            "web_results": [],
            "papers": [],
            "context": [],
            "answer": "",
            "errors": [],
            "observations": [],
            "logs": logs,
        }


@app.get("/search_web")
def search_web(q: str = Query(..., min_length=1), include_raw_content: bool = Query(False)) -> Dict[str, Any]:
    key = _cache_key("/search_web", q, extra=(("raw", str(include_raw_content)),))
    cached = _cache.get(key)
    if cached is not None:
        return {"query": q, "results": cached, "cached": True}

    # Hybrid search: try Tavily first, then DuckDuckGo fallback.
    results = _search_web_tavily(q, include_raw_content)
    if not results:
        results = _search_web_ddg(q)

    _cache.set(key, results)
    return {"query": q, "results": results, "cached": False}


@app.get("/search_paper")
def search_paper(q: str = Query(..., min_length=1)) -> Dict[str, Any]:
    key = _cache_key("/search_paper", q)
    cached = _cache.get(key)
    if cached is not None:
        return {"query": q, "results": cached, "cached": True}

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    print("[Paper Search]")
    headers = {"Accept": "application/json"}

    try:
        resp = requests.get(
            url,
            params={
                "query": q,
                "limit": 3,
                "fields": "title,abstract",
            },
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("[Paper Search]", "failed:", type(e).__name__)
        results: List[Dict[str, str]] = []
        _cache.set(key, results)
        return {"query": q, "results": results, "cached": False}

    items = (data.get("data") or [])[:3]
    results: List[Dict[str, str]] = []
    for it in items:
        results.append(
            {
                "title": str(it.get("title") or ""),
                "abstract": str(it.get("abstract") or ""),
            }
        )

    _cache.set(key, results)
    return {"query": q, "results": results, "cached": False}


@app.get("/retrieve")
def retrieve(q: str = Query(..., min_length=1), k: int = Query(3, ge=1, le=3)) -> Dict[str, Any]:
    key = _cache_key("/retrieve", q, extra=(("k", str(k)),))
    cached = _cache.get(key)
    if cached is not None:
        return {"query": q, "k": k, "docs": cached, "cached": True}

    try:
        docs = _vector_store.retrieve(query=q, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector retrieval failed: {type(e).__name__}")

    out = [{"id": d.id, "text": d.text, "score": d.score, "metadata": d.metadata} for d in docs]
    _cache.set(key, out)
    return {"query": q, "k": k, "docs": out, "cached": False}

