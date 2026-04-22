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
_db_inited = False
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
        if title or content:
            out.append({"title": title, "content": content})
    return out[:3]


def _search_web_tavily(q: str) -> List[Dict[str, str]]:
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
        # Tavily's python SDK wraps HTTP; keep result count small for demos.
        res = client.search(query=q, max_results=3)
        items = res.get("results") or []
        mapped: List[Dict[str, Any]] = []
        for r in items[:3]:
            mapped.append({"title": r.get("title") or "", "content": r.get("content") or ""})
        return _normalize_web_results(mapped)
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


def _db_path() -> str:
    path = os.getenv("RESEARCH_DB_PATH", "/app/data/research.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _init_db_if_needed() -> None:
    global _db_inited
    if _db_inited:
        return
    try:
        conn = sqlite3.connect(_db_path(), timeout=5)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  abstract TEXT,
                  source TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_title ON papers(title);")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query TEXT,
                  pattern TEXT,
                  answer TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_cache (
                  tool TEXT,
                  query TEXT,
                  response_json TEXT,
                  created_at INTEGER
                );
                """
            )
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_cache_tool_query ON tool_cache(tool, query);")
            conn.commit()
            _db_inited = True
        finally:
            conn.close()
    except Exception:
        # Demo-safe: never crash on DB init failure.
        _db_inited = False


def _db_query_papers(q: str, limit: int = 3) -> List[Dict[str, str]]:
    try:
        _init_db_if_needed()
        conn = sqlite3.connect(_db_path(), timeout=5)
        try:
            like = f"%{q}%"
            rows = conn.execute(
                """
                SELECT title, abstract
                FROM papers
                WHERE title LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (like, limit),
            ).fetchall()
            out: List[Dict[str, str]] = []
            for title, abstract in rows:
                out.append({"title": str(title or ""), "content": str(abstract or "")})
            return out[:limit]
        finally:
            conn.close()
    except Exception:
        return []


def _db_save_papers(papers: List[Dict[str, Any]]) -> Dict[str, int]:
    inserted = 0
    ignored = 0
    try:
        _init_db_if_needed()
        conn = sqlite3.connect(_db_path(), timeout=5)
        try:
            for p in papers[:50]:
                title = str(p.get("title") or "").strip()
                abstract = str(p.get("abstract") or "").strip()
                source = str(p.get("source") or "").strip()
                if not title:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO papers(title, abstract, source) VALUES(?, ?, ?)",
                    (title, abstract, source),
                )
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    ignored += 1
            conn.commit()
        finally:
            conn.close()
    except Exception:
        return {"inserted": 0, "ignored": 0}
    return {"inserted": inserted, "ignored": ignored}


def _db_save_history(query: str, pattern: str, answer: str) -> None:
    try:
        _init_db_if_needed()
        conn = sqlite3.connect(_db_path(), timeout=5)
        try:
            conn.execute(
                "INSERT INTO history(query, pattern, answer) VALUES(?, ?, ?)",
                (query, pattern, answer),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # Demo-safe: ignore history failures.
        return


def _db_get_history(limit: int = 20) -> List[Dict[str, str]]:
    try:
        _init_db_if_needed()
        conn = sqlite3.connect(_db_path(), timeout=5)
        try:
            rows = conn.execute(
                """
                SELECT query, pattern, answer, created_at
                FROM history
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            out: List[Dict[str, str]] = []
            for q, pattern, answer, created_at in rows:
                out.append(
                    {
                        "query": str(q or ""),
                        "pattern": str(pattern or ""),
                        "answer": str(answer or ""),
                        "time": str(created_at or ""),
                    }
                )
            return out
        finally:
            conn.close()
    except Exception:
        return []


def _tool_cache_ttl_seconds() -> int:
    # default: 10 minutes
    return int(os.getenv("TOOL_CACHE_TTL_SECONDS", "600"))


def _tool_cache_get(tool: str, q: str) -> Optional[Dict[str, Any]]:
    try:
        _init_db_if_needed()
        if not _db_inited:
            return None
        conn = sqlite3.connect(_db_path(), timeout=5)
        try:
            row = conn.execute(
                "SELECT response_json, created_at FROM tool_cache WHERE tool = ? AND query = ?",
                (tool, q),
            ).fetchone()
            if not row:
                return None
            payload, created_at = row
            if not payload:
                return None
            age = int(time.time()) - int(created_at or 0)
            if age > _tool_cache_ttl_seconds():
                return None
            import json

            return json.loads(payload)
        finally:
            conn.close()
    except Exception:
        return None


def _tool_cache_set(tool: str, q: str, response: Dict[str, Any]) -> None:
    try:
        _init_db_if_needed()
        if not _db_inited:
            return
        import json

        payload = json.dumps(response, ensure_ascii=False)
        conn = sqlite3.connect(_db_path(), timeout=5)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO tool_cache(tool, query, response_json, created_at) VALUES(?, ?, ?, ?)",
                (tool, q, payload, int(time.time())),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        return


def _build_react_graph():
    from langgraph.graph import END, StateGraph

    from agents.rag_agent import rag_node
    from agents.research_agent import research_node
    from agents.synth_agent import synth_node

    # Minimal react-style graph: research -> rag -> synth (no planner node).
    g = StateGraph(dict)
    g.add_node("research_agent", research_node)
    g.add_node("rag_agent", rag_node)
    g.add_node("synth_agent", synth_node)
    g.set_entry_point("research_agent")
    g.add_edge("research_agent", "rag_agent")
    g.add_edge("rag_agent", "synth_agent")
    g.add_edge("synth_agent", END)
    return g.compile()


def _get_graph(pattern: str = "planner"):
    """
    Return a compiled graph per pattern.
    - planner/rewoo: full pipeline graph
    - react: graph without planner node
    """
    p = (pattern or "planner").strip().lower()
    if p in _graphs:
        return _graphs[p]

    if p in ("planner", "rewoo"):
        from graph.build_graph import build_research_graph

        _graphs[p] = build_research_graph()
        return _graphs[p]

    if p == "react":
        _graphs[p] = _build_react_graph()
        return _graphs[p]

    # Fallback to planner for unknown patterns.
    from graph.build_graph import build_research_graph

    _graphs[p] = build_research_graph()
    return _graphs[p]


@app.post("/run")
def run_pipeline(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Thin orchestration endpoint for demos/UI.
    Does NOT change agent logic; it simply invokes the existing LangGraph pipeline
    and captures stdout logs for display.
    """
    q = str((payload or {}).get("q") or "").strip()
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
            graph = _get_graph("planner")
            state = graph.invoke(
                {"query": q},
                config={
                    "tags": ["demo", "ui", "langgraph", "research-assistant"],
                    "metadata": {"step_type": "full_run"},
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


@app.post("/save_papers")
def save_papers(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    papers = (payload or {}).get("papers") or []
    if not isinstance(papers, list):
        papers = []
    counts = _db_save_papers(papers)
    print("[DB Save]", f"inserted={counts.get('inserted', 0)}", f"ignored={counts.get('ignored', 0)}")
    return counts


@app.get("/query_papers")
def query_papers(q: str = Query(..., min_length=1)) -> List[Dict[str, str]]:
    results = _db_query_papers(q, limit=3)
    print("[DB Query]", f"q={q}", f"results={len(results)}")
    return results


@app.post("/run_compare")
def run_compare(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    query = str((payload or {}).get("query") or "").strip()
    patterns = (payload or {}).get("patterns") or []
    if not isinstance(patterns, list):
        patterns = []
    patterns = [str(p).strip().lower() for p in patterns if str(p).strip()]
    if not patterns:
        patterns = ["planner"]

    results: Dict[str, Any] = {}
    for pattern in patterns:
        buf = io.StringIO()
        print("[Compare] Running pattern:", pattern)
        try:
            with contextlib.redirect_stdout(buf):
                graph = _get_graph(pattern)
                # For patterns that don't include planner node, we still attach a plan for UI consistency.
                plan = []
                try:
                    from agents.planner import build_plan

                    plan = build_plan(query)
                except Exception:
                    plan = []

                state = graph.invoke(
                    {"query": query, "plan": plan},
                    config={
                        "tags": ["demo", "compare", "langgraph", "research-assistant", pattern],
                        "metadata": {"pattern": pattern},
                    },
                )
            answer = str(state.get("answer") or "")
            results[pattern] = {
                "answer": answer,
                "plan": state.get("plan") or plan,
                "errors": state.get("errors") or [],
                "observations": state.get("observations") or [],
                "logs": buf.getvalue(),
            }
            _db_save_history(query=query, pattern=pattern, answer=answer)
        except Exception as e:
            logs = buf.getvalue()
            logs += f"\n[Compare] ERROR: {type(e).__name__}\n"
            results[pattern] = {"answer": "", "plan": [], "errors": [], "observations": [], "logs": logs}
            _db_save_history(query=query, pattern=pattern, answer="")

    print("[History] Saved")
    return {"query": query, "results": results}


@app.get("/history")
def history() -> List[Dict[str, str]]:
    return _db_get_history(limit=20)


@app.get("/search_web")
def search_web(q: str = Query(..., min_length=1)) -> Dict[str, Any]:
    key = _cache_key("/search_web", q)
    cached = _cache.get(key)
    if cached is not None:
        return {"query": q, "results": cached, "cached": True}

    db_cached = _tool_cache_get("search_web", q)
    if isinstance(db_cached, dict) and "results" in db_cached:
        results = db_cached.get("results") or []
        _cache.set(key, results)
        return {"query": q, "results": results, "cached": True, "cached_via": "sqlite"}

    # Hybrid search: try Tavily first, then DuckDuckGo fallback.
    results = _search_web_tavily(q)
    if not results:
        results = _search_web_ddg(q)

    _cache.set(key, results)
    _tool_cache_set("search_web", q, {"results": results})
    return {"query": q, "results": results, "cached": False}


@app.get("/search_paper")
def search_paper(q: str = Query(..., min_length=1)) -> Dict[str, Any]:
    key = _cache_key("/search_paper", q)
    cached = _cache.get(key)
    if cached is not None:
        return {"query": q, "results": cached, "cached": True}

    db_cached = _tool_cache_get("search_paper", q)
    if isinstance(db_cached, dict) and "results" in db_cached:
        results = db_cached.get("results") or []
        _cache.set(key, results)
        return {"query": q, "results": results, "cached": True, "cached_via": "sqlite"}

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
        _tool_cache_set("search_paper", q, {"results": results})
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
    _tool_cache_set("search_paper", q, {"results": results})
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

