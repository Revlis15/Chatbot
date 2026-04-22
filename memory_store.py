from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MemoryHit:
    text: str
    score: float
    metadata: Dict[str, Any]


class MemoryStore:
    """
    Chroma-backed long-term memory store.
    - Memory is scoped by session_id via metadata filter.
    - Demo-safe: never raises (returns empty results / False on failure).
    """

    def __init__(
        self,
        *,
        persist_dir: Optional[str] = None,
        collection_name: str = "research_assistant_memory",
    ) -> None:
        self._persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
        self._collection_name = collection_name
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb

        from rag.vector_store import HuggingFaceMiniLMEmbeddingFunction

        client = chromadb.PersistentClient(path=self._persist_dir)
        emb = HuggingFaceMiniLMEmbeddingFunction(
            model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_folder=os.getenv("HF_CACHE_FOLDER", "/app/cache"),
        )
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=emb,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def add_memory(self, session_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            sid = str(session_id or "").strip()
            t = str(text or "").strip()
            if not sid or not t:
                return False
            col = self._get_collection()
            now = int(time.time())
            md = dict(metadata or {})
            md.setdefault("session_id", sid)
            md.setdefault("created_at", now)
            md.setdefault("last_used", now)
            md.setdefault("usage_count", 0)
            md.setdefault("importance", compute_importance(t))
            md.setdefault("type", classify_memory(t))
            mem_id = str(uuid.uuid4())
            col.add(ids=[mem_id], documents=[t], metadatas=[md])
            return True
        except Exception:
            return False

    def _search_raw(self, query: str, session_id: str, top_k: int = 3) -> List[Tuple[str, float, Dict[str, Any]]]:
        try:
            sid = str(session_id or "").strip()
            q = str(query or "").strip()
            if not sid or not q:
                return []
            col = self._get_collection()
            res = col.query(
                query_texts=[q],
                n_results=int(top_k),
                where={"session_id": sid},
                include=["documents", "metadatas", "distances"],
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            ids = (res.get("ids") or [[]])[0]
            out: List[Tuple[str, float, Dict[str, Any]]] = []
            for i, text in enumerate(docs):
                md = metas[i] if i < len(metas) and metas[i] is not None else {}
                dist = float(dists[i]) if i < len(dists) else 1.0
                score = 1.0 / (1.0 + dist)
                md2 = dict(md)
                if i < len(ids) and ids[i]:
                    md2.setdefault("_id", str(ids[i]))
                out.append((str(text or ""), float(score), md2))
            return out
        except Exception:
            return []

    # --- Required APIs (strict names) ---
    def search_memory(self, session_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        raw = self._search_raw(query=query, session_id=session_id, top_k=k)
        out: List[Dict[str, Any]] = []
        for text, score, md in raw:
            out.append({"text": text, "score": float(score), "metadata": dict(md)})
        return out

    # Alias for callers that prefer explicit name
    def add_memory_entry(self, session_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        return self.add_memory(session_id=session_id, text=text, metadata=metadata)

    def update_memory_usage(self, *, session_id: str, memory_id: str, used_at: Optional[int] = None) -> bool:
        """
        Increment usage_count and update last_used for a stored memory id.
        Best-effort: returns False on any failure.
        """
        try:
            sid = str(session_id or "").strip()
            mid = str(memory_id or "").strip()
            if not sid or not mid:
                return False
            col = self._get_collection()
            got = col.get(ids=[mid], include=["documents", "metadatas"])
            docs = (got.get("documents") or [])
            metas = (got.get("metadatas") or [])
            if not docs or not metas:
                return False
            doc = str(docs[0] or "")
            md = dict(metas[0] or {})
            if str(md.get("session_id") or "") != sid:
                return False
            now = int(used_at if isinstance(used_at, int) else time.time())
            usage_count = md.get("usage_count")
            try:
                usage_i = int(usage_count) if usage_count is not None else 0
            except Exception:
                usage_i = 0
            md["usage_count"] = usage_i + 1
            md["last_used"] = now
            col.update(ids=[mid], documents=[doc], metadatas=[md])
            return True
        except Exception:
            return False


def compute_importance(text: str) -> float:
    """
    Score memory importance from 0 to 1
    Rule-based (no LLM for now)
    """
    t = str(text or "")
    score = 0.3  # base

    tl = t.lower()
    if any(k in tl for k in ["i prefer", "i like", "my favorite"]):
        score += 0.5

    if any(k in tl for k in ["my name is", "i am", "i work as"]):
        score += 0.4

    if len(t) > 200:
        score += 0.1

    return float(min(score, 1.0))


def classify_memory(text: str) -> str:
    """
    Classify memory type.
    """
    t = " ".join(str(text or "").lower().split())
    if any(k in t for k in ["i prefer", "i like", "my favorite"]):
        return "user_preference"
    if any(k in t for k in ["my name is", "i am", "i work as"]):
        return "user_profile"
    if any(k in t for k in ["result", "we did", "we decided", "summary:", "conclusion"]):
        return "task_result"
    return "fact"


def add_memory(session_id: str, text: str, metadata: dict) -> None:
    """
    Collection: research_assistant_memory
    """
    _ = MemoryStore().add_memory(session_id=session_id, text=text, metadata=dict(metadata or {}))


def search_memory(session_id: str, query: str, k: int = 5) -> list[dict]:
    """
    Return:
    [
      {"text": "...", "score": float, "metadata": {...}}
    ]
    """
    return MemoryStore().search_memory(session_id=session_id, query=query, k=int(k))

