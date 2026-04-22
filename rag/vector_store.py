from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb

from rag.chunking import chunk_text

class HuggingFaceMiniLMEmbeddingFunction:
    """
    Local embeddings via SentenceTransformers (LangChain wrapper), compatible with Chroma's
    embedding_function interface.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder: str = "/app/cache",
    ) -> None:
        self._model_name = model_name
        self._cache_folder = cache_folder
        os.makedirs(self._cache_folder, exist_ok=True)

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
            print("[HF] Authenticated")
        else:
            print("[HF] No token (public mode)")

        print("[Embedding - HF]", f"model={self._model_name}", f"cache_folder={self._cache_folder}")

        from langchain_huggingface import HuggingFaceEmbeddings

        self._emb = HuggingFaceEmbeddings(model_name=self._model_name, cache_folder=self._cache_folder)

    def name(self) -> str:
        return f"hf-minilm-l6-v2::{self._model_name}"

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self._model_name, "cache_folder": self._cache_folder}

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Chroma expects a list of vectors for list-of-texts input
        return self._emb.embed_documents(list(input))

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._emb.embed_documents(list(input))

    def embed_query(self, input: Any) -> List[List[float]]:
        # Some Chroma versions pass `input` as a single string, others as a list of strings.
        if isinstance(input, list):
            strs = [s for s in input if isinstance(s, str)]
            if not strs:
                return [[]]
            return self._emb.embed_documents(strs)
        if isinstance(input, str):
            return [self._emb.embed_query(input)]
        return [[]]


@dataclass(frozen=True)
class RetrievedDoc:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "research_assistant_docs_hf_minilm_l6_v2",
    ) -> None:
        # Docker default: mount this path to a named volume for persistence.
        self._persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._embedding_fn = HuggingFaceMiniLMEmbeddingFunction(
            model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_folder=os.getenv("HF_CACHE_FOLDER", "/app/cache"),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def bootstrap_if_empty(self) -> None:
        if self._collection.count() > 0:
            return

        seed_docs = [
            {
                "id": "od_overview_1",
                "text": (
                    "Object detection models are often compared on accuracy (mAP), latency, "
                    "throughput (FPS), parameter count, and deployment constraints. "
                    "One-stage detectors (e.g., YOLO family) tend to be faster; two-stage "
                    "detectors (e.g., Faster R-CNN) can be more accurate in some regimes."
                ),
                "metadata": {"source": "seed", "topic": "object_detection"},
            },
            {
                "id": "yolov8_1",
                "text": (
                    "YOLOv8 is a modern one-stage detector/segmenter from Ultralytics. "
                    "It emphasizes real-time performance, supports different model sizes, "
                    "and is often evaluated across COCO with strong speed/accuracy tradeoffs."
                ),
                "metadata": {"source": "seed", "topic": "yolov8"},
            },
            {
                "id": "faster_rcnn_1",
                "text": (
                    "Faster R-CNN is a two-stage detector using a Region Proposal Network (RPN) "
                    "to generate candidate regions, followed by classification and bounding box "
                    "regression. It is commonly used as a strong baseline and for higher-accuracy settings."
                ),
                "metadata": {"source": "seed", "topic": "faster_rcnn"},
            },
            {
                "id": "tradeoffs_1",
                "text": (
                    "In practice, YOLO-style models are preferred for edge and real-time inference, "
                    "while Faster R-CNN variants can be preferred for offline processing or when "
                    "small-object accuracy and proposal-based refinement matter more than speed."
                ),
                "metadata": {"source": "seed", "topic": "tradeoffs"},
            },
        ]

        # Chunk seed docs for more stable retrieval and smaller embedding inputs.
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        for d in seed_docs:
            base_id = str(d["id"])
            text = str(d["text"])
            md = dict(d.get("metadata") or {})
            chunks = chunk_text(text, max_chars=int(os.getenv("RAG_CHUNK_MAX_CHARS", "360")), overlap_chars=40)
            if not chunks:
                continue
            for idx, ch in enumerate(chunks, start=1):
                ids.append(f"{base_id}::c{idx}")
                docs.append(ch)
                metas.append({**md, "chunk": idx, "chunk_count": len(chunks)})

        self._collection.add(ids=ids, documents=docs, metadatas=metas)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedDoc]:
        self.bootstrap_if_empty()
        res = self._collection.query(query_texts=[query], n_results=k)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: List[RetrievedDoc] = []
        for i, doc_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            md = metas[i] if i < len(metas) and metas[i] is not None else {}
            dist = float(dists[i]) if i < len(dists) else 1.0
            score = 1.0 / (1.0 + dist)  # stable, monotonic; higher is better
            out.append(RetrievedDoc(id=str(doc_id), text=text, score=score, metadata=dict(md)))
        return out

