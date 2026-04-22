from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


class WebResult(TypedDict, total=False):
    title: str
    content: str
    url: str
    provider: str


class Paper(TypedDict, total=False):
    title: str
    abstract: str
    source: str
    url: str
    year: int


class Doc(TypedDict, total=False):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class Observation(TypedDict, total=False):
    step: str
    tool: str
    input: str
    output_size: int
    ok: bool
    note: str


class GraphState(TypedDict, total=False):
    # Required invariant: must always exist (best-effort enforced by nodes)
    query: str

    # Pattern and plan
    pattern: Literal["planner", "react", "rewoo"]
    plan: List[str]

    # Research results
    web_results: List[WebResult]
    papers: List[Paper]
    db_papers: List[Paper]

    # RAG
    docs: List[Doc]
    context: Optional[str]

    # Output
    answer: str

    # Reliability / observability
    errors: List[Dict[str, Any]]
    observations: List[Observation]

