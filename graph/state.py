from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    query: str
    session_id: str | None

    # memory inputs
    history: list[dict]
    memory_hits: list[dict]

    # memory scoring
    memory_topk: int
    memory_coverage: int
    memory_quality: float
    memory_conflict: bool
    memory_sufficient: bool

    # planning
    plan: list[str]

    # outputs
    context: str
    answer: str

    # observability
    errors: list[str]
    observations: list[dict]

