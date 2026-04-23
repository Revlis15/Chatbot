from __future__ import annotations

from typing import Annotated, TypedDict
import operator


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

    # tool policy
    tool_policy: dict

    # tool outputs
    web_results: list[dict]
    papers: list[dict]
    db_papers: list[dict]
    docs: list[dict]

    # outputs
    context: str
    answer: str
    synth_failed: bool

    # observability
    observations: Annotated[list[dict], operator.add] # Tự động cộng dồn thay vì ghi đè
    errors: Annotated[list[dict], operator.add]