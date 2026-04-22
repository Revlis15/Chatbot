from __future__ import annotations

from typing import List


def build_plan(query: str) -> List[str]:
    """
    Production-lite planner: returns a structured tool plan.
    """
    _ = query  # reserved for future routing
    return ["search_web", "search_paper", "retrieve", "synthesize"]

