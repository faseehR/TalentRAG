"""Shared state that flows through the LangGraph screening workflow."""

from typing import TypedDict


class ScreeningState(TypedDict, total=False):
    job_title: str
    job_rule: dict
    query: str

    retrieved_candidate_ids: list[str]     # after hybrid retrieval
    reranked_candidate_ids: list[str]      # after reranking, trimmed to top_n

    qualified_ids: list[str]
    rejected_ids: list[str]
    screening_results: dict                # candidate_id -> ScreeningResult dict

    evidence: dict                         # candidate_id -> {skill: snippet}
    explanations: dict                     # candidate_id -> explanation text

    final_report: list                     # ordered list of per-candidate report dicts
