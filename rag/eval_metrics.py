"""
Standard information-retrieval metrics, used to evaluate retrieval
quality (dense vs hybrid vs hybrid+reranked) against a ground-truth
relevant set. Binary relevance (a candidate is either relevant to a
job or not — from the deterministic rule engine, not opinion).
"""

import math


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / min(k, len(top_k)) if top_k else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # rank is 0-indexed here, +2 so log2(2)=1 at rank 0
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def precision_recall_f1(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
