"""
Reranking stage: takes the top-K candidates from hybrid retrieval and
rescores them by how well each one actually answers the query — this
is what fixes cases where initial retrieval gets high recall but puts
the truly best match at position #7 instead of #1.

  - CrossEncoderReranker: production. Uses a real cross-encoder
    (e.g. BAAI/bge-reranker-v2-m3 via sentence-transformers or
    Pinecone's hosted reranker) that jointly scores (query, document)
    pairs — much more precise than comparing independent embeddings.
  - LexicalOverlapReranker: dependency-light stand-in used for local
    testing here — scores by query/document token overlap. Same
    interface, so swapping in the real cross-encoder later doesn't
    touch any calling code.
"""

from abc import ABC, abstractmethod
import re


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        """candidates: [{"id":..., "text":..., ...}]. Returns re-sorted subset with 'rerank_score' added."""
        ...


class CrossEncoderReranker(BaseReranker):
    """Production backend. Requires `pip install sentence-transformers` (downloads model weights)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_n=5):
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]


class LexicalOverlapReranker(BaseReranker):
    """
    Local stand-in for a cross-encoder. Scores each candidate by
    token-overlap (Jaccard-like) with the query, with a small bonus
    for exact substring matches. Not as precise as a real
    cross-encoder, but implements the identical interface so the rest
    of the pipeline (and the evaluation code that compares
    "with reranking" vs "without") can be built and run today.
    """

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"[a-z0-9+#.]+", text.lower()))

    def rerank(self, query, candidates, top_n=5):
        q_tokens = self._tokenize(query)
        q_lower = query.lower()
        for c in candidates:
            c_tokens = self._tokenize(c["text"])
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            substring_bonus = 0.2 if q_lower in c["text"].lower() else 0.0
            c["rerank_score"] = overlap + substring_bonus
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]


def get_reranker(backend: str = "local_lexical", **kwargs) -> BaseReranker:
    if backend == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    if backend == "local_lexical":
        return LexicalOverlapReranker(**kwargs)
    raise ValueError(f"Unknown reranker backend: {backend}")
