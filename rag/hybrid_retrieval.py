"""
Hybrid retrieval: combines dense (semantic) search from the vector
store with sparse (keyword/BM25) search, then merges results with
Reciprocal Rank Fusion (RRF). This is the "hybrid retrieval" module
from the project plan — dense search alone can miss exact terms like
"AWS Certified Solutions Architect" or "PostgreSQL", which keyword
search catches reliably; RRF combines the strengths of both without
having to hand-tune a weighting.
"""

from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25 = None
        self.bm25_ids: list[str] = []

    def index_bm25(self, ids: list[str], texts: list[str]):
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_ids = ids

    def _dense_search(self, query: str, top_k: int, filter: dict | None) -> list[tuple[str, float]]:
        qvec = self.embedder.embed_query(query)
        results = self.vector_store.query(qvec, top_k=top_k, filter=filter)
        return [(r["id"], r["score"]) for r in results]

    def _sparse_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.bm25_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter: dict | None = None,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of dense + sparse rankings."""
        dense = self._dense_search(query, top_k=top_k * 2, filter=filter)
        sparse = self._sparse_search(query, top_k=top_k * 2)

        fused_scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(dense):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, (doc_id, _) in enumerate(sparse):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

        ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": doc_id, "rrf_score": score} for doc_id, score in ranked_ids]

    def dense_only(self, query: str, top_k: int = 20, filter: dict | None = None) -> list[dict]:
        """Exposed separately so evaluation can compare dense-only vs hybrid."""
        return [{"id": i, "score": s} for i, s in self._dense_search(query, top_k, filter)]
