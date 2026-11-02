"""
Phase 3: Embeddings + Retrieval + Reranking.

Builds the searchable index from out/chunks.jsonl (produced by
run_ingestion.py), then lets you run queries through three pipelines
side by side so you can see what each stage actually buys you:

    1. dense only
    2. hybrid (dense + BM25, fused with RRF)
    3. hybrid + reranked

Usage:
    python run_retrieval.py --chunks out/chunks.jsonl --query "Find candidates with Django and PostgreSQL experience"

Defaults to the local TF-IDF embedder + local in-memory vector store +
lexical-overlap reranker, so this runs with zero external API keys or
model downloads. Swap --embed-backend openai / --store-backend pinecone
/ --rerank-backend cross_encoder for the production versions — nothing
else in this file changes.
"""

import argparse
import json

from rag.embeddings import get_embedder
from rag.vector_store import get_vector_store
from rag.hybrid_retrieval import HybridRetriever
from rag.reranker import get_reranker


def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_index(chunks: list[dict], embed_backend: str, store_backend: str):
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    embedder = get_embedder(embed_backend)
    if hasattr(embedder, "fit"):
        embedder.fit(texts)  # local_tfidf needs a corpus-level fit

    print(f"Embedding {len(texts)} chunks...")
    vectors = embedder.embed(texts)

    store = get_vector_store(store_backend)
    store.upsert(ids, vectors, metadatas)

    retriever = HybridRetriever(store, embedder)
    retriever.index_bm25(ids, texts)

    text_by_id = dict(zip(ids, texts))
    return retriever, text_by_id


def show_results(label: str, results: list[dict], text_by_id: dict, k: int = 5):
    print(f"\n--- {label} (top {k}) ---")
    for r in results[:k]:
        candidate = r["id"].split("__")[0]
        text = text_by_id.get(r["id"], "")[:100]
        score = r.get("rerank_score", r.get("rrf_score", r.get("score", 0)))
        print(f"  {candidate:>10}  score={score:.4f}  | {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--embed-backend", default="local_tfidf", choices=["local_tfidf", "openai", "gemini"])
    ap.add_argument("--store-backend", default="local", choices=["local", "pinecone"])
    ap.add_argument("--rerank-backend", default="local_lexical", choices=["local_lexical", "cross_encoder"])
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    retriever, text_by_id = build_index(chunks, args.embed_backend, args.store_backend)

    print(f'\nQuery: "{args.query}"')

    dense = retriever.dense_only(args.query, top_k=args.top_k)
    show_results("1) Dense only", dense, text_by_id, k=args.top_n)

    hybrid = retriever.search(args.query, top_k=args.top_k)
    show_results("2) Hybrid (dense + BM25, RRF-fused)", hybrid, text_by_id, k=args.top_n)

    reranker = get_reranker(args.rerank_backend)
    hybrid_candidates = [{"id": r["id"], "text": text_by_id.get(r["id"], "")} for r in hybrid]
    reranked = reranker.rerank(args.query, hybrid_candidates, top_n=args.top_n)
    show_results("3) Hybrid + Reranked", reranked, text_by_id, k=args.top_n)


if __name__ == "__main__":
    main()
