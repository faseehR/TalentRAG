"""
Phase 5: Run the full LangGraph screening workflow end-to-end —
retrieval -> reranking -> rule engine -> evidence collection ->
evidence-grounded explanation -> final report.

Usage:
    python run_agent.py --chunks out/chunks.jsonl --job-rules out/job_rules.json \
        --job-title "Software Engineer" --top-n 15

Defaults to the local no-API-key backends everywhere (TF-IDF embedder,
in-memory vector store, lexical reranker, template explainer), same as
Phases 3-4, so this runs standalone. Pass --explainer anthropic
(+ ANTHROPIC_API_KEY) for real LLM-generated explanations in production.
"""

import argparse
import json

from rag.embeddings import get_embedder
from rag.vector_store import get_vector_store
from rag.hybrid_retrieval import HybridRetriever
from rag.reranker import get_reranker
from rag.llm_explainer import get_explainer
from agents.graph import build_graph


def load_chunks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_candidate_views(chunks: list[dict]):
    """From flat chunks, rebuild one profile + one text bundle per candidate."""
    profiles, texts = {}, {}
    for chunk in chunks:
        meta = chunk["metadata"]
        cid = meta["candidate_id"]
        if cid not in profiles:
            profiles[cid] = {
                "candidate_id": cid,
                "skills": meta.get("skills", []),
                "education": meta.get("education", []),
                "experience_years": meta.get("experience_years"),
            }
            texts[cid] = {"summary_text": "", "sections": {}}
        texts[cid]["sections"][chunk["metadata"]["section"]] = chunk["text"]
    for cid, t in texts.items():
        t["summary_text"] = " ".join(t["sections"].values())
    return profiles, texts


def build_index(chunks, embed_backend, store_backend):
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    embedder = get_embedder(embed_backend)
    if hasattr(embedder, "fit"):
        embedder.fit(texts)
    print(f"Embedding {len(texts)} chunks...")
    vectors = embedder.embed(texts)

    store = get_vector_store(store_backend)
    store.upsert(ids, vectors, metadatas)

    retriever = HybridRetriever(store, embedder)
    retriever.index_bm25(ids, texts)
    return retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--job-rules", required=True)
    ap.add_argument("--job-title", required=True)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--embed-backend", default="local_tfidf", choices=["local_tfidf", "openai", "gemini"])
    ap.add_argument("--store-backend", default="local", choices=["local", "pinecone"])
    ap.add_argument("--rerank-backend", default="local_lexical", choices=["local_lexical", "cross_encoder"])
    ap.add_argument("--explainer", default="local_template", choices=["local_template", "anthropic", "gemini"])
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    profiles, texts = build_candidate_views(chunks)

    with open(args.job_rules, encoding="utf-8") as f:
        job_rules = json.load(f)
    job_rule = next(
        (r for r in job_rules if r["job_title"].strip().lower() == args.job_title.strip().lower()), None
    )
    if not job_rule:
        raise SystemExit(f'No job role matching "{args.job_title}"')

    retriever = build_index(chunks, args.embed_backend, args.store_backend)
    reranker = get_reranker(args.rerank_backend)
    explainer = get_explainer(args.explainer)

    app = build_graph(retriever, reranker, explainer, profiles, texts, top_n=args.top_n)

    print(f"\nRunning LangGraph screening workflow for: {job_rule['job_title']}\n")
    final_state = app.invoke({"job_title": args.job_title, "job_rule": job_rule})

    report = final_state["final_report"]
    qualified = [r for r in report if r["status"] == "QUALIFIED"]
    rejected = [r for r in report if r["status"] == "REJECTED"]

    print(f"Shortlist size: {len(report)}  |  Qualified: {len(qualified)}  |  Rejected: {len(rejected)}\n")

    for r in qualified:
        print(f"{r['candidate_id']}  —  {r['score']}%  QUALIFIED")
        print(f"  {r['explanation']}")
        print()

    if rejected:
        print("--- Example rejected (from shortlist) ---")
        for r in rejected[:3]:
            print(f"{r['candidate_id']}  —  {r['score']}%  REJECTED")
            print(f"  {r['explanation']}")
            print()


if __name__ == "__main__":
    main()
