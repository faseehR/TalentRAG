"""
Phase 7: FastAPI layer.

Wraps everything built in Phases 1-6 (ingestion, chunking, hybrid
retrieval, reranking, rule engine, LangGraph orchestration, evidence-
grounded explanation) behind a REST API a frontend/dashboard can call.

On startup:
  - loads out/job_rules.json (324 roles) into SQLite, if not already loaded
  - loads out/chunks.jsonl candidate profiles into SQLite, if not already loaded
  - builds the in-memory hybrid retrieval index from out/chunks.jsonl
    (same local TF-IDF/BM25 backends as Phases 3-6 — swap via env vars
    for production Pinecone/OpenAI, same as every other script here)

Run:
    uvicorn api.main:app --reload --port 8000

Then see interactive docs at http://localhost:8000/docs
"""

import json
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api import db
from rag.embeddings import get_embedder
from rag.vector_store import get_vector_store
from rag.hybrid_retrieval import HybridRetriever
from rag.reranker import get_reranker
from rag.llm_explainer import get_explainer
from rag.rule_engine import evaluate_candidate
from rag.skills_taxonomy import SkillsTaxonomy
from rag.pdf_ingest import parse_resume_pdf
from rag.section_chunker import chunk_resume
from agents.graph import build_graph

DATA_DIR = Path(os.environ.get("TALENTRAG_DATA_DIR", Path(__file__).parent.parent))
OUT_DIR = DATA_DIR / "out"

EMBED_BACKEND = os.environ.get("TALENTRAG_EMBED_BACKEND", "local_tfidf")
STORE_BACKEND = os.environ.get("TALENTRAG_STORE_BACKEND", "local")
RERANK_BACKEND = os.environ.get("TALENTRAG_RERANK_BACKEND", "local_lexical")
EXPLAIN_BACKEND = os.environ.get("TALENTRAG_EXPLAIN_BACKEND", "local_template")

app = FastAPI(title="TalentRAG API", version="0.1.0")

# Wide open for local dev so the static dashboard (opened from disk or a
# different port) can call this API. Lock this down to your actual
# frontend origin before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated at startup; module-level so route handlers can use them.
state = {"retriever": None, "reranker": None, "explainer": None, "taxonomy": None,
         "text_by_id": {}, "profiles": {}, "texts": {}}


# ----------------------------- startup ----------------------------- #

@app.on_event("startup")
def startup():
    db.init_db()

    if db.job_count() == 0:
        job_rules_path = OUT_DIR / "job_rules.json"
        if job_rules_path.exists():
            job_rules = json.loads(job_rules_path.read_text())
            for r in job_rules:
                r["id"] = _slug(r["job_title"])
                db.insert_job(r)

    chunks_path = OUT_DIR / "chunks.jsonl"

    # Stream the file once, line by line, so we never hold all 40k chunk dicts
    # in memory at the same time (matters on small hosts like Render's free tier).
    profiles, texts = {}, {}
    ids, texts_list, metadatas = [], [], []
    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                meta = chunk["metadata"]
                cid = meta["candidate_id"]
                if cid not in profiles:
                    profiles[cid] = {
                        "candidate_id": cid, "skills": meta.get("skills", []),
                        "education": meta.get("education", []), "experience_years": meta.get("experience_years"),
                    }
                    texts[cid] = {"summary_text": "", "sections": {}}
                texts[cid]["sections"][meta["section"]] = chunk["text"]
                ids.append(chunk["id"])
                texts_list.append(chunk["text"])
                # The API only needs candidate_id + section per vector.
                metadatas.append({"candidate_id": cid, "section": meta["section"]})
    for cid, t in texts.items():
        t["summary_text"] = " ".join(t["sections"].values())

    if db.candidate_count() == 0:
        for p in profiles.values():
            db.insert_candidate(p)

    state["profiles"], state["texts"] = profiles, texts

    if ids:
        embed_kwargs = {}
        if EMBED_BACKEND == "local_tfidf":
            # Smaller feature space keeps memory under Render's 512MB free tier.
            embed_kwargs["max_features"] = int(os.environ.get("TALENTRAG_TFIDF_FEATURES", "512"))
        embedder = get_embedder(EMBED_BACKEND, **embed_kwargs)
        if hasattr(embedder, "fit"):
            embedder.fit(texts_list)
        store = get_vector_store(STORE_BACKEND)
        if hasattr(store, "expect"):
            store.expect(len(ids) + 5000)  # headroom for resumes uploaded later
        # Embed + upsert in batches so we never hold the full dense matrix as Python lists.
        batch = 2000
        for i in range(0, len(texts_list), batch):
            vecs = embedder.embed(texts_list[i:i + batch])
            store.upsert(ids[i:i + batch], vecs, metadatas[i:i + batch])
            del vecs
        if hasattr(store, "vectors"):
            _ = store.vectors  # merge now, while memory use is still low (before BM25 is built)
        retriever = HybridRetriever(store, embedder)
        retriever.index_bm25(ids, texts_list)
        state["retriever"] = retriever
        state["embedder"] = embedder
        state["text_by_id"] = dict(zip(ids, texts_list))
        del texts_list, metadatas
        import gc; gc.collect()

    state["reranker"] = get_reranker(RERANK_BACKEND)
    state["explainer"] = get_explainer(EXPLAIN_BACKEND)

    skills_list = DATA_DIR / "skills_list.csv"
    skills_db = DATA_DIR / "skills_database.json"
    extended = OUT_DIR / "skills_database_extended.json"   # 793 skills mined from the dataset (see Phase 6)
    if extended.exists():
        skills_db = extended
    if skills_list.exists() and skills_db.exists():
        state["taxonomy"] = SkillsTaxonomy(str(skills_list), str(skills_db))


def _slug(title: str) -> str:
    return title.strip().lower().replace(" ", "-").replace("/", "-")


def _require_retriever():
    if state["retriever"] is None:
        raise HTTPException(503, "Index not built — no out/chunks.jsonl found. Run run_ingestion.py first.")


# ----------------------------- schemas ----------------------------- #

class JobCreate(BaseModel):
    job_title: str
    category: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    minimum_experience: float = 0
    maximum_experience: float | None = None


class ChatRequest(BaseModel):
    query: str
    job_id: str | None = None
    top_k: int = 5


# ----------------------------- routes ----------------------------- #

@app.get("/health")
def health():
    return {
        "status": "ok",
        "jobs_loaded": db.job_count(),
        "candidates_loaded": db.candidate_count(),
        "index_built": state["retriever"] is not None,
    }


@app.get("/jobs")
def list_jobs(limit: int = 50, offset: int = 0, category: str | None = None):
    return db.list_jobs(limit=limit, offset=offset, category=category)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/jobs")
def create_job(job: JobCreate):
    job_id = _slug(job.job_title) + "-" + uuid.uuid4().hex[:6]
    rule = job.model_dump()
    rule["id"] = job_id
    rule["minimum_required_skills"] = max(1, round(len(rule["required_skills"]) * 0.6))
    rule["weights"] = {
        "required_skills": 0.40, "preferred_skills": 0.20, "experience": 0.15,
        "education": 0.10, "projects": 0.10, "certifications": 0.05,
    }
    db.insert_job(rule)
    return db.get_job(job_id)


@app.get("/candidates")
def list_candidates(limit: int = 50, offset: int = 0):
    return db.list_candidates(limit=limit, offset=offset)


@app.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str):
    c = db.get_candidate(candidate_id)
    if not c:
        raise HTTPException(404, "Candidate not found")
    return c


@app.get("/candidates/{candidate_id}/evidence")
def get_candidate_evidence(candidate_id: str, job_id: str):
    """Which resume text supports each matched skill for this candidate against this job."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    candidate = db.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(404, "Candidate not found")

    result = evaluate_candidate(job, candidate)
    sections = state["texts"].get(candidate_id, {}).get("sections", {})
    evidence = {}
    for skill in result.required_skills_matched + result.preferred_skills_matched:
        for section_text in sections.values():
            if skill.lower() in section_text.lower():
                evidence[skill] = section_text
                break
    return {"candidate_id": candidate_id, "job_id": job_id,
            "screening_result": result.to_dict(), "evidence": evidence}


@app.post("/resumes/upload")
async def upload_resume(file: UploadFile = File(...)):
    """Parses an arbitrary-format PDF resume (rag/pdf_ingest.py), chunks it
    (rag/section_chunker.py), and adds it to both SQLite and the live
    retrieval index — so it's searchable/screenable immediately."""
    if state["taxonomy"] is None:
        raise HTTPException(503, "Skill taxonomy not loaded")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    candidate_id = f"upload-{uuid.uuid4().hex[:10]}"
    try:
        record = parse_resume_pdf(tmp_path, state["taxonomy"], candidate_id=candidate_id)
    finally:
        os.unlink(tmp_path)

    chunks = chunk_resume(record)
    profile = {
        "candidate_id": candidate_id, "skills": record.skills,
        "education": record.education, "experience_years": record.experience_years,
        "source": "pdf", "name": record.name,
    }
    db.insert_candidate(profile)
    state["profiles"][candidate_id] = profile
    state["texts"][candidate_id] = {
        "summary_text": " ".join(record.sections.values()), "sections": record.sections,
    }

    if state["retriever"] is not None and chunks:
        ids = [c["id"] for c in chunks]
        texts_list = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        vectors = state["embedder"].embed(texts_list)
        state["retriever"].vector_store.upsert(ids, vectors, metadatas)
        for i, t in zip(ids, texts_list):
            state["text_by_id"][i] = t
        # BM25 needs the full corpus retokenized; acceptable to rebuild
        # lazily/periodically in production rather than per-upload.

    return {"candidate_id": candidate_id, "name": record.name, "skills": record.skills,
            "education": record.education, "experience_years": record.experience_years,
            "sections_found": list(record.sections.keys())}


@app.post("/screening/{job_id}")
def run_screening(job_id: str, top_n: int = 15):
    """Runs the full LangGraph workflow (Phase 5) for this job and persists the report."""
    _require_retriever()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    graph = build_graph(
        state["retriever"], state["reranker"], state["explainer"],
        state["profiles"], state["texts"], top_n=top_n,
    )
    result = graph.invoke({"job_title": job["job_title"], "job_rule": job})
    report = result["final_report"]
    db.insert_screening_results(job_id, report)
    return {"job_id": job_id, "shortlist_size": len(report), "results": report}


@app.get("/screening/{job_id}/results")
def get_screening_results(job_id: str):
    if not db.get_job(job_id):
        raise HTTPException(404, "Job not found")
    return db.get_latest_screening_results(job_id)


@app.post("/chat")
def chat(req: ChatRequest):
    """RAG chat ('Mode 2' from the plan): retrieves relevant resume chunks
    and answers grounded in that retrieved evidence, optionally
    restricted to candidates who qualify for a given job."""
    _require_retriever()
    hits = state["retriever"].search(req.query, top_k=req.top_k * 4)

    allowed_ids = None
    if req.job_id:
        job = db.get_job(req.job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        allowed_ids = {
            cid for cid, profile in state["profiles"].items()
            if evaluate_candidate(job, profile).status == "QUALIFIED"
        }

    results = []
    seen = set()
    for h in hits:
        cid = h["id"].split("__")[0]
        if allowed_ids is not None and cid not in allowed_ids:
            continue
        if cid in seen:
            continue
        seen.add(cid)
        results.append({
            "candidate_id": cid,
            "matched_text": state["text_by_id"].get(h["id"], ""),
            "score": h.get("rrf_score"),
        })
        if len(results) >= req.top_k:
            break

    return {"query": req.query, "job_id": req.job_id, "results": results}
