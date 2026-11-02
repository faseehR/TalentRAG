# TalentRAG — Project Context

**Paste this whole file at the start of a new conversation** instead of
re-sharing the full project history — it has everything needed to
keep working without burning tokens re-deriving decisions already made.

## What this is

An explainable AI resume-screening RAG system: rule-guided retrieval +
reranking + a deterministic rule engine + evidence-grounded LLM
explanation, orchestrated with LangGraph, exposed via FastAPI, with a
recruiter dashboard on top. Built as a portfolio project targeting AI
developer roles in Lahore, Pakistan. Dataset: Kaggle
`trendcart/resume-dataset` (10,000 synthetic resumes, 324 job roles,
a 115-skill taxonomy, 8 hand-labeled test resumes).

## Requirements checklist

| # | Requirement (from original brief) | Status |
|---|---|---|
| 1 | RAG system with rules, screens resumes | ✅ Done |
| 2 | Chunking | ✅ Section-aware (education/experience/skills/summary), not fixed-size |
| 3 | Reranking | ✅ Interface + local lexical-overlap (testing) + cross-encoder/bge-reranker-v2-m3 (production, needs API) |
| 4 | Pinecone as vector DB | ✅ Real Pinecone client implemented (`rag/vector_store.py`); **not yet tested live — needs your API key** |
| 5 | LangChain / LangGraph | ✅ LangGraph orchestration (`agents/graph.py`) with a real conditional branch |
| 6 | Rule engine separate from LLM | ✅ `rag/rule_engine.py` — deterministic, LLM only explains, never decides |
| 7 | Explainability / evidence-grounded output | ✅ `rag/llm_explainer.py`, restricted to supplied evidence only |
| 8 | Hybrid retrieval (semantic + keyword) | ✅ Dense + BM25, merged via Reciprocal Rank Fusion |
| 9 | Weighted deterministic scoring | ✅ `required 40 / preferred 20 / experience 15 / education 10 / projects 10 / certifications 5`, renormalized when a dimension has no data |
| 10 | Evaluation (Recall@K, MRR, NDCG, groundedness) | ✅ `run_evaluation.py` — real numbers measured, not invented (see below) |
| 11 | PDF resume parser, any format | ✅ Header-based + full-text-fallback extraction; tested on a header-less resume |
| 12 | Large resume dataset sourced | ✅ Using the dataset you uploaded (`archive.zip` → 10K resumes) |
| 13 | Recruiter dashboard, modern UI | ✅ `frontend/dashboard.html` — this delivery |
| 14 | FastAPI backend | ✅ `api/main.py` — 9 endpoints, CORS-enabled for the dashboard |
| 15 | Auth (JWT) | ❌ Not built — API is currently open, local-dev only |
| 16 | Next.js frontend (as originally planned) | ⚠️ Built as static HTML/CSS/JS instead (zero build step, faster to ship); portable to Next.js later if wanted |
| 17 | PostgreSQL for app data | ⚠️ SQLite used instead (`api/db.py`), schema written PG-compatible |
| 18 | Deployment (Vercel/Render/AWS) | ❌ Not done — runs locally only so far |
| 19 | Production API keys (OpenAI/Pinecone/Claude) | ❌ You said you'll swap these in later — every layer already has the production code path, gated behind env vars/CLI flags |

## Key architecture decision: local stand-ins everywhere

Every external-dependency layer has **two backends behind one
interface**: a production backend (OpenAI, Pinecone, `bge-reranker-v2-m3`,
Claude) and a local, no-API-key stand-in (TF-IDF, in-memory numpy
vector store, lexical-overlap reranker, template explainer). This is
why the whole pipeline could be **built and actually tested
end-to-end**, on the real 10,000-resume dataset, without any API keys
yet — not just written and hoped to work.

**When you add your API keys**, swap:
- `run_agent.py` / `run_evaluation.py` / `api/main.py`: `--embed-backend openai --store-backend pinecone --rerank-backend cross_encoder --explainer anthropic` (CLI flags for scripts; `TALENTRAG_EMBED_BACKEND` etc. env vars for the API)
- Nothing else changes — retrieval, rule engine, LangGraph, API routes, and the dashboard are all backend-agnostic.

## Honest findings from evaluation (Phase 6) — don't lose these for your README

- The provided `skills_database.json` only covers 115 of the 793
  unique skills actually used in the dataset (85% gap). Fixed via
  `build_extended_taxonomy.py`, which mines the corpus's own
  vocabulary — doubled role-prediction F1 (0.062 → 0.125).
- Remaining gap: exact string matching still misses synonyms (e.g.
  job_roles.csv says "Electrical Work", a resume says "Electrical
  Systems"). Next fix: fuzzy/embedding-based skill matching.
- Retrieval quality numbers (Recall@10, NDCG@10, etc. in `out/eval_report.json`)
  were measured with the **local TF-IDF stand-in embedder**, not a
  real semantic model — don't publish these as your portfolio numbers.
  Re-run `run_evaluation.py` with `--embed-backend openai` once you
  have a key, for numbers that actually mean something.

## File map

```
talentrag/
├── rag/                    # Phases 1-6: ingestion → chunking → embeddings →
│                            #   vector store → hybrid retrieval → reranking →
│                            #   rule engine → LLM explainer → eval metrics
├── agents/                 # Phase 5: LangGraph state + orchestration graph
├── api/                    # Phase 7: FastAPI app + SQLite persistence
├── frontend/dashboard.html # Phase 8: recruiter dashboard (this delivery)
├── run_ingestion.py        # CLI: build chunks.jsonl + job_rules.json
├── run_retrieval.py        # CLI: dense vs hybrid vs reranked demo
├── run_screening.py        # CLI: rule-engine-only screening (no LLM)
├── run_agent.py            # CLI: full LangGraph workflow for one job
├── run_evaluation.py       # CLI: retrieval quality + role-prediction eval
├── build_extended_taxonomy.py
└── out/                    # chunks.jsonl, job_rules.json, eval_report.json (real run outputs)
```

## How to run everything

```bash
pip install pdfplumber pypdf rank_bm25 langgraph langchain-core fastapi uvicorn python-multipart

# 1. Rebuild the data pipeline outputs (or just use out/ as-is, already run on full dataset)
python run_ingestion.py --training-csv training_data.csv --job-roles-csv job_roles.csv \
    --skills-list skills_list.csv --skills-db skills_database.json --out-dir out

# 2. Start the API (loads out/, builds the retrieval index)
uvicorn api.main:app --reload --port 8000

# 3. Open the dashboard
open frontend/dashboard.html   # or serve it: python -m http.server 8020, then visit the page
#   it points at http://localhost:8000 by default — change the "API base" field at top right if different
```

Startup takes ~10-15 seconds (embedding all 40,000 chunks with
TF-IDF) — that's expected, not a hang.

## Deployment status & lessons (latest)

- Backend target: **Render free tier** (chosen over Vercel — serverless can't hold the in-memory index or SQLite; Railway is the upgrade path if it needs to be always-on). Frontend stays a static file.
- Render's free tier has 512 MB RAM. The original startup peaked at **1,739 MB**; fixed to **~370 MB** by streaming `chunks.jsonl`, embedding in batches, `TALENTRAG_TFIDF_FEATURES=512`, trimmed vector metadata, a preallocated vector buffer, and dropping sklearn's `stop_words_`. Re-measure if you add anything heavy.
- The API now uses `out/skills_database_extended.json` (793 skills). Known quirk: generic skills like "Software" can appear as extracted skills.
- GitHub push + Render deploy are done by the user (no GitHub connector available); steps are in `DEPLOY.md`.

- **Gemini added** (`rag/llm_explainer.py: GeminiExplainer`, `rag/embeddings.py: GeminiEmbedder`; model defaults `gemini-2.5-flash`, `gemini-embedding-001` @768 dims). Tested with a mocked client only — not yet against the real Gemini API. Explainer falls back to the template on errors/rate limits.
- **Correction to earlier docs:** swapping *explainer* backends is a pure env-var change, but swapping *embeddings/Pinecone* is NOT — the API embeds all chunks at startup. TODO: one-time `run_index_pinecone.py` + query-only API mode.

## Next steps, in priority order

1. Swap in `OPENAI_API_KEY` / `PINECONE_API_KEY` / `ANTHROPIC_API_KEY`, re-run `run_evaluation.py`, get real numbers for your README.
2. Add JWT auth before deploying anywhere public.
3. Fuzzy/semantic skill matching in the rule engine.
4. Deploy: frontend → Vercel (or keep static), backend → Render/Railway, Pinecone → cloud.
5. Optional: port `frontend/dashboard.html` to Next.js if you want the originally-planned stack for the README, or keep it as-is (it works, and "no unnecessary complexity" is also a legitimate engineering choice worth defending to an interviewer).
