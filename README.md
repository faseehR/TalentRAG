# TalentRAG — Phase 1 + Phase 2: Ingestion & Chunking

This is the data-ingestion and section-aware chunking pipeline for the
TalentRAG resume-screening system, built against the Kaggle
`resume-dataset` (10,000 synthetic resumes + 324 job roles + skill
taxonomy).

## What it does

1. **Loads the 10,000-resume training set** (`training_data.csv`) and
   normalizes each row into a common `ResumeRecord` schema.
2. **Converts `job_roles.csv` (324 roles) into rule-engine JSON** —
   required skills, education, minimum experience, and default scoring
   weights (required 40% / preferred 20% / experience 15% / education
   10% / projects 10% / certifications 5%), matching the rule-engine
   design from the project plan.
3. **Section-aware chunks every resume** — one chunk per section
   (education / experience / skills / summary), each tagged with
   metadata (`candidate_id`, `section`, `skills`, `education`,
   `experience_years`, `job_role`, `category`) — ready to embed and
   upsert into Pinecone.
4. **Parses arbitrary new PDF resumes**, regardless of format/layout,
   into the exact same schema, so they flow through the same chunker.

## Why the PDF parser works "no matter the format"

Real resumes don't reliably have clean `Education:` / `Skills:`
headers the way the synthetic CSV data does. `rag/pdf_ingest.py` uses
**two independent strategies and merges them**, so it degrades
gracefully instead of failing:

- **Header-based sectioning** (best case) — detects common section
  headers (`Education`, `Experience`, `Skills`, `Summary`,
  `Projects`, `Certifications`, and common synonyms like `Core
  Competencies`) and splits the document there.
- **Full-text fallback extraction** (always runs) — regardless of
  whether headers were found, it scans the *entire* raw text against
  the 100+ skill taxonomy (`skills_database.json`) using
  word-boundary matching, plus regex for degree mentions and "X years
  experience" phrasing.

This means even a resume with **zero section headers** (tested below)
still gets its skills, education, and experience correctly extracted
and chunked, because the fallback doesn't depend on layout at all.

**Verified on two test cases during development:**
| Resume | Headers found | Skills extracted | Education | Experience |
|---|---|---|---|---|
| Standard, clearly labeled | summary/education/experience/skills | 5/5 correct | ✅ | 3.0 yrs ✅ |
| Single unstructured paragraph, no headers | none | 6/6 correct | ✅ | 4.0 yrs ✅ |

## Files

```
talentrag/
├── rag/
│   ├── schema.py            # Shared ResumeRecord — the format-agnostic contract
│   ├── skills_taxonomy.py   # Loads skill DB, does word-boundary skill matching
│   ├── csv_ingest.py        # Phase 1: loads training_data.csv
│   ├── job_rules.py         # Phase 1: job_roles.csv -> rule-engine JSON
│   ├── pdf_ingest.py        # Phase 1: arbitrary-format PDF resume parser
│   └── section_chunker.py   # Phase 2: ResumeRecord -> Pinecone-ready chunks
├── run_ingestion.py         # CLI orchestrator
├── out/
│   ├── chunks.jsonl         # 40,000 chunks from the full 10K-resume dataset
│   ├── job_rules.json       # 324 rule-engine definitions
│   └── ingestion_report.json
└── README.md
```

## Usage

```bash
pip install pdfplumber pypdf

python run_ingestion.py \
    --training-csv training_data.csv \
    --job-roles-csv job_roles.csv \
    --skills-list skills_list.csv \
    --skills-db skills_database.json \
    --pdf-dir uploaded_resumes/ \
    --out-dir out
```

Put any new resume PDFs (any format) in `--pdf-dir` and they'll be
parsed and chunked alongside the CSV data using the same pipeline —
you don't need to specify their format up front.

## Phase 3 — Embeddings, Retrieval & Reranking

```
rag/embeddings.py       # OpenAI (production) + local TF-IDF (no-key testing)
rag/vector_store.py     # Pinecone (production) + local numpy cosine store (no-key testing)
rag/hybrid_retrieval.py # dense + BM25 keyword search, merged via Reciprocal Rank Fusion
rag/reranker.py         # cross-encoder (production, bge-reranker-v2-m3) + lexical-overlap stand-in
run_retrieval.py         # CLI: builds the index from chunks.jsonl, runs a query through
                          #      dense-only vs hybrid vs hybrid+reranked side by side
```

Every layer is a small interface (`BaseEmbedder`, `BaseVectorStore`,
`BaseReranker`) with two implementations each: a **production**
backend (OpenAI/Pinecone/bge-reranker-v2-m3) and a **local, no-API-key**
backend (TF-IDF, in-memory cosine store, lexical overlap). This is
what let the whole pipeline be built and run end-to-end — including
on the full 40,000-chunk index from the 10K resumes — without needing
any external API keys yet. Swap `--embed-backend openai
--store-backend pinecone --rerank-backend cross_encoder` when you're
ready to point it at real infra; no other code changes.

```bash
pip install rank_bm25

python run_retrieval.py \
    --chunks out/chunks.jsonl \
    --query "Django PostgreSQL backend developer with Docker experience"
```

This prints the same top-5 results through all three stages so you
can literally see what hybrid search and reranking each change —
this is the evidence behind the "Recall@10 improves after reranking"
kind of claim your project plan wants on your GitHub README, once you
swap in the real embedder/reranker and measure it against
`test_resumes.json`.

**Honest caveat:** the local TF-IDF embedder is a functional stand-in,
not a semantic model — it's there so retrieval/hybrid/reranking logic
can be built and tested without API keys, not as a claim about
production-quality relevance. Swap in the real `openai` /
`cross_encoder` backends before measuring real Recall@K / NDCG numbers
for your portfolio.

## Phase 4 — Rule Engine

```
rag/rule_engine.py   # evaluate_candidate(job_rule, candidate) -> ScreeningResult
run_screening.py      # CLI: screens every candidate against a chosen job role, ranked leaderboard
```

This is the piece that keeps the system defensible: PASS/FAIL and the
match score are computed **deterministically from structured data**,
never by an LLM. Two rule types:

- **Hard rules** (cause rejection regardless of score): fewer than
  `minimum_required_skills` required skills matched, education not
  met, experience outside the required range.
- **Soft rules** (affect score only, never reject): preferred-skill
  coverage.

Scoring is a weighted sum (`required 40% / preferred 20% / experience
15% / education 10% / projects 10% / certifications 5%`), and weights
for dimensions with no data (this dataset has no projects/
certifications fields) are honestly dropped and the rest renormalized
to sum to 1 — rather than faking a placeholder score.

```bash
python run_screening.py \
    --chunks out/chunks.jsonl \
    --job-rules out/job_rules.json \
    --job-title "Software Engineer" \
    --top-n 10
```

**Verified on the full 10,000-resume dataset**: screening for
"Software Engineer" (requires Python, Java, C++, Git, Software
Design, Problem Solving + CS/Engineering degree + 2+ years) correctly
found 24/10,000 qualified candidates, and correctly rejected
otherwise-strong 88.5%-scoring candidates purely because they had 1
year of experience against a 2-year hard minimum — proving the hard
vs. soft rule distinction actually holds, not just scores everyone
highly.

## Phase 5 — LangGraph Orchestration + Evidence-Grounded Explanation

```
agents/state.py         # ScreeningState — the shared state that flows through the graph
agents/graph.py         # build_graph(): the actual LangGraph StateGraph, node by node
rag/llm_explainer.py    # Claude/Anthropic (production) + local template stand-in
run_agent.py             # CLI: runs the full workflow end-to-end for a chosen job title
```

This wires Phases 2–4 into one stateful workflow matching the
architecture diagram from the plan:

```
parse_job -> retrieve -> rerank -> apply_rules
                                        |
                    (real conditional edge, not an if-statement:
                     skips the entire evidence+LLM path when the
                     shortlist has zero qualified candidates)
                                        |
              qualified --> collect_evidence -> deep_evaluate --+
                  |no                                            |
                  +------------------> explain_rejected <--------+
                                              |
                                        generate_report -> END
```

The LLM (`rag/llm_explainer.py`) never decides PASS/FAIL or the score
— `rag/rule_engine.py` already did that deterministically in the
`apply_rules` node. The explainer is only ever handed the rule
engine's already-decided result plus retrieved evidence text, and its
system prompt explicitly forbids inferring anything not in that
evidence (production `AnthropicExplainer`); the local
`LocalTemplateExplainer` stand-in enforces the same grounding rule by
construction (it can literally only reference fields it was passed).

```bash
pip install langgraph langchain-core

python run_agent.py \
    --chunks out/chunks.jsonl \
    --job-rules out/job_rules.json \
    --job-title "Software Engineer" \
    --top-n 15
```

**Verified end-to-end on the full 40,000-chunk index**: retrieval →
reranking → rule engine → evidence collection → explanation all ran
through the compiled graph, producing a 15-candidate shortlist (11
qualified / 4 rejected) with per-candidate grounded explanations. The
rejected candidates are a good sanity check: they show 88.5% scores
with all 6 required skills matched, but are still correctly REJECTED
because the graph enforces the rule engine's hard experience
requirement rather than letting a high score override it.

Swap `--explainer anthropic` (+ `ANTHROPIC_API_KEY`) for real
LLM-generated prose explanations in production — the rest of the
graph doesn't change.

## Phase 6 — Evaluation Harness

```
rag/eval_metrics.py           # Recall@K, Precision@K, MRR, NDCG, Precision/Recall/F1
run_evaluation.py              # A) retrieval quality  B) rule-engine role-prediction accuracy
build_extended_taxonomy.py     # diagnostic fix discovered by running (A)+(B) — see below
out/eval_report.json           # full machine-readable results
```

Two independent evaluations, both against **real** ground truth — no
invented numbers:

**A) Retrieval quality (dense vs hybrid vs hybrid+reranked).**
Ground truth = the rule engine's own QUALIFIED set for 20 sampled job
roles (a full scan over all 10,000 candidates), which is exactly the
"build your own ground truth" approach from the plan.

| Method | Recall@10 | Precision@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| Dense only | 0.316 | 0.835 | 1.000 | 0.848 |
| Hybrid (RRF) | 0.311 | 0.825 | 0.925 | 0.826 |
| Hybrid + reranked | 0.311 | 0.825 | 0.925 | 0.826 |

**Honest read of these numbers**, not a marketing spin: with the local
TF-IDF embedder, dense-only already scores highest here because the
resumes are short, keyword-dense synthetic strings — hybrid/BM25 and
the lexical reranker aren't adding much on data this small and
literal. That's expected and worth stating plainly on a GitHub README
rather than hiding it: hybrid search and cross-encoder reranking earn
their keep on longer, noisier real-world resumes (and with a real
semantic embedder instead of TF-IDF) — this local benchmark is a
sanity check that the pipeline computes correctly, not a claim that
hybrid+rerank beats dense here. Re-run with `--embed-backend openai
--rerank-backend cross_encoder` once you have API keys for numbers
that are actually representative.

**B) Rule-engine role-prediction accuracy**, using the 8 hand-labeled
resumes in `test_resumes.json` (`expected_roles` as ground truth),
parsed through the same format-agnostic text parser used for PDFs:

| | Precision | Recall | F1 |
|---|---|---|---|
| Original taxonomy (115 skills) | 0.125 | 0.042 | 0.062 |
| Extended taxonomy (793 skills) | 0.250 | 0.083 | 0.125 |

Running this evaluation surfaced a genuine bug worth having found
before shipping: `skills_database.json` only covers 115 skills, but
`training_data.csv` + `job_roles.csv` actually use **793** unique
skill strings between them — an 85% coverage gap. Trade/domain skills
like "Safety", "Troubleshooting", and "Electrical Work" were silently
invisible to free-text extraction even when present verbatim in a
resume. `build_extended_taxonomy.py` mines the corpus's own
vocabulary and merges it in, which doubled F1. The remaining gap is a
different, harder problem: `job_roles.csv` says "Electrical Work"
while a resume says "Electrical Systems" — synonyms that exact
word-boundary matching still can't bridge. The natural next fix is
fuzzy/embedding-based skill matching instead of exact string match.

```bash
python run_evaluation.py \
    --chunks out/chunks.jsonl --job-rules out/job_rules.json \
    --test-resumes test_resumes.json \
    --skills-list skills_list.csv --skills-db out/skills_database_extended.json \
    --num-roles 20 --out out/eval_report.json
```

This — a real number, a real bug found by measuring instead of
assuming, and a real fix with a before/after — is worth more on a
portfolio than a claimed "94% accuracy" with no evaluation code behind
it.

## Phase 7 — FastAPI Layer

```
api/db.py     # SQLite persistence for jobs/candidates/screening results
              # (local stand-in for the plan's PostgreSQL — same swap-later pattern as every other layer)
api/main.py   # FastAPI app: wraps ingestion, retrieval, rule engine, and the LangGraph agent as REST endpoints
```

```bash
pip install fastapi uvicorn python-multipart
uvicorn api.main:app --reload --port 8000
# interactive docs: http://localhost:8000/docs
```

On startup it loads `out/job_rules.json` (324 roles) and
`out/chunks.jsonl` (10,000 candidates) into SQLite, and builds the
same hybrid retrieval index used in Phases 3-6, in memory.

| Endpoint | What it does |
|---|---|
| `GET /health` | jobs/candidates loaded, index status |
| `GET /jobs`, `GET /jobs/{id}` | browse the 324 predefined roles |
| `POST /jobs` | create a custom job with its own required/preferred skills |
| `GET /candidates`, `GET /candidates/{id}` | browse candidates |
| `GET /candidates/{id}/evidence?job_id=` | matched-skill evidence snippets for one candidate against one job |
| `POST /resumes/upload` | upload **any-format PDF resume** — parsed (Phase 1), chunked (Phase 2), and added to the **live** retrieval index immediately |
| `POST /screening/{job_id}` | runs the full Phase 5 LangGraph workflow for a job, persists the ranked report |
| `GET /screening/{job_id}/results` | fetch the latest stored screening run |
| `POST /chat` | RAG chat ("Mode 2" from the plan) — retrieve relevant resume evidence for a free-text question, optionally scoped to candidates qualified for a given job |

**Verified end-to-end with FastAPI's TestClient** (not just "it
imports"): `/health` correctly reports 324 jobs / 10,000 candidates
loaded and the index built; `POST /screening/software-engineer`
returned a 10-candidate ranked shortlist and `GET .../results`
correctly returned all 10 stored rows (this caught and fixed a real
bug — the first version only persisted 1 of 10 results because each
row got its own timestamp instead of one per screening run); a
generated test PDF uploaded via `/resumes/upload` was correctly
parsed (all 5 skills, education, 3.0 years experience), and was
immediately findable through `/chat` in the same process — proving
new resumes don't need a separate re-ingestion step to become
searchable.

Swap `TALENTRAG_EMBED_BACKEND=openai`, `TALENTRAG_STORE_BACKEND=pinecone`,
`TALENTRAG_RERANK_BACKEND=cross_encoder`, `TALENTRAG_EXPLAIN_BACKEND=anthropic`
env vars for the production backends — no route code changes needed.

## Phase 8 — Recruiter Dashboard

```
frontend/dashboard.html   # single-file HTML/CSS/JS, no build step
```

A static dashboard (not Next.js — zero build step for a faster
portfolio demo; see `CONTEXT.md` for the trade-off if you want to
port it later) calling the FastAPI backend directly. Design: cool
off-white background, indigo accent, status color coded by meaning
(green = qualified, red = rejected, amber = preferred-skill gaps) —
Space Grotesk for headings, Inter for body/data, deliberately avoiding
the generic "AI dashboard" defaults (no cream+terracotta, no
near-black+neon, no identical shadowed cards everywhere).

Six views: **Overview** (live stats + one-click screening),
**Jobs** (browse/search all 324 roles), **Screening** (ranked results
with per-candidate evidence you can expand inline), **Candidates**
(lookup + browse, including your own uploads), **Evidence Chat**
(ask a question, get back the actual resume snippet that answers it,
optionally scoped to one job's qualified candidates), **Upload**
(drag-and-drop a PDF, see it parsed and immediately searchable).

```bash
uvicorn api.main:app --reload --port 8000   # backend
# then just open frontend/dashboard.html in a browser
# (or python -m http.server 8020 inside frontend/, then visit it)
```

**Verified against the live backend**, not just visually reviewed:
started the real FastAPI server, hit every endpoint the dashboard
calls with the exact same request shapes (`/health`, `/jobs`,
`POST /screening/{id}`, `/candidates`, `/candidates/{id}/evidence`,
`POST /chat`), and confirmed CORS headers are actually present —
this caught that the API needed `CORSMiddleware` added (a static HTML
file calling a different-origin API fails silently without it).
JS was also syntax-checked and the HTML tag balance verified.

## Project context file

**`CONTEXT.md`** — a requirements-vs-completed checklist, the
architecture decisions made and why (local stand-ins vs production
backends, SQLite vs PostgreSQL, static HTML vs Next.js), the honest
evaluation findings, and exact commands to resume. Paste it at the
start of a new conversation instead of re-sharing this whole project's
history — keeps future sessions token-efficient.

## What's next

- Swap in real API keys (`OPENAI_API_KEY` / `PINECONE_API_KEY` /
  `ANTHROPIC_API_KEY`) and re-run `run_evaluation.py` for numbers
  worth publishing.
- JWT auth before any public deployment.
- Fuzzy/semantic skill matching (the clearest remaining accuracy lever).
- Deploy: dashboard → Vercel/static host, API → Render/Railway, Pinecone → cloud.
