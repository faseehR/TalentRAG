# TalentRAG — Run in VS Code, push to GitHub, deploy on Render

## A. Run locally in VS Code

1. Install **Python 3.12** and **VS Code** (with the *Python* extension).
2. Unzip the project, then VS Code → **File → Open Folder…** → pick `talentrag/`.
3. Open the terminal (**Ctrl+`**) and create an environment:
   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Ctrl+Shift+P → "Python: Select Interpreter"** → choose the `.venv` one.
5. Start the API (takes ~15 s to build the index — normal):
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
   (or press **F5** — a launch config is included in `.vscode/launch.json`).
   Check http://localhost:8000/health and http://localhost:8000/docs
6. Open the dashboard: right-click `frontend/dashboard.html` → **Open with
   Live Server** (install the *Live Server* extension), or just double-click
   the file. Top-right "API base" must say `http://localhost:8000` → **Connect**.

## B. Push to GitHub

1. Create an account at github.com, then **New repository** → name `talentrag`
   → leave "Add README" **unchecked** → Create.
2. In the VS Code terminal, inside the project folder:
   ```bash
   git init
   git add .
   git commit -m "TalentRAG: explainable resume screening RAG system"
   git branch -M main
   git remote add origin https://github.com/<YOUR-USERNAME>/talentrag.git
   git push -u origin main
   ```
   (VS Code can also do this: Source Control icon → Initialize Repository →
   Commit → Publish Branch.) `out/chunks.jsonl` is ~20 MB — fine for GitHub.

## C. Deploy the API on Render (free)

1. Sign up at render.com with your GitHub account.
2. **New + → Blueprint** → select your `talentrag` repo → Render reads
   `render.yaml` → click **Apply**. (Manual alternative: New + → Web Service →
   build `pip install -r requirements.txt`, start
   `uvicorn api.main:app --host 0.0.0.0 --port $PORT`, plan Free.)
3. Wait for the build (3–6 min). When it says **Live**, open
   `https://<your-service>.onrender.com/health` — you should see
   `jobs_loaded: 324, candidates_loaded: 10000, index_built: true`.
4. Open `frontend/dashboard.html`, paste the Render URL into **API base**,
   click **Connect**.

### Things to know on the free tier
- Sleeps after 15 min idle; first request afterwards takes ~15–60 s while it
  wakes and rebuilds the index. **Open the link yourself a minute before
  showing recruiters.**
- Disk is wiped on restart: uploaded resumes and screening history are lost
  (the 10,000 candidates + 324 jobs reload automatically).
- Memory: tuned to ~370 MB peak so it fits the 512 MB limit. Don't raise
  `TALENTRAG_TFIDF_FEATURES` or add heavy libraries (e.g. sentence-transformers)
  without moving to a paid instance.

## D. Optional: host the dashboard publicly too
`frontend/dashboard.html` is a static file. Put it in its own GitHub repo (or
a `docs/` folder) and enable **GitHub Pages**, or deploy via Render
**New + → Static Site**. Then recruiters can open one link. Remember it
still needs the API base set to your Render URL (change the default
`API_BASE` near the top of the `<script>` to your Render URL so it
connects automatically).

## E. Using Gemini (Google's AI) — free key

1. Go to **aistudio.google.com**, sign in with a Google account.
2. Click **Get API key** (left sidebar) → **Create API key** → copy it (starts with `AIza…`).
   Treat it like a password: never paste it into code, GitHub, or screenshots.
3. **Locally (VS Code):** `pip install google-genai`, then set the key in the terminal:
   - Windows PowerShell: `$env:GEMINI_API_KEY="AIza..."; $env:TALENTRAG_EXPLAIN_BACKEND="gemini"`
   - macOS/Linux: `export GEMINI_API_KEY="AIza..." TALENTRAG_EXPLAIN_BACKEND=gemini`
   then start `uvicorn` in the same terminal.
4. **On Render:** uncomment `google-genai` in `requirements.txt`, commit + push, then
   Render → your service → **Environment** → add `GEMINI_API_KEY` (your key) and
   `TALENTRAG_EXPLAIN_BACKEND` = `gemini`. Render redeploys automatically.

Notes: the free tier is rate-limited (~10 requests/min on Gemini 2.5 Flash); if a call
fails the app falls back to the built-in explanation instead of crashing. Google may use
free-tier prompts to improve its products — fine for this synthetic dataset, but do NOT
send real candidates' resumes through a free key.

**Gemini embeddings + Pinecone** (`TALENTRAG_EMBED_BACKEND=gemini`) are implemented but
NOT a simple env-var swap in the API: the API currently embeds all 40,000 chunks at every
startup, which would be slow and burn your quota on each Render wake-up. It needs a
one-time indexing script that uploads vectors to Pinecone, and an API mode that only
queries. That script is the next piece to build.

## F. Adding other API keys later
Render dashboard → your service → **Environment** → add `OPENAI_API_KEY`,
`PINECONE_API_KEY`, `ANTHROPIC_API_KEY` and change the `TALENTRAG_*_BACKEND`
values (see `.env.example`). Add the matching packages to `requirements.txt`.
