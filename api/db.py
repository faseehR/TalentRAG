"""
Application-data persistence: jobs, candidates, screening results.

This is SQLite rather than the PostgreSQL from the original plan, for
the same reason the rest of this project uses local stand-ins:
zero external services needed to run and test the API right now.
The schema is deliberately PostgreSQL-compatible (plain columns, JSON
stored as TEXT) so migrating is a schema-translation exercise, not a
redesign — swap this module for a SQLAlchemy/asyncpg version pointed
at real PostgreSQL for production, nothing above the DB layer changes.
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "talentrag.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT,
    required_skills TEXT NOT NULL,   -- JSON list
    preferred_skills TEXT NOT NULL,  -- JSON list
    education TEXT NOT NULL,         -- JSON list
    minimum_experience REAL,
    maximum_experience REAL,
    minimum_required_skills INTEGER,
    weights TEXT NOT NULL,           -- JSON dict
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    name TEXT,
    source TEXT NOT NULL,            -- "csv" | "pdf"
    skills TEXT NOT NULL,            -- JSON list
    education TEXT NOT NULL,         -- JSON list
    experience_years REAL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS screening_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL NOT NULL,
    result_json TEXT NOT NULL,       -- full ScreeningResult + explanation + evidence
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_screening_job ON screening_results(job_id);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def job_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) c FROM jobs").fetchone()["c"]


def candidate_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) c FROM candidates").fetchone()["c"]


def insert_job(job: dict):
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO jobs
               (id, title, category, required_skills, preferred_skills, education,
                minimum_experience, maximum_experience, minimum_required_skills, weights, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                job["id"], job["job_title"], job.get("category"),
                json.dumps(job.get("required_skills", [])),
                json.dumps(job.get("preferred_skills", [])),
                json.dumps(job.get("education", [])),
                job.get("minimum_experience"), job.get("maximum_experience"),
                job.get("minimum_required_skills"), json.dumps(job.get("weights", {})),
                time.time(),
            ),
        )


def get_job(job_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _job_row_to_rule(row) if row else None


def list_jobs(limit: int = 50, offset: int = 0, category: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if category:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE category = ? ORDER BY title LIMIT ? OFFSET ?",
                (category, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY title LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [_job_row_to_rule(r) for r in rows]


def _job_row_to_rule(row) -> dict:
    return {
        "id": row["id"],
        "job_title": row["title"],
        "category": row["category"],
        "required_skills": json.loads(row["required_skills"]),
        "preferred_skills": json.loads(row["preferred_skills"]),
        "education": json.loads(row["education"]),
        "minimum_experience": row["minimum_experience"],
        "maximum_experience": row["maximum_experience"],
        "minimum_required_skills": row["minimum_required_skills"],
        "weights": json.loads(row["weights"]),
    }


def insert_candidate(candidate: dict):
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO candidates
               (id, name, source, skills, education, experience_years, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                candidate["candidate_id"], candidate.get("name"), candidate.get("source", "csv"),
                json.dumps(candidate.get("skills", [])), json.dumps(candidate.get("education", [])),
                candidate.get("experience_years"), time.time(),
            ),
        )


def get_candidate(candidate_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        return _candidate_row_to_dict(row) if row else None


def list_candidates(limit: int = 50, offset: int = 0) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM candidates ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        return [_candidate_row_to_dict(r) for r in rows]


def _candidate_row_to_dict(row) -> dict:
    return {
        "candidate_id": row["id"],
        "name": row["name"],
        "source": row["source"],
        "skills": json.loads(row["skills"]),
        "education": json.loads(row["education"]),
        "experience_years": row["experience_years"],
    }


def insert_screening_results(job_id: str, report: list[dict]):
    run_time = time.time()  # single timestamp for the whole batch, so a later
                             # "latest run" lookup groups all of them together
    with get_conn() as conn:
        for r in report:
            conn.execute(
                """INSERT INTO screening_results (job_id, candidate_id, status, score, result_json, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (job_id, r["candidate_id"], r["status"], r["score"], json.dumps(r), run_time),
            )


def get_latest_screening_results(job_id: str) -> list[dict]:
    with get_conn() as conn:
        latest_run = conn.execute(
            "SELECT MAX(created_at) t FROM screening_results WHERE job_id = ?", (job_id,)
        ).fetchone()["t"]
        if latest_run is None:
            return []
        rows = conn.execute(
            """SELECT * FROM screening_results WHERE job_id = ? AND created_at = ?
               ORDER BY score DESC""",
            (job_id, latest_run),
        ).fetchall()
        return [json.loads(r["result_json"]) for r in rows]
