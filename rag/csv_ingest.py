"""
Loads the Kaggle training_data.csv (10,000 synthetic resumes) into the
shared ResumeRecord schema. Since the CSV already gives us Education /
Experience Years / Skills as clean structured columns, we don't need
heuristic parsing here — we just normalize field names and synthesize
short natural-language section texts so the SAME chunker used for PDFs
also works on this data.
"""

import csv
from .schema import ResumeRecord


def _parse_skills(raw: str) -> list[str]:
    return [s.strip() for s in raw.split("|") if s.strip()]


def load_training_csv(path: str) -> list[ResumeRecord]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            skills = _parse_skills(row.get("Skills", ""))
            education = row.get("Education", "").strip()
            try:
                exp_years = float(row.get("Experience Years", "") or 0)
            except ValueError:
                exp_years = None

            sections = {
                "summary": row.get("Resume Text", "").strip(),
                "education": f"Education: {education}" if education else "",
                "experience": f"Experience: {exp_years} years professional experience"
                if exp_years is not None else "",
                "skills": f"Skills: {', '.join(skills)}" if skills else "",
            }
            sections = {k: v for k, v in sections.items() if v}

            records.append(
                ResumeRecord(
                    candidate_id=row.get("Resume ID", "").strip(),
                    source="csv",
                    raw_text=row.get("Resume Text", "").strip(),
                    education=[education] if education else [],
                    experience_years=exp_years,
                    skills=skills,
                    job_role=row.get("Job Role", "").strip() or None,
                    category=row.get("Category", "").strip() or None,
                    sections=sections,
                )
            )
    return records
