"""
Converts job_roles.csv (324 predefined roles) into the deterministic
rule-engine format described in the project plan: required skills,
education, experience bounds, plus default scoring weights. This is
what the rule engine (not the LLM) uses to decide PASS/FAIL and to
compute the objective part of the match score.
"""

import csv

DEFAULT_WEIGHTS = {
    "required_skills": 0.40,
    "preferred_skills": 0.20,
    "experience": 0.15,
    "education": 0.10,
    "projects": 0.10,
    "certifications": 0.05,
}


def load_job_roles(path: str) -> list[dict]:
    roles = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            required_skills = [
                s.strip() for s in row.get("Required Skills", "").split("|") if s.strip()
            ]
            education_options = [
                e.strip() for e in row.get("Education Requirement", "").split("|") if e.strip()
            ]
            try:
                min_exp = float(row.get("Experience Years", "") or 0)
            except ValueError:
                min_exp = 0

            roles.append({
                "job_title": row.get("Job Title", "").strip(),
                "category": row.get("Category", "").strip(),
                "required_skills": required_skills,
                "preferred_skills": [],       # left for a recruiter/LLM to fill in later
                "education": education_options,
                "minimum_experience": min_exp,
                "maximum_experience": None,
                "salary_range": row.get("Salary Range", "").strip(),
                "minimum_required_skills": max(1, round(len(required_skills) * 0.6)),
                "weights": DEFAULT_WEIGHTS,
            })
    return roles
