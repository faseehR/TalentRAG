"""
Diagnostic from Phase 6 evaluation: the provided skills_database.json
only covers 115 skills, but training_data.csv + job_roles.csv actually
use 793 unique skill strings between them — an 85% coverage gap. This
is why free-text skill extraction (used for PDFs and the labeled
test_resumes.json) was missing skills like "Safety", "Troubleshooting",
and "Electrical Work" even when they appeared verbatim in the text.

This script closes that gap by mining the corpus's own vocabulary and
merging it into the taxonomy, rather than silently living with a
taxonomy that doesn't match the data it's meant to describe.

Usage:
    python build_extended_taxonomy.py \
        --training-csv training_data.csv --job-roles-csv job_roles.csv \
        --skills-list skills_list.csv --skills-db skills_database.json \
        --out out/skills_database_extended.json
"""

import argparse
import csv
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--job-roles-csv", required=True)
    ap.add_argument("--skills-list", required=True)
    ap.add_argument("--skills-db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    db = json.load(open(args.skills_db, encoding="utf-8"))
    existing = set()
    for cat, skills in db.items():
        existing.update(skills)
    with open(args.skills_list, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            existing.add(row["Skill Name"].strip())

    corpus_skills = set()
    with open(args.training_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            corpus_skills.update(s.strip() for s in row["Skills"].split("|") if s.strip())
    with open(args.job_roles_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            corpus_skills.update(s.strip() for s in row["Required Skills"].split("|") if s.strip())

    missing = sorted(corpus_skills - existing)
    db["From Dataset (uncategorized)"] = missing
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"Provided taxonomy: {len(existing)} skills")
    print(f"Dataset vocabulary: {len(corpus_skills)} skills")
    print(f"Added {len(missing)} previously-missing skills -> {args.out}")


if __name__ == "__main__":
    main()
