"""
Phase 4: Screening CLI — "Recruiter Search" mode.

Rebuilds one profile per candidate from out/chunks.jsonl (skills/
education/experience are duplicated across a candidate's chunks by
design — grouping just dedupes them), evaluates every candidate
against a chosen job role's rules, and prints a ranked, explainable
leaderboard: who's QUALIFIED vs REJECTED and exactly why.

Usage:
    python run_screening.py --chunks out/chunks.jsonl --job-rules out/job_rules.json \
        --job-title "Software Engineer" --top-n 10
"""

import argparse
import json
from collections import defaultdict

from rag.rule_engine import evaluate_candidate


def load_candidate_profiles(chunks_path: str) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            meta = chunk["metadata"]
            cid = meta["candidate_id"]
            if cid not in profiles:
                profiles[cid] = {
                    "candidate_id": cid,
                    "skills": meta.get("skills", []),
                    "education": meta.get("education", []),
                    "experience_years": meta.get("experience_years"),
                    "job_role": meta.get("job_role"),
                    "category": meta.get("category"),
                }
    return profiles


def load_job_rule(job_rules_path: str, job_title: str) -> dict:
    with open(job_rules_path, encoding="utf-8") as f:
        rules = json.load(f)
    for r in rules:
        if r["job_title"].strip().lower() == job_title.strip().lower():
            return r
    available = ", ".join(sorted({r["job_title"] for r in rules})[:15])
    raise ValueError(f'No job role matching "{job_title}". Examples available: {available} ...')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--job-rules", required=True)
    ap.add_argument("--job-title", required=True)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--show-rejected", type=int, default=3, help="also show N example rejections")
    args = ap.parse_args()

    profiles = load_candidate_profiles(args.chunks)
    job_rule = load_job_rule(args.job_rules, args.job_title)

    print(f"Job: {job_rule['job_title']}  ({job_rule['category']})")
    print(f"Required: {', '.join(job_rule['required_skills'])}")
    print(f"Education: {' | '.join(job_rule['education']) or 'any'}")
    print(f"Min experience: {job_rule['minimum_experience']} years")
    print(f"Screening {len(profiles)} candidates...\n")

    results = [evaluate_candidate(job_rule, c) for c in profiles.values()]
    qualified = sorted(
        [r for r in results if r.status == "QUALIFIED"],
        key=lambda r: r.final_score, reverse=True,
    )
    rejected = sorted(
        [r for r in results if r.status == "REJECTED"],
        key=lambda r: r.final_score, reverse=True,
    )

    print(f"=== QUALIFIED: {len(qualified)} / {len(results)} ===")
    for r in qualified[: args.top_n]:
        print(f"\n{r.candidate_id}  —  {r.final_score}%")
        print(f"  Required skills: {len(r.required_skills_matched)}/{len(r.required_skills_matched) + len(r.required_skills_missing)} "
              f"matched -> {', '.join(r.required_skills_matched)}")
        if r.required_skills_missing:
            print(f"  Missing (required): {', '.join(r.required_skills_missing)}")
        if r.preferred_skills_matched or r.preferred_skills_missing:
            print(f"  Preferred matched: {', '.join(r.preferred_skills_matched) or 'none'}")
        print(f"  Education OK: {r.education_matched}  |  Experience: {r.experience_years} yrs (OK: {r.experience_ok})")

    if args.show_rejected and rejected:
        print(f"\n=== Example REJECTED (showing {min(args.show_rejected, len(rejected))} of {len(rejected)}) ===")
        for r in rejected[: args.show_rejected]:
            print(f"\n{r.candidate_id}  —  {r.final_score}%  (REJECTED)")
            for reason in r.reasons:
                print(f"  ✗ {reason}")


if __name__ == "__main__":
    main()
