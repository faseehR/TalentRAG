"""
Phase 1 + Phase 2: Ingestion & Chunking pipeline for TalentRAG.

Usage:
    python run_ingestion.py \
        --training-csv data/training_data.csv \
        --job-roles-csv data/job_roles.csv \
        --skills-list data/skills_list.csv \
        --skills-db data/skills_database.json \
        --pdf-dir data/uploaded_resumes/ \
        --out-dir out/

Produces (in --out-dir):
    chunks.jsonl          One JSON object per chunk, ready to embed
                           and upsert into Pinecone (id, text, metadata).
    job_rules.json         Rule-engine definitions derived from job_roles.csv.
    ingestion_report.json  Summary stats for a sanity check / README screenshot.

Any file dropped into --pdf-dir (any resume format/layout) is parsed by
rag/pdf_ingest.py and chunked with the exact same section_chunker used
for the CSV data, so downstream retrieval doesn't care where a resume
came from.
"""

import argparse
import json
from pathlib import Path

from rag.csv_ingest import load_training_csv
from rag.job_rules import load_job_roles
from rag.pdf_ingest import parse_resume_pdf
from rag.section_chunker import chunk_resume
from rag.skills_taxonomy import SkillsTaxonomy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--job-roles-csv", required=True)
    ap.add_argument("--skills-list", required=True)
    ap.add_argument("--skills-db", required=True)
    ap.add_argument("--pdf-dir", default=None, help="Optional folder of new resume PDFs to ingest")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on # of CSV resumes (for a quick test run)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = SkillsTaxonomy(args.skills_list, args.skills_db)

    all_chunks = []
    stats = {"csv_resumes": 0, "pdf_resumes": 0, "total_chunks": 0, "pdf_failures": []}

    # --- Phase 1+2 on the Kaggle training data ---
    records = load_training_csv(args.training_csv)
    if args.limit:
        records = records[: args.limit]
    for rec in records:
        all_chunks.extend(chunk_resume(rec))
    stats["csv_resumes"] = len(records)

    # --- Same pipeline, applied to any newly uploaded PDF resumes ---
    if args.pdf_dir and Path(args.pdf_dir).exists():
        pdf_paths = sorted(Path(args.pdf_dir).glob("*.pdf"))
        for pdf_path in pdf_paths:
            try:
                rec = parse_resume_pdf(str(pdf_path), taxonomy)
                all_chunks.extend(chunk_resume(rec))
                stats["pdf_resumes"] += 1
            except Exception as e:
                stats["pdf_failures"].append({"file": pdf_path.name, "error": str(e)})

    stats["total_chunks"] = len(all_chunks)

    # --- Write chunks (ready for the embedding step) ---
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # --- Phase 1: job roles -> rule engine definitions ---
    job_rules = load_job_roles(args.job_roles_csv)
    with open(out_dir / "job_rules.json", "w", encoding="utf-8") as f:
        json.dump(job_rules, f, indent=2, ensure_ascii=False)

    with open(out_dir / "ingestion_report.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Chunks written: {stats['total_chunks']} -> {out_dir/'chunks.jsonl'}")
    print(f"Job rules written: {len(job_rules)} -> {out_dir/'job_rules.json'}")
    print(f"CSV resumes ingested: {stats['csv_resumes']}")
    print(f"PDF resumes ingested: {stats['pdf_resumes']}  (failures: {len(stats['pdf_failures'])})")


if __name__ == "__main__":
    main()
