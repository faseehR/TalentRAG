"""
Section-aware chunking, as specified in the project plan: one chunk
per resume SECTION (education / experience / skills / projects /
certifications / summary), each carrying rich metadata — never one
giant 500-token blob per resume.

This is deliberately the LAST step and takes a normalized
ResumeRecord as input, so it doesn't matter whether that record came
from the clean CSV or a messy real-world PDF — chunking behaves
identically either way. That's the "works no matter the format" piece:
all the format-specific work happens earlier in csv_ingest.py /
pdf_ingest.py, and by the time content reaches this function it's
already uniform.
"""

from .schema import ResumeRecord


def chunk_resume(record: ResumeRecord) -> list[dict]:
    chunks = []
    for section, text in record.sections.items():
        if not text or not text.strip():
            continue
        chunk_id = f"{record.candidate_id}__{section}"
        chunks.append({
            "id": chunk_id,
            "text": text.strip(),
            "metadata": {
                "candidate_id": record.candidate_id,
                "section": section,
                "source": record.source,
                "skills": record.skills,
                "education": record.education,
                "experience_years": record.experience_years,
                "job_role": record.job_role,
                "category": record.category,
            },
        })

    # Always add one skills-only chunk even if there was no natural
    # "skills" section text (e.g. a PDF where skills were scattered
    # across bullets, not listed under a header) — this is what keeps
    # skill-based retrieval reliable regardless of resume format.
    if record.skills and "skills" not in record.sections:
        chunks.append({
            "id": f"{record.candidate_id}__skills_extracted",
            "text": f"Skills: {', '.join(record.skills)}",
            "metadata": {
                "candidate_id": record.candidate_id,
                "section": "skills",
                "source": record.source,
                "skills": record.skills,
                "education": record.education,
                "experience_years": record.experience_years,
                "job_role": record.job_role,
                "category": record.category,
            },
        })

    return chunks
