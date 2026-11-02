"""
Shared schema for TalentRAG ingestion.

Every resume — whether it came from the Kaggle training CSV or an
arbitrary uploaded PDF — is normalized into this ONE structured
record before chunking. This is what makes the rest of the pipeline
(chunking, embeddings, Pinecone, rule engine) format-agnostic: it
never has to know or care whether the resume was structured CSV data
or messy real-world PDF text.
"""

from dataclasses import dataclass, field
from typing import Optional


# Canonical section names used everywhere downstream (chunker, Pinecone
# metadata, rule engine). PDF parsing maps whatever headers it finds
# onto this fixed set.
SECTION_NAMES = [
    "summary",
    "education",
    "experience",
    "skills",
    "projects",
    "certifications",
]


@dataclass
class ResumeRecord:
    candidate_id: str
    source: str                      # "csv" | "pdf"
    raw_text: str
    name: Optional[str] = None
    education: list[str] = field(default_factory=list)
    experience_years: Optional[float] = None
    skills: list[str] = field(default_factory=list)
    job_role: Optional[str] = None   # only present for labeled CSV rows
    category: Optional[str] = None
    sections: dict[str, str] = field(default_factory=dict)  # section -> text

    def to_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "name": self.name,
            "education": self.education,
            "experience_years": self.experience_years,
            "skills": self.skills,
            "job_role": self.job_role,
            "category": self.category,
            "sections": self.sections,
        }
