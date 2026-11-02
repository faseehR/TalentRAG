"""
Parses an arbitrary, previously-unseen resume PDF into the same
ResumeRecord schema used for the Kaggle CSV data — no matter how that
specific resume is laid out.

The key design decision: we do NOT assume the resume has clean
"Education:" / "Experience:" / "Skills:" headers like the training
data does. Real resumes vary wildly (two-column layouts, no headers
at all, skills buried inside bullet points, non-standard section
names like "Core Competencies" instead of "Skills"). So this parser
uses two independent, redundant strategies and merges them:

  1. HEADER-BASED SECTIONING (best case): look for lines that match
     common section header vocabulary and split the document there.
     Works well on standard single-column resumes.

  2. FULL-TEXT FALLBACK EXTRACTION (works on anything): regardless of
     whether headers were found, we ALSO run skill-taxonomy matching,
     education-degree regex, and experience-years regex over the
     ENTIRE raw text. This guarantees skills/education/experience are
     still captured even when section detection fails completely
     (e.g. a PDF with no headers, weird formatting, or a resume
     template the header list has never seen).

If header-based sectioning found nothing usable, sections falls back
to a single "summary" section containing the full text, so the
resume is still chunked and searchable rather than silently dropped.
"""

import re
from pathlib import Path

from .schema import ResumeRecord
from .skills_taxonomy import SkillsTaxonomy

# Section header vocabulary: canonical_name -> regex alternatives seen
# in the wild across different resume templates/styles.
HEADER_PATTERNS = {
    "summary": r"(summary|objective|profile|about\s*me)",
    "education": r"(education|academic\s*background|qualifications)",
    "experience": r"(experience|employment(?:\s*history)?|work\s*history|professional\s*experience)",
    "skills": r"(skills|technical\s*skills|core\s*competencies|competencies)",
    "projects": r"(projects|personal\s*projects|key\s*projects)",
    "certifications": r"(certifications?|licenses?|credentials)",
}
_HEADER_LINE_RE = re.compile(
    r"^\s*(" + "|".join(HEADER_PATTERNS.values()) + r")\s*:?\s*$",
    re.IGNORECASE,
)

_DEGREE_RE = re.compile(
    r"(Bachelor'?s|Master'?s|B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|Ph\.?D\.?|"
    r"Associate'?s?|Diploma|MBA)\s*(?:of|in|degree in)?\s*"
    r"([A-Z][A-Za-z&,/ ]{2,40})?",
)
_EXPERIENCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b", re.IGNORECASE
)


def extract_text(pdf_path: str) -> str:
    """Try pdfplumber first (better layout handling), fall back to pypdf."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _split_into_sections(raw_text: str) -> dict[str, str]:
    lines = raw_text.splitlines()
    sections: dict[str, list[str]] = {}
    current = None
    for line in lines:
        m = _HEADER_LINE_RE.match(line.strip())
        if m:
            header_text = m.group(1).lower()
            for canonical, pattern in HEADER_PATTERNS.items():
                if re.match(pattern, header_text, re.IGNORECASE):
                    current = canonical
                    sections.setdefault(current, [])
                    break
            continue
        if current:
            sections[current].append(line)
        # lines before any header found are ignored here; full raw_text
        # is preserved separately as the fallback.
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def _guess_name(raw_text: str) -> str | None:
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # crude heuristic: first non-empty line, short, no digits/@ symbol
        if 2 <= len(line.split()) <= 4 and not any(c.isdigit() for c in line) and "@" not in line:
            return line
        return None
    return None


def _extract_education(raw_text: str) -> list[str]:
    found = []
    for m in _DEGREE_RE.finditer(raw_text):
        degree = m.group(1)
        field = (m.group(2) or "").strip()
        found.append(f"{degree} in {field}".strip() if field else degree)
    # dedupe, preserve order
    seen = set()
    out = []
    for e in found:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _extract_experience_years(raw_text: str) -> float | None:
    matches = [float(m.group(1)) for m in _EXPERIENCE_RE.finditer(raw_text)]
    return max(matches) if matches else None


def parse_resume_text(raw_text: str, taxonomy: SkillsTaxonomy, candidate_id: str) -> ResumeRecord:
    """
    Same format-agnostic parsing as parse_resume_pdf(), but takes raw
    text directly. Used both internally (after PDF text extraction)
    and directly by the evaluation harness for text-based ground-truth
    resumes (test_resumes.json) that never went through a PDF at all.
    """
    # Strategy 1: header-based sectioning (works when the resume follows
    # a conventional layout with real section headers).
    sections = _split_into_sections(raw_text)

    # Strategy 2 (always runs): full-text fallback extraction, so
    # skills/education/experience are captured even if strategy 1
    # found no usable sections at all.
    skills = taxonomy.extract_skills(raw_text)
    education = _extract_education(sections.get("education", "") or raw_text)
    experience_years = _extract_experience_years(sections.get("experience", "") or raw_text)

    if not sections:
        # No headers detected at all -> keep the whole document
        # searchable as a single "summary" section rather than
        # dropping the resume.
        sections = {"summary": raw_text.strip()}

    # Make sure skills are represented as a section chunk too, even if
    # no "Skills" header existed (skills lived in bullets/paragraphs).
    if "skills" not in sections and skills:
        sections["skills"] = f"Skills: {', '.join(skills)}"

    return ResumeRecord(
        candidate_id=candidate_id,
        source="text",
        raw_text=raw_text,
        name=_guess_name(raw_text),
        education=education,
        experience_years=experience_years,
        skills=skills,
        job_role=None,
        category=None,
        sections=sections,
    )


def parse_resume_pdf(pdf_path: str, taxonomy: SkillsTaxonomy, candidate_id: str | None = None) -> ResumeRecord:
    raw_text = extract_text(pdf_path)
    candidate_id = candidate_id or Path(pdf_path).stem
    record = parse_resume_text(raw_text, taxonomy, candidate_id)
    record.source = "pdf"
    return record
