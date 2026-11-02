"""
Phase 4: Rule Engine.

This is the piece from the plan that keeps the system honest: hard
eligibility requirements and the match score are computed
DETERMINISTICALLY here, from structured data. The LLM (added in
Phase 5/6) only ever explains a result the rule engine already
produced — it never decides PASS/FAIL or invents a score.

Hard rules (reject regardless of everything else):
  - fewer than `minimum_required_skills` required skills matched
  - no education match (when the job specifies accepted education)
  - experience below minimum (or above maximum, if set)

Soft rules (affect score, never cause rejection):
  - preferred skills coverage

Scoring is a weighted sum over whichever dimensions actually have
data, renormalized to sum to 1 — e.g. this dataset has no
projects/certifications fields, so those weights are honestly dropped
and redistributed rather than faked with a placeholder score.
"""

from dataclasses import dataclass, field


def _normalize(s: str) -> str:
    return s.strip().lower()


def _skills_match(candidate_skills: list[str], target_skills: list[str]) -> tuple[list[str], list[str]]:
    cand_norm = {_normalize(s) for s in candidate_skills}
    matched, missing = [], []
    for skill in target_skills:
        if _normalize(skill) in cand_norm:
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


def _education_match(candidate_education: list[str], accepted: list[str]) -> bool:
    if not accepted:
        return True  # job doesn't specify education requirements
    if not candidate_education:
        return False
    cand_norm = [_normalize(e) for e in candidate_education]
    for req in accepted:
        req_n = _normalize(req)
        for cand in cand_norm:
            if req_n in cand or cand in req_n:
                return True
    return False


@dataclass
class ScreeningResult:
    candidate_id: str
    status: str                       # "QUALIFIED" | "REJECTED"
    final_score: float                # 0-100
    required_skills_matched: list[str] = field(default_factory=list)
    required_skills_missing: list[str] = field(default_factory=list)
    preferred_skills_matched: list[str] = field(default_factory=list)
    preferred_skills_missing: list[str] = field(default_factory=list)
    education_matched: bool = False
    experience_years: float | None = None
    experience_ok: bool = False
    dimension_scores: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self):
        return self.__dict__


def evaluate_candidate(job_rule: dict, candidate: dict) -> ScreeningResult:
    """
    job_rule: one entry from out/job_rules.json
    candidate: {"candidate_id", "skills", "education", "experience_years",
                "projects" (optional), "certifications" (optional)}
    """
    reasons = []

    required_matched, required_missing = _skills_match(
        candidate.get("skills", []), job_rule.get("required_skills", [])
    )
    preferred_matched, preferred_missing = _skills_match(
        candidate.get("skills", []), job_rule.get("preferred_skills", [])
    )
    education_ok = _education_match(candidate.get("education", []), job_rule.get("education", []))

    exp_years = candidate.get("experience_years")
    min_exp = job_rule.get("minimum_experience") or 0
    max_exp = job_rule.get("maximum_experience")
    experience_ok = exp_years is not None and exp_years >= min_exp
    if experience_ok and max_exp is not None:
        experience_ok = exp_years <= max_exp

    # ---- Hard rules ----
    min_required = job_rule.get("minimum_required_skills", len(job_rule.get("required_skills", [])))
    if len(required_matched) < min_required:
        reasons.append(
            f"Only {len(required_matched)}/{len(job_rule.get('required_skills', []))} "
            f"required skills matched (needs {min_required}). Missing: {', '.join(required_missing) or 'none'}"
        )
    if not education_ok:
        reasons.append(
            f"Education requirement not met. Needs one of: {', '.join(job_rule.get('education', [])) or 'N/A'}"
        )
    if not experience_ok:
        exp_str = f"{exp_years} years" if exp_years is not None else "unknown"
        bound = f"{min_exp}-{max_exp}" if max_exp is not None else f">= {min_exp}"
        reasons.append(f"Experience requirement not met ({exp_str}, needs {bound} years)")

    status = "REJECTED" if reasons else "QUALIFIED"

    # ---- Weighted scoring (only over dimensions with real data) ----
    weights = dict(job_rule.get("weights", {}))
    dimension_scores = {}

    total_required = len(job_rule.get("required_skills", []))
    dimension_scores["required_skills"] = (
        len(required_matched) / total_required if total_required else 1.0
    )

    total_preferred = len(job_rule.get("preferred_skills", []))
    if total_preferred:
        dimension_scores["preferred_skills"] = len(preferred_matched) / total_preferred
    else:
        weights.pop("preferred_skills", None)  # nothing to score, drop from weighting

    dimension_scores["experience"] = 1.0 if experience_ok else min(1.0, max(0.0, (exp_years or 0) / max(min_exp, 1)))
    dimension_scores["education"] = 1.0 if education_ok else 0.0

    # This dataset has no projects/certifications fields — honestly
    # drop those weight dimensions rather than faking a score.
    if not candidate.get("projects"):
        weights.pop("projects", None)
    if not candidate.get("certifications"):
        weights.pop("certifications", None)

    weight_sum = sum(weights.values()) or 1.0
    weights = {k: v / weight_sum for k, v in weights.items()}  # renormalize to sum to 1

    final_score = sum(
        dimension_scores.get(dim, 0.0) * w for dim, w in weights.items()
    ) * 100

    return ScreeningResult(
        candidate_id=candidate["candidate_id"],
        status=status,
        final_score=round(final_score, 1),
        required_skills_matched=required_matched,
        required_skills_missing=required_missing,
        preferred_skills_matched=preferred_matched,
        preferred_skills_missing=preferred_missing,
        education_matched=education_ok,
        experience_years=exp_years,
        experience_ok=experience_ok,
        dimension_scores={k: round(v, 3) for k, v in dimension_scores.items()},
        reasons=reasons,
    )
