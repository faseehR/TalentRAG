"""
Generates the human-readable "why did this candidate match/fail"
explanation — the last step in the pipeline, and the one place an LLM
touches the decision at all. Critically, the LLM never decides
PASS/FAIL or the score (rag/rule_engine.py already did that
deterministically); it only explains a decision it's handed, and is
restricted to the evidence text it's given so it can't invent resume
content that isn't actually there.

  - AnthropicExplainer: production. Uses the Claude API
    (`pip install anthropic`, ANTHROPIC_API_KEY). The prompt
    explicitly instructs the model to use ONLY the supplied evidence
    and to say "No evidence found" rather than infer.
  - LocalTemplateExplainer: dependency-light stand-in — builds the
    same style of grounded explanation from the rule engine's
    structured output with plain string templates, no LLM call at
    all. Same interface, so this is what runs in local dev/CI and
    swaps for the real backend in production with one flag.
"""

from abc import ABC, abstractmethod


SYSTEM_PROMPT = """You are writing a short, evidence-grounded explanation of a resume \
screening result for a recruiter. You are given the job's requirements, a deterministic \
rule-engine result (already decided — you do not change it), and short resume text \
snippets as evidence for each skill.

Rules you must follow:
- Use ONLY the supplied evidence text. Never infer or assume experience, skills, or \
  qualifications that are not present in the evidence.
- If a required or preferred skill has no evidence snippet, say so explicitly \
  ("No supporting evidence found in resume") rather than guessing.
- Do not change or second-guess the QUALIFIED/REJECTED decision or the score — only \
  explain it.
- Keep it to 3-5 sentences, professional and factual, no marketing language.
"""


def build_user_prompt(job_rule: dict, screening_result: dict, evidence: dict) -> str:
    return (
        f"Job: {job_rule['job_title']}\n"
        f"Required skills: {', '.join(job_rule['required_skills'])}\n"
        f"Preferred skills: {', '.join(job_rule.get('preferred_skills', [])) or 'none'}\n\n"
        f"Rule engine result: {screening_result['status']}, score {screening_result['final_score']}%\n"
        f"Matched required: {', '.join(screening_result['required_skills_matched'])}\n"
        f"Missing required: {', '.join(screening_result['required_skills_missing'])}\n\n"
        f"Evidence snippets:\n"
        + "\n".join(f"- {skill}: \"{text}\"" for skill, text in evidence.items())
    )


class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, job_rule: dict, screening_result: dict, evidence: dict[str, str]) -> str:
        ...


class AnthropicExplainer(BaseExplainer):
    """Production backend. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import os
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def explain(self, job_rule, screening_result, evidence):
        user_prompt = build_user_prompt(job_rule, screening_result, evidence)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class GeminiExplainer(BaseExplainer):
    """Production backend using Google Gemini. Requires `pip install google-genai`
    and GEMINI_API_KEY (get one free at aistudio.google.com)."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        import os
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def explain(self, job_rule, screening_result, evidence):
        from google.genai import types
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=build_user_prompt(job_rule, screening_result, evidence),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, max_output_tokens=400, temperature=0.2,
                ),
            )
            text = (resp.text or "").strip()
            if text:
                return text
        except Exception:
            # Free-tier rate limits (HTTP 429), network blips, or safety blocks must not
            # break a whole screening run — fall back to the deterministic template.
            pass
        return LocalTemplateExplainer().explain(job_rule, screening_result, evidence)


class LocalTemplateExplainer(BaseExplainer):
    """No-API-key stand-in. Deterministic, template-based, still evidence-grounded."""

    def explain(self, job_rule, screening_result, evidence):
        lines = []
        status = screening_result["status"]
        score = screening_result["final_score"]
        lines.append(
            f"Candidate is {status} for {job_rule['job_title']} with a match score of {score}%."
        )

        matched = screening_result["required_skills_matched"]
        if matched:
            parts = []
            for skill in matched:
                snippet = evidence.get(skill)
                if snippet:
                    parts.append(f'{skill} (evidence: "{snippet.strip()[:80]}")')
                else:
                    parts.append(f"{skill} (no direct snippet captured)")
            lines.append("Required skills confirmed: " + "; ".join(parts) + ".")

        missing = screening_result["required_skills_missing"]
        if missing:
            lines.append(
                "Missing required skills with no supporting evidence in the resume: "
                + ", ".join(missing) + "."
            )

        if status == "REJECTED":
            lines.append("Rejection reasons: " + " ".join(screening_result.get("reasons", [])))

        return " ".join(lines)


def get_explainer(backend: str = "local_template", **kwargs) -> BaseExplainer:
    if backend == "anthropic":
        return AnthropicExplainer(**kwargs)
    if backend == "gemini":
        return GeminiExplainer(**kwargs)
    if backend == "local_template":
        return LocalTemplateExplainer(**kwargs)
    raise ValueError(f"Unknown explainer backend: {backend}")
