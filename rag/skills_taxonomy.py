"""
Loads the skill taxonomy (skills_database.json / skills_list.csv) and
provides a format-agnostic skill matcher.

Why this matters for "works no matter the resume format": a real PDF
resume will almost never say "Skills:" followed by a clean comma list.
Skills are often scattered inside project bullets, summaries, and
experience paragraphs. So instead of relying on a "Skills" section
existing at all, we scan the FULL raw text against the known taxonomy
using word-boundary matching. This is source-agnostic by design.
"""

import csv
import json
import re
from pathlib import Path


class SkillsTaxonomy:
    def __init__(self, skills_list_csv: str, skills_database_json: str):
        self.skill_to_category: dict[str, str] = {}
        self._load_csv(skills_list_csv)
        self._load_json(skills_database_json)

        # Sort longest-first so "Machine Learning" matches before "Learning"
        self.all_skills = sorted(self.skill_to_category.keys(), key=len, reverse=True)

        # Precompile one regex per skill using word boundaries. Skills with
        # punctuation (C++, C#, CI/CD) need boundaries relaxed since \b
        # doesn't work well around non-word characters.
        self._patterns = []
        for skill in self.all_skills:
            escaped = re.escape(skill)
            if re.match(r"^\w", skill) and re.search(r"\w$", skill):
                pattern = rf"(?<!\w){escaped}(?!\w)"
            else:
                pattern = rf"(?<!\S){escaped}(?!\S)"
            self._patterns.append((skill, re.compile(pattern, re.IGNORECASE)))

    def _load_csv(self, path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("Skill Name", "").strip()
                cat = row.get("Category", "").strip()
                if name:
                    self.skill_to_category[name] = cat

    def _load_json(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for category, skills in data.items():
            for s in skills:
                s = s.strip()
                if s and s not in self.skill_to_category:
                    self.skill_to_category[s] = category

    def extract_skills(self, text: str) -> list[str]:
        """Scan free text and return canonical skill names found in it."""
        if not text:
            return []
        found = []
        for skill, pattern in self._patterns:
            if pattern.search(text):
                found.append(skill)
        return found

    def category_of(self, skill: str) -> str:
        return self.skill_to_category.get(skill, "Other")
