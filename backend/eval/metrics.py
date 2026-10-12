"""
Pure, dependency-free metrics for the RAG evaluation harness.

A retrieved chunk "matches" a gold section when:
  - its section_number equals the gold section, or
  - it is a descendant of the gold section (gold 1025.6.1 vs chunk 1025.6.1.2), or
  - the gold section number starts a line inside the chunk text (chunks are
    section-aware but a parent chunk like 507.8 also carries 507.8.1 inline).
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence


def section_matches(gold: str, section_number: str, text: str = "") -> bool:
    if section_number == gold or section_number.startswith(gold + "."):
        return True
    if text and re.search(rf"(?m)^\s*{re.escape(gold)}(?=[\s.])", text):
        return True
    return False


def first_hit_rank(gold_sections: Sequence[str], ranked: Sequence[dict]) -> int | None:
    """1-based rank of the first retrieved chunk matching ANY gold section."""
    for i, chunk in enumerate(ranked, 1):
        if any(section_matches(g, chunk["section_number"], chunk.get("text", "")) for g in gold_sections):
            return i
    return None


def hit_at_k(gold_sections: Sequence[str], ranked: Sequence[dict], k: int) -> float:
    rank = first_hit_rank(gold_sections, ranked[:k])
    return 1.0 if rank is not None else 0.0


def reciprocal_rank(gold_sections: Sequence[str], ranked: Sequence[dict], k: int) -> float:
    rank = first_hit_rank(gold_sections, ranked[:k])
    return 1.0 / rank if rank else 0.0


def recall_at_k(gold_sections: Sequence[str], ranked: Sequence[dict], k: int) -> float:
    """Fraction of gold sections covered by at least one of the top-k chunks."""
    if not gold_sections:
        return 0.0
    top = ranked[:k]
    covered = sum(
        1 for g in gold_sections
        if any(section_matches(g, c["section_number"], c.get("text", "")) for c in top)
    )
    return covered / len(gold_sections)


def citation_scores(
    cited: Iterable[str],
    gold_sections: Sequence[str],
    retrieved: Sequence[dict],
) -> dict:
    """
    Answer-level citation quality.

    recall    — fraction of gold sections the answer cites
    precision — fraction of cited sections that are gold
    grounded  — fraction of cited sections that were actually in the
                retrieved context (a citation outside the context is a
                fabricated or leaked-from-memory citation)
    """
    cited = [c.strip().lstrip("§") for c in cited if c and c.strip()]
    retrieved_sections = {c["section_number"] for c in retrieved}

    def is_gold(c: str) -> bool:
        return any(c == g or c.startswith(g + ".") or g.startswith(c + ".") for g in gold_sections)

    def covered(g: str) -> bool:
        return any(c == g or c.startswith(g + ".") or g.startswith(c + ".") for c in cited)

    return {
        "recall": (sum(covered(g) for g in gold_sections) / len(gold_sections)) if gold_sections else None,
        "precision": (sum(is_gold(c) for c in cited) / len(cited)) if cited else None,
        "grounded": (sum(c in retrieved_sections for c in cited) / len(cited)) if cited else None,
    }


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
