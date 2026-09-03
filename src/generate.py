"""
Phase 4: Answer generation grounded strictly in retrieved chunks.

Supports OpenAI (default) and Anthropic via LLM_PROVIDER env var.
Returns: answer, cited section numbers, raw chunks, confidence score.

Confidence is estimated from the top-k retrieval scores and how many
unique sections are cited in the answer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv

from retrieve import RetrievedChunk

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """\
You are a precise technical assistant for NYC Building Code questions.
Answer ONLY using the provided code sections below. Do not add information
from general knowledge.

Rules:
1. Cite every claim with the exact section number in the format §NNNN.N.
2. If multiple sections support a point, cite all of them.
3. Use plain language an architect can act on.
4. If the retrieved sections do not contain enough information to answer
   the question, respond with exactly:
   "The retrieved code sections do not contain sufficient information to
   answer this question. Please consult a NY-licensed professional or
   search additional code sections."
5. Never invent rules, numbers, or requirements not present in the text.
"""

CONTEXT_TEMPLATE = """\
--- Retrieved Code Sections ---
{sections}
--- End of Retrieved Sections ---

Question: {question}
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    answer: str
    cited_sections: list[str]
    confidence: float
    retrieved_chunks: list[RetrievedChunk]
    insufficient: bool = False


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _build_context(chunks: list[RetrievedChunk], question: str) -> str:
    sections_text = ""
    for chunk in chunks:
        sections_text += (
            f"\n§{chunk.section_number} — {chunk.section_title}\n"
            f"{chunk.text}\n"
        )
    return CONTEXT_TEMPLATE.format(sections=sections_text, question=question)


def _call_openai(context: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(context: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
        temperature=0,
    )
    return resp.content[0].text


def _llm_call(context: str) -> str:
    if LLM_PROVIDER == "anthropic":
        return _call_anthropic(context)
    return _call_openai(context)


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

_RE_CITE = re.compile(r"§(\d{4}(?:\.\d+){0,3})")


def extract_citations(answer: str) -> list[str]:
    """Pull all §NNNN.N citations from the answer text."""
    return list(dict.fromkeys(_RE_CITE.findall(answer)))  # deduplicated, ordered


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def estimate_confidence(
    chunks: list[RetrievedChunk],
    cited_sections: list[str],
    answer: str,
) -> float:
    """
    Simple heuristic confidence:
    - Base: average of top-3 retrieval scores (normalized 0–1)
    - Bonus: +0.1 if ≥2 unique sections cited
    - Penalty: -0.3 if answer contains "insufficient" keyword
    Clamped to [0.0, 1.0].
    """
    if not chunks:
        return 0.0

    top3_scores = sorted([c.score for c in chunks], reverse=True)[:3]
    # Weaviate hybrid scores are in [0,1] range
    base = sum(top3_scores) / len(top3_scores)

    bonus = 0.1 if len(cited_sections) >= 2 else 0.0
    penalty = -0.3 if "do not contain sufficient" in answer.lower() else 0.0

    return round(min(1.0, max(0.0, base + bonus + penalty)), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    question: str,
    chunks: list[RetrievedChunk],
) -> GenerationResult:
    """
    Generate a grounded answer from retrieved chunks.
    Returns GenerationResult with answer, citations, confidence.
    """
    if not chunks:
        insufficient_msg = (
            "The retrieved code sections do not contain sufficient information "
            "to answer this question. Please consult a NY-licensed professional "
            "or search additional code sections."
        )
        return GenerationResult(
            answer=insufficient_msg,
            cited_sections=[],
            confidence=0.0,
            retrieved_chunks=[],
            insufficient=True,
        )

    context = _build_context(chunks, question)
    answer = _llm_call(context)

    cited = extract_citations(answer)
    insufficient = "do not contain sufficient" in answer.lower()
    confidence = estimate_confidence(chunks, cited, answer)

    return GenerationResult(
        answer=answer,
        cited_sections=cited,
        confidence=confidence,
        retrieved_chunks=chunks,
        insufficient=insufficient,
    )


if __name__ == "__main__":
    import sys
    from retrieve import retrieve

    question = " ".join(sys.argv[1:]) or "What is the minimum floor live load for office occupancies?"
    print(f"Question: {question}\n")
    chunks = retrieve(question)
    result = generate(question, chunks)
    print("Answer:")
    print(result.answer)
    print(f"\nCited: {result.cited_sections}")
    print(f"Confidence: {result.confidence:.3f}")
