"""
Phase 4: Answer generation grounded strictly in retrieved chunks.

Supports OpenAI (default) and Anthropic via LLM_PROVIDER env var.
Returns: answer, cited section numbers, raw chunks, confidence score.

The LLM is forced to respond with structured output (OpenAI Structured
Outputs / Anthropic forced tool use) matching StructuredAnswer, rather than
free-form prose. That removes the need to regex citations or substring-match
an "insufficient" phrase out of the model's own wording — both citations and
the sufficiency flag come back as typed, schema-validated fields regardless
of which provider answered.

Confidence is estimated from the top-k retrieval scores and how many
unique sections are cited in the answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from retrieve import RetrievedChunk

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

INSUFFICIENT_MESSAGE = (
    "The retrieved code sections do not contain sufficient information "
    "to answer this question. Please consult a NY-licensed professional "
    "or search additional code sections."
)

SYSTEM_PROMPT = f"""\
You are a precise technical assistant for NYC Building Code questions.
Answer ONLY using the provided code sections below. Do not add information
from general knowledge.

You must respond by submitting your answer through the provided tool/function.
Populate its fields as follows:
- "answer": the plain-language answer an architect can act on, citing every
  claim inline with the exact section number in the format §NNNN.N. Cite all
  supporting sections when more than one applies.
- "cited_sections": the same section numbers referenced in "answer", as bare
  numbers without the § symbol (e.g. "1607.1").
- "sufficient": true if the retrieved sections contain enough information to
  answer the question, false otherwise.

Rules:
1. Never invent rules, numbers, or requirements not present in the text.
2. If "sufficient" is false, set "answer" to exactly:
   "{INSUFFICIENT_MESSAGE}"
   and leave "cited_sections" empty.
"""

CONTEXT_TEMPLATE = """\
--- Retrieved Code Sections ---
{sections}
--- End of Retrieved Sections ---

Question: {question}
"""


# ---------------------------------------------------------------------------
# Structured LLM output contract
# ---------------------------------------------------------------------------

class StructuredAnswer(BaseModel):
    answer: str = Field(
        description="Plain-language answer, citing every claim inline as §NNNN.N."
    )
    cited_sections: list[str] = Field(
        description="Bare section numbers cited in the answer, e.g. '1607.1' (no § symbol)."
    )
    sufficient: bool = Field(
        description="False if the retrieved sections do not contain enough information to answer the question."
    )


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


def _call_openai(context: str) -> StructuredAnswer:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        response_format=StructuredAnswer,
        temperature=0,
    )
    message = resp.choices[0].message
    if message.parsed is None:
        raise RuntimeError(f"OpenAI declined to produce a structured answer: {message.refusal}")
    return message.parsed


_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the structured answer to the building code question.",
    "input_schema": StructuredAnswer.model_json_schema(),
}


def _call_anthropic(context: str) -> StructuredAnswer:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
        temperature=0,
        tools=[_ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "submit_answer"},
    )
    tool_use = next(block for block in resp.content if block.type == "tool_use")
    return StructuredAnswer.model_validate(tool_use.input)


def _llm_call(context: str) -> StructuredAnswer:
    if LLM_PROVIDER == "anthropic":
        return _call_anthropic(context)
    return _call_openai(context)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def estimate_confidence(
    chunks: list[RetrievedChunk],
    cited_sections: list[str],
    sufficient: bool,
) -> float:
    """
    Simple heuristic confidence:
    - Base: average of top-3 retrieval scores (normalized 0–1)
    - Bonus: +0.1 if ≥2 unique sections cited
    - Penalty: -0.3 if the model reported the sections were insufficient
    Clamped to [0.0, 1.0].
    """
    if not chunks:
        return 0.0

    top3_scores = sorted([c.score for c in chunks], reverse=True)[:3]
    # Weaviate hybrid scores are in [0,1] range
    base = sum(top3_scores) / len(top3_scores)

    bonus = 0.1 if len(cited_sections) >= 2 else 0.0
    penalty = -0.3 if not sufficient else 0.0

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
        return GenerationResult(
            answer=INSUFFICIENT_MESSAGE,
            cited_sections=[],
            confidence=0.0,
            retrieved_chunks=[],
            insufficient=True,
        )

    context = _build_context(chunks, question)
    structured = _llm_call(context)

    confidence = estimate_confidence(chunks, structured.cited_sections, structured.sufficient)

    return GenerationResult(
        answer=structured.answer,
        cited_sections=structured.cited_sections,
        confidence=confidence,
        retrieved_chunks=chunks,
        insufficient=not structured.sufficient,
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
