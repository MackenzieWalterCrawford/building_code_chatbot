"""
Phase 1: PDF ingestion and section-aware chunking for NYC Building Code.

Parsing strategy:
  - Use pymupdf to extract text block by block, preserving reading order.
  - Detect section boundaries with NYC Building Code numbering conventions:
      CHAPTER N  /  SECTION NNNN  /  NNNN.N  /  NNNN.N.N  /  NNNN.N.N.N
  - Each numbered section becomes one chunk.  Very short sections (<MIN_TOKENS)
    are merged into their parent; oversized sections (>MAX_TOKENS) are split on
    paragraph boundaries.
  - Metadata attached to every chunk: code_name, chapter, section_number,
    section_title, edition, source_file, page_start, page_end, token_count.
  - Output: ./data/processed/<stem>.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EDITION = "2022"
CODE_NAME = "NYC Building Code"

TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "768"))
MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "1024"))
MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "100"))

_ENC = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Regex patterns for NYC Building Code structure
# ---------------------------------------------------------------------------

# Chapter header: "CHAPTER 16" or just the chapter line in the TOC/header
_RE_CHAPTER = re.compile(
    r"^CHAPTER\s+(\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Numbered section: 1601.1, 1601.1.1, 1601.1.1.1
# Followed by at least one word character (the title)
_RE_SECTION = re.compile(
    r"^(\d{4}(?:\.\d+){1,3})\s{1,6}([A-Z][^\n]{2,80})",
    re.MULTILINE,
)



# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    code_name: str
    edition: str
    source_file: str
    chapter: str          # e.g. "16"
    chapter_title: str
    section_number: str   # e.g. "1607.1"
    section_title: str
    text: str
    page_start: int
    page_end: int
    token_count: int
    parent_section: str = ""  # e.g. "1607" for subsection 1607.1.1
    split_index: int = 0      # >0 when an oversized section was split


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: Path) -> list[dict]:
    """Return list of {page_num, text} for every page in the PDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        # sort=True preserves reading order
        text = page.get_text("text", sort=True)
        pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages


def _clean_line(line: str) -> str:
    """Remove watermark artifacts and normalize whitespace."""
    # NYC DOB watermark lines are all-caps short lines — skip them
    stripped = line.strip()
    if not stripped:
        return ""
    # Skip pure page-number lines
    if re.match(r"^\d{1,4}$", stripped):
        return ""
    # Skip header/footer patterns like "2022 NEW YORK CITY BUILDING CODE"
    if re.match(r"^20\d{2}\s+NEW YORK CITY", stripped, re.IGNORECASE):
        return ""
    return stripped


def pages_to_text(pages: list[dict]) -> list[tuple[int, str]]:
    """Return [(page_num, cleaned_line), ...] preserving page attribution.

    Blank lines are kept as empty-string entries (rather than dropped) so
    the `\\n\\n` paragraph boundaries in the source PDF survive into the
    section body text the chunker later splits on.
    """
    result = []
    for p in pages:
        for line in p["text"].splitlines():
            if not line.strip():
                result.append((p["page_num"], ""))
                continue
            cleaned = _clean_line(line)
            if cleaned:
                result.append((p["page_num"], cleaned))
    return result


# ---------------------------------------------------------------------------
# Section boundary detection
# ---------------------------------------------------------------------------

def detect_chapter(lines: list[tuple[int, str]]) -> tuple[str, str]:
    """
    Heuristically determine chapter number and title from the first ~50 lines.
    Returns (chapter_number, chapter_title).
    """
    chapter_num = ""
    chapter_title = ""
    for _, line in lines[:80]:
        m = _RE_CHAPTER.match(line)
        if m:
            chapter_num = m.group(1)
            continue
        # Line immediately after "CHAPTER N" that looks like a title
        if chapter_num and not chapter_title and len(line) > 4:
            # Skip lines that are section numbers
            if not re.match(r"^\d{4}", line):
                chapter_title = line.title()
                break
    return chapter_num, chapter_title


@dataclass
class _SectionSpan:
    section_number: str
    section_title: str
    page_start: int
    page_end: int
    lines: list[str] = field(default_factory=list)


def extract_sections(
    lines: list[tuple[int, str]],
) -> list[_SectionSpan]:
    """
    Walk line-by-line; start a new span whenever we hit a section heading.
    """
    spans: list[_SectionSpan] = []
    current: Optional[_SectionSpan] = None

    for page_num, line in lines:
        m = _RE_SECTION.match(line)
        if m:
            sec_num = m.group(1)
            sec_title = m.group(2).strip()
            if current is not None:
                current.page_end = page_num
                spans.append(current)
            current = _SectionSpan(
                section_number=sec_num,
                section_title=sec_title,
                page_start=page_num,
                page_end=page_num,
                lines=[line],
            )
        else:
            if current is not None:
                current.lines.append(line)
                current.page_end = page_num

    if current is not None:
        spans.append(current)

    return spans


# ---------------------------------------------------------------------------
# Chunking: merge short / split long spans
# ---------------------------------------------------------------------------

_RE_PARAGRAPH_SEP = re.compile(r"\n{2,}")
_RE_SENTENCE_SEP = re.compile(r"(?<=[.;:])\s+")


def _hard_split_tokens(text: str, max_tokens: int) -> list[str]:
    """Guaranteed fallback: slice the raw token stream into fixed windows.

    Used when a piece has no paragraph or sentence boundaries at all
    (e.g. a code table rendered as one dense block) — the only way left
    to actually stay under max_tokens.
    """
    ids = _ENC.encode(text)
    if not ids:
        return [text]
    return [_ENC.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)]


def _bin_pack(
    pieces: list[str], max_tokens: int, joiner: str, next_level
) -> list[str]:
    """Greedily pack pieces into <=max_tokens groups, recursing into
    next_level on any single piece that's still oversized on its own."""
    chunks: list[str] = []
    current: list[str] = []
    current_toks = 0

    for piece in pieces:
        piece_toks = token_count(piece)
        if piece_toks > max_tokens:
            if current:
                chunks.append(joiner.join(current))
                current, current_toks = [], 0
            chunks.extend(next_level(piece, max_tokens))
            continue
        if current_toks + piece_toks > max_tokens and current:
            chunks.append(joiner.join(current))
            current, current_toks = [piece], piece_toks
        else:
            current.append(piece)
            current_toks += piece_toks

    if current:
        chunks.append(joiner.join(current))

    return chunks


def _split_on_sentences(text: str, max_tokens: int) -> list[str]:
    """Split text at sentence-ish boundaries; hard-split if that's not
    enough (e.g. text with no terminal punctuation at all)."""
    if token_count(text) <= max_tokens:
        return [text]
    sentences = [s for s in _RE_SENTENCE_SEP.split(text) if s.strip()]
    if len(sentences) <= 1:
        return _hard_split_tokens(text, max_tokens)
    return _bin_pack(sentences, max_tokens, " ", _split_on_sentences)


def _split_on_paragraphs(text: str, max_tokens: int) -> list[str]:
    """Split text to keep every returned piece under max_tokens.

    Recurses through progressively finer boundaries — paragraphs, then
    sentences, then a hard token-count slice — so a section with no
    paragraph breaks, or even no sentence breaks (dense tables), still
    ends up under max_tokens instead of passing through unsplit.
    """
    if token_count(text) <= max_tokens:
        return [text]
    paragraphs = [p for p in _RE_PARAGRAPH_SEP.split(text) if p.strip()]
    if len(paragraphs) <= 1:
        return _split_on_sentences(text, max_tokens)
    return _bin_pack(paragraphs, max_tokens, "\n\n", _split_on_sentences)


def spans_to_chunks(
    spans: list[_SectionSpan],
    chapter: str,
    chapter_title: str,
    source_file: str,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[Chunk]:
    """Convert raw spans → final chunks with merge/split applied."""
    chunks: list[Chunk] = []
    pending_text = ""
    pending_span: Optional[_SectionSpan] = None

    def _make_chunk(span: _SectionSpan, text: str, split_idx: int = 0) -> Chunk:
        toks = token_count(text)
        # Derive parent section: "1607.1.1" → "1607.1"
        parts = span.section_number.split(".")
        parent = ".".join(parts[:-1]) if len(parts) > 2 else ""
        return Chunk(
            chunk_id=f"{source_file}__{span.section_number}{'_' + str(split_idx) if split_idx else ''}",
            code_name=CODE_NAME,
            edition=EDITION,
            source_file=source_file,
            chapter=chapter,
            chapter_title=chapter_title,
            section_number=span.section_number,
            section_title=span.section_title,
            text=text,
            page_start=span.page_start,
            page_end=span.page_end,
            token_count=toks,
            parent_section=parent,
            split_index=split_idx,
        )

    for span in spans:
        body = "\n".join(span.lines)
        toks = token_count(body)

        # Merge very short spans into the previous pending span
        if toks < min_tokens:
            if pending_span is None:
                pending_span = span
                pending_text = body
            else:
                pending_text += "\n\n" + body
                pending_span.page_end = span.page_end
            continue

        # Flush any pending short span first
        if pending_span is not None:
            merged_text = pending_text
            merged_toks = token_count(merged_text)
            if merged_toks > max_tokens:
                for idx, part in enumerate(_split_on_paragraphs(merged_text, max_tokens)):
                    chunks.append(_make_chunk(pending_span, part, split_idx=idx + 1))
            else:
                chunks.append(_make_chunk(pending_span, merged_text))
            pending_span = None
            pending_text = ""

        # Handle the current (non-short) span
        if toks > max_tokens:
            for idx, part in enumerate(_split_on_paragraphs(body, max_tokens)):
                chunks.append(_make_chunk(span, part, split_idx=idx + 1))
        else:
            chunks.append(_make_chunk(span, body))

    # Flush remaining pending
    if pending_span is not None:
        merged_toks = token_count(pending_text)
        if merged_toks > max_tokens:
            for idx, part in enumerate(_split_on_paragraphs(pending_text, max_tokens)):
                chunks.append(_make_chunk(pending_span, part, split_idx=idx + 1))
        else:
            chunks.append(_make_chunk(pending_span, pending_text))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: Path, output_dir: Path) -> list[Chunk]:
    """
    Parse one Building Code chapter PDF → list of Chunk objects.
    Also writes <output_dir>/<stem>.json for inspection.
    """
    print(f"Ingesting: {pdf_path.name}")
    pages = extract_pages(pdf_path)
    lines = pages_to_text(pages)

    chapter_num, chapter_title = detect_chapter(lines)
    if not chapter_num:
        print(f"  WARNING: could not detect chapter number in {pdf_path.name}")
        chapter_num = pdf_path.stem.split("Chapter")[-1][:2].lstrip("0") or "?"

    print(f"  Chapter {chapter_num}: {chapter_title}")
    print(f"  Pages: {len(pages)}, Lines: {len(lines)}")

    spans = extract_sections(lines)
    print(f"  Raw section spans: {len(spans)}")

    chunks = spans_to_chunks(
        spans,
        chapter=chapter_num,
        chapter_title=chapter_title,
        source_file=pdf_path.stem,
    )
    print(f"  Final chunks: {len(chunks)}")

    if chunks:
        toks = [c.token_count for c in chunks]
        print(f"  Token range: {min(toks)}–{max(toks)}, avg {sum(toks)//len(toks)}")

    # Write intermediate JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)
    print(f"  Written: {out_path}")

    return chunks


def ingest_directory(
    raw_dir: Path,
    processed_dir: Path,
    pattern: str = "2022BC_Chapter*.pdf",
) -> list[Chunk]:
    """Ingest all matching PDFs in raw_dir."""
    pdf_files = sorted(raw_dir.glob(pattern))
    if not pdf_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {raw_dir}")

    all_chunks: list[Chunk] = []
    for pdf in pdf_files:
        chunks = ingest_pdf(pdf, processed_dir)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all files: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest NYC Building Code PDFs")
    parser.add_argument(
        "--file",
        help="Single PDF file to ingest (default: all 2022BC_Chapter*.pdf in data/raw)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing source PDFs",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Directory for processed JSON output",
    )
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)

    if args.file:
        ingest_pdf(Path(args.file), out)
    else:
        ingest_directory(raw, out)
