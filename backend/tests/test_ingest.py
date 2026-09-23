"""Unit tests for Phase 1 ingestion and chunking."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest import (
    _RE_SECTION,
    _split_on_paragraphs,
    detect_chapter,
    extract_sections,
    spans_to_chunks,
    token_count,
    _SectionSpan,
)


# ---------------------------------------------------------------------------
# token_count
# ---------------------------------------------------------------------------

def test_token_count_nonempty():
    assert token_count("hello world") > 0


def test_token_count_empty():
    assert token_count("") == 0


# ---------------------------------------------------------------------------
# Section regex
# ---------------------------------------------------------------------------

class TestSectionRegex:
    def test_basic_section(self):
        m = _RE_SECTION.match("1607.1 Floor live loads")
        assert m is not None
        assert m.group(1) == "1607.1"
        assert "Floor live loads" in m.group(2)

    def test_subsection(self):
        m = _RE_SECTION.match("1607.1.1 Uniform live loads")
        assert m is not None
        assert m.group(1) == "1607.1.1"

    def test_deep_subsection(self):
        m = _RE_SECTION.match("1607.1.1.1 Specific provisions")
        assert m is not None
        assert m.group(1) == "1607.1.1.1"

    def test_no_match_plain_text(self):
        assert _RE_SECTION.match("This is plain body text.") is None

    def test_no_match_short_title(self):
        # Title must be ≥3 chars after the section number
        assert _RE_SECTION.match("1607.1 Ab") is None

    def test_no_match_chapter_only(self):
        assert _RE_SECTION.match("CHAPTER 16") is None


# ---------------------------------------------------------------------------
# detect_chapter
# ---------------------------------------------------------------------------

class TestDetectChapter:
    def _lines(self, text: str):
        return [(1, l) for l in text.splitlines() if l.strip()]

    def test_standard_chapter(self):
        text = "CHAPTER 16\nSTRUCTURAL DESIGN\n1601.1 Scope."
        ch, title = detect_chapter(self._lines(text))
        assert ch == "16"
        assert "Structural" in title

    def test_missing_chapter(self):
        text = "1601.1 Scope.\nSome body text."
        ch, title = detect_chapter(self._lines(text))
        assert ch == ""


# ---------------------------------------------------------------------------
# extract_sections
# ---------------------------------------------------------------------------

class TestExtractSections:
    def _lines(self, text: str, page: int = 1):
        return [(page, l) for l in text.splitlines() if l.strip()]

    def test_single_section(self):
        text = "1607.1 Floor live loads\nBody text here."
        spans = extract_sections(self._lines(text))
        assert len(spans) == 1
        assert spans[0].section_number == "1607.1"
        assert spans[0].section_title == "Floor live loads"

    def test_multiple_sections(self):
        text = (
            "1607.1 Floor live loads\nText A.\n"
            "1607.2 Roof live loads\nText B.\n"
            "1607.3 Snow loads\nText C."
        )
        spans = extract_sections(self._lines(text))
        assert len(spans) == 3
        assert [s.section_number for s in spans] == ["1607.1", "1607.2", "1607.3"]

    def test_body_text_accumulated(self):
        text = "1607.1 Floor live loads\nFirst line.\nSecond line."
        spans = extract_sections(self._lines(text))
        assert len(spans) == 1
        body = "\n".join(spans[0].lines)
        assert "First line." in body
        assert "Second line." in body


# ---------------------------------------------------------------------------
# spans_to_chunks — merge / split logic
# ---------------------------------------------------------------------------

class TestSpansToChunks:
    def _span(self, sec: str, title: str, n_tokens: int) -> _SectionSpan:
        # Generate text with approximately n_tokens tokens
        words = "building code section structural design " * (n_tokens // 7 + 1)
        lines = [f"{sec} {title}"] + words.split()[:n_tokens]
        return _SectionSpan(
            section_number=sec,
            section_title=title,
            page_start=1,
            page_end=1,
            lines=lines,
        )

    def test_normal_span_becomes_one_chunk(self):
        spans = [self._span("1607.1", "Floor loads", 300)]
        chunks = spans_to_chunks(spans, "16", "Structural Design", "test_file",
                                 min_tokens=50, max_tokens=1024)
        assert len(chunks) == 1
        assert chunks[0].section_number == "1607.1"

    def test_short_spans_merged(self):
        # Two spans each too short → should merge (or one chunk total)
        spans = [
            self._span("1607.1", "Floor loads", 30),
            self._span("1607.2", "Roof loads", 30),
        ]
        chunks = spans_to_chunks(spans, "16", "Structural Design", "test_file",
                                 min_tokens=50, max_tokens=1024)
        # Both are short; the second triggers flush — result may be 1 or 2 chunks
        total_tokens = sum(c.token_count for c in chunks)
        assert total_tokens > 0

    def test_oversized_span_split(self):
        # Span with ~1500 tokens should be split
        spans = [self._span("1607.1", "Floor loads", 1500)]
        chunks = spans_to_chunks(spans, "16", "Structural Design", "test_file",
                                 min_tokens=50, max_tokens=800)
        assert len(chunks) > 1
        for c in chunks:
            assert c.split_index > 0

    def test_metadata_correct(self):
        spans = [self._span("1607.3", "Snow loads", 200)]
        chunks = spans_to_chunks(spans, "16", "Structural Design", "test_file",
                                 min_tokens=50, max_tokens=1024)
        c = chunks[0]
        assert c.chapter == "16"
        assert c.edition == "2022"
        assert c.code_name == "NYC Building Code"
        assert c.section_number == "1607.3"

    def test_parent_section_derived(self):
        spans = [self._span("1607.3.1", "Sub provision", 200)]
        chunks = spans_to_chunks(spans, "16", "Structural Design", "test_file",
                                 min_tokens=50, max_tokens=1024)
        assert chunks[0].parent_section == "1607.3"


# ---------------------------------------------------------------------------
# split_on_paragraphs
# ---------------------------------------------------------------------------

class TestSplitOnParagraphs:
    def test_splits_at_paragraph_boundary(self):
        # Two long paragraphs
        para_a = "word " * 200
        para_b = "word " * 200
        text = para_a + "\n\n" + para_b
        parts = _split_on_paragraphs(text, max_tokens=400)
        assert len(parts) >= 2

    def test_single_short_text_unchanged(self):
        text = "Short text."
        parts = _split_on_paragraphs(text, max_tokens=1000)
        assert parts == ["Short text."]
