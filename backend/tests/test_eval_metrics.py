"""Unit tests for the evaluation harness metrics (no network, no Weaviate)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from metrics import (
    citation_scores,
    first_hit_rank,
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
    section_matches,
)


def chunk(section, text=""):
    return {"section_number": section, "text": text}


class TestSectionMatches:
    def test_exact(self):
        assert section_matches("1011.6", "1011.6")

    def test_descendant_matches(self):
        assert section_matches("1025.6", "1025.6.1")

    def test_prefix_is_not_substring(self):
        assert not section_matches("1011.6", "1011.65")
        assert not section_matches("507.8.1", "507.8.10")

    def test_parent_chunk_carrying_child_inline(self):
        text = "507.8 Group H occupancies. ...\n507.8.1 Allowable area. The aggregate ..."
        assert section_matches("507.8.1", "507.8", text)

    def test_inline_match_requires_line_start(self):
        assert not section_matches("507.8.1", "507.8", "see Section 507.8.1 for details")


class TestRankMetrics:
    ranked = [chunk("100.1"), chunk("200.2"), chunk("300.3")]

    def test_hit_and_rank(self):
        assert first_hit_rank(["200.2"], self.ranked) == 2
        assert hit_at_k(["200.2"], self.ranked, 1) == 0.0
        assert hit_at_k(["200.2"], self.ranked, 2) == 1.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["300.3"], self.ranked, 3) == 1 / 3
        assert reciprocal_rank(["999.9"], self.ranked, 3) == 0.0

    def test_recall_multi_gold(self):
        assert recall_at_k(["100.1", "300.3"], self.ranked, 2) == 0.5
        assert recall_at_k(["100.1", "300.3"], self.ranked, 3) == 1.0

    def test_hit_uses_any_gold(self):
        assert hit_at_k(["999.9", "100.1"], self.ranked, 1) == 1.0


class TestCitationScores:
    retrieved = [chunk("1011.6"), chunk("1011.5")]

    def test_perfect(self):
        s = citation_scores(["1011.6"], ["1011.6"], self.retrieved)
        assert s == {"recall": 1.0, "precision": 1.0, "grounded": 1.0, "referenced": 0.0}

    def test_strips_section_symbol(self):
        assert citation_scores(["§1011.6"], ["1011.6"], self.retrieved)["recall"] == 1.0

    def test_ungrounded_citation_detected(self):
        s = citation_scores(["1011.6", "9999.9"], ["1011.6"], self.retrieved)
        assert s["precision"] == 0.5
        assert s["grounded"] == 0.5
        assert s["referenced"] == 0.0

    def test_inline_child_counts_as_grounded(self):
        parent = [chunk("1025.6", "1025.6 General.\n1025.6.1 Design of door signs.")]
        assert citation_scores(["1025.6.1"], ["1025.6.1"], parent)["grounded"] == 1.0

    def test_cross_reference_is_referenced_not_grounded(self):
        ctx = [chunk("1026.2", "separation shall be a fire barrier complying with Section 707.")]
        s = citation_scores(["1026.2", "707"], ["1026.2"], ctx)
        assert s["grounded"] == 0.5 and s["referenced"] == 0.5

    def test_reference_does_not_match_longer_number(self):
        ctx = [chunk("1.1", "see Section 7071.2 and 707.10")]
        assert citation_scores(["707"], [], ctx)["referenced"] == 0.0

    def test_no_citations(self):
        s = citation_scores([], ["1011.6"], self.retrieved)
        assert s["recall"] == 0.0 and s["precision"] is None


def test_mean_ignores_none():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None
