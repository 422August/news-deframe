"""Tests for the diff pipeline (aligner + coverage).

Sentence-transformer downloads are avoided by patching ``embed_sentences``
with a deterministic function that returns fixed numpy vectors.
This makes the test suite fast, reproducible, and offline-safe.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from news_deframe.schemas import (
    DiffReport,
    ParsedArticle,
    SentenceAlignment,
    SVORecord,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

def _make_article(
    article_id: str,
    sentences: list[str],
    passive_flags: list[bool] | None = None,
) -> ParsedArticle:
    """Build a minimal ParsedArticle with optional passive flags."""
    flags = passive_flags or [False] * len(sentences)
    svo_records = [
        SVORecord(
            sentence=s,
            verb="verb",
            subjects=[],
            objects=[],
            is_passive=p,
            voice_markers=["被"] if p else [],
        )
        for s, p in zip(sentences, flags)
    ]
    return ParsedArticle(
        article_id=article_id,
        sentences=sentences,
        svo_records=svo_records,
        entity_modifiers=[],
    )


def _fixed_embedder(sentences: list[str]) -> np.ndarray:
    """Return unique, deterministic unit vectors for each sentence.

    Sentence index is encoded as a one-hot-like dimension so that
    sim(sent_i, sent_j) = 1 iff i == j, else ≈ 0.
    """
    dim = max(len(sentences), 16)
    embs = np.zeros((len(sentences), dim), dtype=np.float32)
    for idx in range(len(sentences)):
        embs[idx, idx % dim] = 1.0
    return embs


# ─── Tests: SentenceAlignment schema ──────────────────────────────────────────

class TestSentenceAlignment:
    def test_valid_alignment(self):
        a = SentenceAlignment(sent_a="A", sent_b="B", similarity_score=0.85)
        assert a.sent_b == "B"
        assert a.similarity_score == pytest.approx(0.85)

    def test_none_sent_b_allowed(self):
        a = SentenceAlignment(sent_a="A", sent_b=None, similarity_score=0.0)
        assert a.sent_b is None

    def test_score_bounds(self):
        with pytest.raises(Exception):
            SentenceAlignment(sent_a="A", sent_b=None, similarity_score=1.5)

        with pytest.raises(Exception):
            SentenceAlignment(sent_a="A", sent_b=None, similarity_score=-0.1)


# ─── Tests: align_sentences ────────────────────────────────────────────────────

class TestAlignSentences:
    def test_identical_sentences_get_high_score(self):
        """When sents_a == sents_b, each sentence should align perfectly."""
        from news_deframe.diff.aligner import align_sentences

        sents = ["火灾发生", "警察逮捕", "居民疏散"]

        with patch("news_deframe.diff.aligner.embed_sentences", side_effect=_fixed_embedder):
            results = align_sentences(sents, sents, threshold=0.6)

        assert len(results) == len(sents)
        for r in results:
            assert r.similarity_score >= 0.6
            assert r.sent_b is not None

    def test_empty_sents_a_returns_empty(self):
        from news_deframe.diff.aligner import align_sentences

        with patch("news_deframe.diff.aligner.embed_sentences", side_effect=_fixed_embedder):
            results = align_sentences([], ["something"], threshold=0.6)

        assert results == []

    def test_empty_sents_b_returns_all_unmatched(self):
        from news_deframe.diff.aligner import align_sentences

        sents_a = ["火灾发生"]
        results = align_sentences(sents_a, [], threshold=0.6)

        assert len(results) == 1
        assert results[0].sent_b is None
        assert results[0].similarity_score == 0.0

    def test_low_similarity_gives_no_match(self):
        """Orthogonal vectors should produce zero similarity → no match."""
        from news_deframe.diff.aligner import align_sentences

        sents_a = ["火灾发生"]
        sents_b = ["经济增长"]

        def _orthogonal_embedder(sentences):
            dim = 16
            embs = np.zeros((len(sentences), dim), dtype=np.float32)
            # sents_a gets dim 0, sents_b gets dim 8
            return embs

        # All-zero vectors → dot product == 0
        with patch("news_deframe.diff.aligner.embed_sentences", side_effect=_orthogonal_embedder):
            results = align_sentences(sents_a, sents_b, threshold=0.6)

        assert results[0].sent_b is None


# ─── Tests: compute_coverage ──────────────────────────────────────────────────

class TestComputeCoverage:
    def test_returns_diff_report(self):
        from news_deframe.diff.coverage import compute_coverage

        a = _make_article("A", ["警察逮捕了男子。", "火灾造成三人受伤。"])
        b = _make_article("B", ["一名男子遭警方逮捕。", "三人因火灾送医。"])

        with patch("news_deframe.diff.coverage.embed_sentences", side_effect=_fixed_embedder):
            report = compute_coverage(a, b, threshold=0.6)

        assert isinstance(report, DiffReport)
        assert report.article_a_id == "A"
        assert report.article_b_id == "B"

    def test_passive_ratio_computed(self):
        from news_deframe.diff.coverage import compute_coverage

        # Article A: 1 passive out of 2
        a = _make_article("A", ["s1", "s2"], passive_flags=[True, False])
        b = _make_article("B", ["s3"], passive_flags=[False])

        with patch("news_deframe.diff.coverage.embed_sentences", side_effect=_fixed_embedder):
            report = compute_coverage(a, b, threshold=0.6)

        assert report.passive_ratio_a == pytest.approx(0.5)
        assert report.passive_ratio_b == pytest.approx(0.0)

    def test_empty_article_a(self):
        from news_deframe.diff.coverage import compute_coverage

        a = _make_article("A", [])
        b = _make_article("B", ["内容"])

        with patch("news_deframe.diff.coverage.embed_sentences", side_effect=_fixed_embedder):
            report = compute_coverage(a, b, threshold=0.6)

        assert report.alignments == []
        assert report.unshared_claims_b == ["内容"]

    def test_unshared_claims_populated(self):
        """Orthogonal articles should have all sentences as unshared."""
        from news_deframe.diff.coverage import compute_coverage

        sents_a = [f"A{i}" for i in range(4)]
        sents_b = [f"B{i}" for i in range(4)]

        a = _make_article("A", sents_a)
        b = _make_article("B", sents_b)

        def _block_diag_embedder(sentences: list[str]) -> np.ndarray:
            # first 4 sentences → dims 0-3, next 4 → dims 4-7
            dim = 16
            embs = np.zeros((len(sentences), dim), dtype=np.float32)
            for idx, s in enumerate(sentences):
                if s.startswith("A"):
                    embs[idx, int(s[1:])] = 1.0
                else:
                    embs[idx, int(s[1:]) + 8] = 1.0
            return embs

        with patch("news_deframe.diff.coverage.embed_sentences", side_effect=_block_diag_embedder):
            report = compute_coverage(a, b, threshold=0.6)

        # All A sentences are unshared w.r.t. B (orthogonal block diagonal)
        assert set(report.unshared_claims_a) == set(sents_a)
        assert set(report.unshared_claims_b) == set(sents_b)

    def test_report_alignment_count_matches_sents_a(self):
        from news_deframe.diff.coverage import compute_coverage

        a = _make_article("A", ["s1", "s2", "s3"])
        b = _make_article("B", ["s4", "s5"])

        with patch("news_deframe.diff.coverage.embed_sentences", side_effect=_fixed_embedder):
            report = compute_coverage(a, b, threshold=0.6)

        assert len(report.alignments) == 3
