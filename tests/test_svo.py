"""Tests for the SVO extractor (news_deframe.parser.svo).

These tests use a mock spaCy Doc so they run without downloading the
zh_core_web_md model.  Structural correctness of the extraction logic is
verified independently of the actual model weights.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from news_deframe.parser.svo import extract_svo, passive_ratio
from news_deframe.schemas import SVORecord


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_token(
    text: str,
    pos_: str = "NOUN",
    dep_: str = "nsubj",
    lemma_: str = "",
    i: int = 0,
    children: list | None = None,
    head: MagicMock | None = None,
) -> MagicMock:
    """Build a minimal fake spaCy Token."""
    tok = MagicMock()
    tok.text = text
    tok.text_with_ws = text + " "
    tok.pos_ = pos_
    tok.dep_ = dep_
    tok.lemma_ = lemma_ or text
    tok.i = i
    tok.children = iter(children or [])
    tok.subtree = [tok]
    # head defaults to self (root)
    tok.head = head if head is not None else tok
    return tok


def _make_sent(tokens: list[MagicMock], text: str) -> MagicMock:
    """Build a minimal fake spaCy Span (sentence)."""
    sent = MagicMock()
    sent.text = text
    sent.__iter__ = lambda self: iter(tokens)
    return sent


def _make_doc(sentences: list[MagicMock]) -> MagicMock:
    """Build a minimal fake spaCy Doc."""
    doc = MagicMock()
    doc.sents = sentences
    return doc


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestExtractSVO:
    def test_returns_list_of_svo_records(self):
        """extract_svo should always return a list."""
        doc = _make_doc([])
        result = extract_svo(doc)
        assert isinstance(result, list)

    def test_empty_doc_returns_empty_list(self):
        doc = _make_doc([])
        assert extract_svo(doc) == []

    def test_sentence_with_no_verb_returns_empty(self):
        noun = _make_token("猫", pos_="NOUN", dep_="ROOT")
        noun.children = iter([])
        sent = _make_sent([noun], "猫")
        doc = _make_doc([sent])
        result = extract_svo(doc)
        assert result == []

    def test_active_svo_extraction(self):
        """A simple active SVO sentence should produce one record."""
        # 警察 逮捕 男子
        subj = _make_token("警察", pos_="NOUN", dep_="nsubj", i=0)
        verb = _make_token("逮捕", pos_="VERB", dep_="ROOT", i=1)
        obj_ = _make_token("男子", pos_="NOUN", dep_="dobj", i=2)

        subj.children = iter([])
        subj.subtree = [subj]
        subj.head = verb

        obj_.children = iter([])
        obj_.subtree = [obj_]
        obj_.head = verb

        verb.children = iter([subj, obj_])
        verb.subtree = [subj, verb, obj_]
        verb.head = verb

        sent_text = "警察逮捕男子"
        sent = _make_sent([subj, verb, obj_], sent_text)
        doc = _make_doc([sent])

        results = extract_svo(doc)

        assert len(results) == 1
        record = results[0]
        assert isinstance(record, SVORecord)
        assert record.verb == "逮捕"
        assert "警察" in record.subjects
        assert "男子" in record.objects
        assert record.is_passive is False

    def test_passive_detection_via_bei(self):
        """Sentences containing 被 should be flagged as passive."""
        bei = _make_token("被", pos_="PART", dep_="auxpass", i=0)
        subj = _make_token("男子", pos_="NOUN", dep_="nsubjpass", i=1)
        verb = _make_token("逮捕", pos_="VERB", dep_="ROOT", i=2)

        bei.children = iter([])
        bei.subtree = [bei]
        bei.head = verb

        subj.children = iter([])
        subj.subtree = [subj]
        subj.head = verb

        verb.children = iter([bei, subj])
        verb.subtree = [bei, subj, verb]
        verb.head = verb

        sent_text = "男子被逮捕"
        sent = _make_sent([bei, subj, verb], sent_text)
        doc = _make_doc([sent])

        results = extract_svo(doc)
        assert len(results) == 1
        assert results[0].is_passive is True
        assert "被" in results[0].voice_markers

    def test_svo_record_fields(self):
        """SVORecord must have all required fields."""
        record = SVORecord(
            sentence="测试句子",
            verb="测试",
            subjects=["主语"],
            objects=["宾语"],
            is_passive=False,
            voice_markers=[],
        )
        assert record.sentence == "测试句子"
        assert record.subjects == ["主语"]
        assert record.objects == ["宾语"]
        assert record.is_passive is False


class TestPassiveRatio:
    def test_empty_list_returns_zero(self):
        assert passive_ratio([]) == 0.0

    def test_all_active(self):
        records = [
            SVORecord(sentence="s", verb="v", subjects=[], objects=[], is_passive=False, voice_markers=[]),
            SVORecord(sentence="s2", verb="v2", subjects=[], objects=[], is_passive=False, voice_markers=[]),
        ]
        assert passive_ratio(records) == 0.0

    def test_all_passive(self):
        records = [
            SVORecord(sentence="s", verb="v", subjects=[], objects=[], is_passive=True, voice_markers=["被"]),
            SVORecord(sentence="s2", verb="v2", subjects=[], objects=[], is_passive=True, voice_markers=["遭"]),
        ]
        assert passive_ratio(records) == 1.0

    def test_half_passive(self):
        records = [
            SVORecord(sentence="s", verb="v", subjects=[], objects=[], is_passive=True, voice_markers=["被"]),
            SVORecord(sentence="s2", verb="v2", subjects=[], objects=[], is_passive=False, voice_markers=[]),
        ]
        assert passive_ratio(records) == pytest.approx(0.5)
