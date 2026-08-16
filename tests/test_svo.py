"""Tests for the SVO extractor (news_deframe.parser.svo).

These tests use a mock spaCy Doc so they run without downloading the
zh_core_web_md or en_core_web_md models.  Structural correctness of the
extraction logic is verified independently of the actual model weights.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from news_deframe.parser.svo import extract_svo, passive_ratio
from news_deframe.parser.spacy_loader import detect_language
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
    """Build a minimal fake spaCy Token with a multi-pass ``children`` iterator.

    ``children`` is stored internally so that each access returns a fresh
    iterator — mirroring spaCy's actual behaviour where ``token.children``
    is a generator that can be consumed multiple times.
    """
    _children_list: list = list(children or [])

    # Create a unique subclass per token so the property is truly per-instance.
    class _FakeToken(MagicMock):
        @property  # type: ignore[override]
        def children(self):
            return iter(_children_list)

        @children.setter
        def children(self, value):
            # Allow tests to reassign children after construction
            _children_list.clear()
            _children_list.extend(value if not hasattr(value, "__next__") else list(value))

    tok = _FakeToken()
    tok.text = text
    tok.text_with_ws = text + " "
    tok.pos_ = pos_
    tok.dep_ = dep_
    tok.lemma_ = lemma_ or text
    tok.i = i
    tok.subtree = [tok]
    tok.head = head if head is not None else tok
    return tok


def _make_sent(tokens: list[MagicMock], text: str) -> MagicMock:
    """Build a minimal fake spaCy Span (sentence)."""
    sent = MagicMock()
    sent.text = text
    sent.__iter__ = lambda self: iter(tokens)
    return sent


def _make_doc(sentences: list[MagicMock], text: str = "") -> MagicMock:
    """Build a minimal fake spaCy Doc."""
    doc = MagicMock()
    doc.sents = sentences
    doc.text = text
    return doc


# ─── Tests: Chinese SVO ───────────────────────────────────────────────────────

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

        results = extract_svo(doc, lang="zh")

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

        results = extract_svo(doc, lang="zh")
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


# ─── Tests: English SVO ───────────────────────────────────────────────────────

class TestExtractSVOEnglish:
    """English active / passive SVO extraction using mocked tokens."""

    def test_english_active_svo(self):
        """Police arrested a suspect → active, subject=Police, object=suspect."""
        subj = _make_token("Police", pos_="NOUN", dep_="nsubj", i=0)
        verb = _make_token("arrested", pos_="VERB", dep_="ROOT", i=1, lemma_="arrest")
        obj_ = _make_token("suspect", pos_="NOUN", dep_="dobj", i=3)

        subj.children = iter([])
        subj.subtree = [subj]
        subj.head = verb

        obj_.children = iter([])
        obj_.subtree = [obj_]
        obj_.head = verb

        verb.children = iter([subj, obj_])
        verb.subtree = [subj, verb, obj_]
        verb.head = verb

        sent_text = "Police arrested a suspect yesterday."
        sent = _make_sent([subj, verb, obj_], sent_text)
        doc = _make_doc([sent], text=sent_text)

        results = extract_svo(doc, lang="en")

        assert len(results) == 1
        record = results[0]
        assert record.verb == "arrest"
        assert "Police" in record.subjects
        assert record.is_passive is False
        assert record.voice_markers == []

    def test_english_passive_was_arrested_by(self):
        """'A man was arrested by police' → passive via aux:pass + agent."""
        # Tokens: man(nsubj:pass) was(aux:pass) arrested(ROOT) by(agent) police
        was = _make_token("was", pos_="AUX", dep_="aux:pass", i=1)
        subj = _make_token("man", pos_="NOUN", dep_="nsubj:pass", i=0)
        verb = _make_token("arrested", pos_="VERB", dep_="ROOT", i=2, lemma_="arrest")
        by_tok = _make_token("by", pos_="ADP", dep_="agent", i=3)

        was.children = iter([])
        was.subtree = [was]
        was.head = verb

        subj.children = iter([])
        subj.subtree = [subj]
        subj.head = verb

        by_tok.children = iter([])
        by_tok.subtree = [by_tok]
        by_tok.head = verb

        verb.children = iter([subj, was, by_tok])
        verb.subtree = [subj, was, verb, by_tok]
        verb.head = verb

        sent_text = "A man was arrested by police amid violent clashes."
        sent = _make_sent([subj, was, verb, by_tok], sent_text)
        doc = _make_doc([sent], text=sent_text)

        results = extract_svo(doc, lang="en")

        assert len(results) >= 1
        record = results[0]
        assert record.is_passive is True
        assert any(m in record.voice_markers for m in ("was", "by"))

    def test_english_passive_has_been_charged(self):
        """'He has been charged' → passive via auxpass-style dep."""
        has = _make_token("has", pos_="AUX", dep_="aux", i=1)
        been = _make_token("been", pos_="AUX", dep_="aux:pass", i=2)
        subj = _make_token("He", pos_="PRON", dep_="nsubj:pass", i=0)
        verb = _make_token("charged", pos_="VERB", dep_="ROOT", i=3, lemma_="charge")

        has.children = iter([])
        has.subtree = [has]
        has.head = verb

        been.children = iter([])
        been.subtree = [been]
        been.head = verb

        subj.children = iter([])
        subj.subtree = [subj]
        subj.head = verb

        verb.children = iter([subj, has, been])
        verb.subtree = [subj, has, been, verb]
        verb.head = verb

        sent_text = "He has been charged."
        sent = _make_sent([subj, has, been, verb], sent_text)
        doc = _make_doc([sent], text=sent_text)

        results = extract_svo(doc, lang="en")

        assert len(results) >= 1
        assert results[0].is_passive is True

    def test_english_active_is_not_passive(self):
        """Ensure an active English sentence is never marked passive."""
        subj = _make_token("Officers", pos_="NOUN", dep_="nsubj", i=0)
        verb = _make_token("detained", pos_="VERB", dep_="ROOT", i=1, lemma_="detain")
        obj_ = _make_token("man", pos_="NOUN", dep_="dobj", i=2)

        subj.children = iter([])
        subj.subtree = [subj]
        subj.head = verb

        obj_.children = iter([])
        obj_.subtree = [obj_]
        obj_.head = verb

        verb.children = iter([subj, obj_])
        verb.subtree = [subj, verb, obj_]
        verb.head = verb

        sent_text = "Officers detained the man without incident."
        sent = _make_sent([subj, verb, obj_], sent_text)
        doc = _make_doc([sent], text=sent_text)

        results = extract_svo(doc, lang="en")

        assert len(results) == 1
        assert results[0].is_passive is False


# ─── Tests: Language Detection & Routing ──────────────────────────────────────

class TestLanguageDetection:
    """Verify detect_language routing logic without loading any models."""

    def test_chinese_text_detected_as_zh(self):
        text = "警方昨日逮捕了涉嫌纵火的男子。"
        assert detect_language(text) == "zh"

    def test_english_text_detected_as_en(self):
        text = "Police arrested a suspect yesterday in connection with an arson attack."
        assert detect_language(text) == "en"

    def test_empty_string_defaults_to_zh(self):
        assert detect_language("") == "zh"

    def test_mixed_but_mostly_cjk_is_zh(self):
        # More CJK than latin → zh
        text = "警察逮捕了 suspect 男子 in the area"
        assert detect_language(text) == "zh"

    def test_mixed_but_mostly_latin_is_en(self):
        # A few CJK characters among mostly English words → en
        text = "The police arrested 人 in downtown."
        assert detect_language(text) == "en"

    def test_get_nlp_for_lang_routes_correctly(self):
        """get_nlp_for_lang calls spacy.load with the right model name."""
        from news_deframe.parser.spacy_loader import get_nlp_for_lang, _cache, ZH_MODEL, EN_MODEL

        fake_model = MagicMock()

        with patch("news_deframe.parser.spacy_loader._cache", {}):
            with patch("spacy.load", return_value=fake_model) as mock_load:
                get_nlp_for_lang("en")
                mock_load.assert_called_once_with(EN_MODEL)

        with patch("news_deframe.parser.spacy_loader._cache", {}):
            with patch("spacy.load", return_value=fake_model) as mock_load:
                get_nlp_for_lang("zh")
                mock_load.assert_called_once_with(ZH_MODEL)

    def test_get_nlp_uses_text_for_detection(self):
        """get_nlp(text) routes to the correct model based on detected language."""
        from news_deframe.parser.spacy_loader import get_nlp, EN_MODEL, ZH_MODEL

        fake_model = MagicMock()

        with patch("news_deframe.parser.spacy_loader._cache", {}):
            with patch("spacy.load", return_value=fake_model) as mock_load:
                get_nlp("Police arrested a suspect.")
                mock_load.assert_called_once_with(EN_MODEL)

        with patch("news_deframe.parser.spacy_loader._cache", {}):
            with patch("spacy.load", return_value=fake_model) as mock_load:
                get_nlp("警察逮捕了男子。")
                mock_load.assert_called_once_with(ZH_MODEL)

    def test_missing_en_model_raises_runtime_error(self):
        """RuntimeError with install instructions when en_core_web_md is absent."""
        from news_deframe.parser.spacy_loader import get_nlp_for_lang, EN_MODEL

        with patch("news_deframe.parser.spacy_loader._cache", {}):
            with patch("spacy.load", side_effect=OSError("not found")):
                with pytest.raises(RuntimeError, match=EN_MODEL):
                    get_nlp_for_lang("en")


# ─── Tests: passive_ratio ─────────────────────────────────────────────────────

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
