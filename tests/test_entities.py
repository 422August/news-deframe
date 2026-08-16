"""Offline unit tests for the entities extractor (news_deframe.parser.entities).

All tests use mock spaCy Doc/Token/Span objects so they run without downloading
any models.  Structural correctness of:

* NER boundary-artefact filtering (_is_valid_entity)
* NER modifier collection (_get_entity_modifiers)
* Verb-adverb extraction (_extract_verb_action_modifiers)
* Noun-adjective extraction (_extract_event_noun_modifiers)
* Combined public API (extract_entity_modifiers)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from news_deframe.parser.entities import (
    _is_valid_entity,
    _extract_ner_modifiers,
    _extract_verb_action_modifiers,
    _extract_event_noun_modifiers,
    extract_entity_modifiers,
)
from news_deframe.schemas import EntityModifier


# ─── Token / Span / Doc builder helpers ──────────────────────────────────────


def _make_token(
    text: str,
    pos_: str = "NOUN",
    dep_: str = "nsubj",
    i: int = 0,
    children: list | None = None,
    head: MagicMock | None = None,
) -> MagicMock:
    """Build a minimal fake spaCy Token with a re-iterable ``children`` property."""
    _children_list: list = list(children or [])

    class _FakeToken(MagicMock):
        @property  # type: ignore[override]
        def children(self):
            return iter(_children_list)

        @children.setter
        def children(self, value):
            _children_list.clear()
            _children_list.extend(value if not hasattr(value, "__next__") else list(value))

    tok = _FakeToken()
    tok.text = text
    tok.pos_ = pos_
    tok.dep_ = dep_
    tok.i = i
    tok.head = head if head is not None else tok
    return tok


def _make_span(tokens: list[MagicMock], text: str, label_: str) -> MagicMock:
    """Build a fake spaCy Span (entity) with a root pointing to the last token."""
    span = MagicMock()
    span.text = text
    span.label_ = label_
    span.root = tokens[-1] if tokens else MagicMock()
    span.__iter__ = lambda self: iter(tokens)
    span.__contains__ = lambda self, tok: tok in tokens
    return span


def _make_doc(
    ents: list[MagicMock] | None = None,
    tokens: list[MagicMock] | None = None,
) -> MagicMock:
    """Build a fake spaCy Doc with ``ents`` and token iteration."""
    doc = MagicMock()
    doc.ents = ents or []
    _tokens = tokens or []
    doc.__iter__ = lambda self: iter(_tokens)
    return doc


# ─── Tests: _is_valid_entity ──────────────────────────────────────────────────


class TestIsValidEntity:
    def test_normal_two_char_entity_is_valid(self):
        assert _is_valid_entity("立法院") is True

    def test_single_char_rejected(self):
        assert _is_valid_entity("院") is False

    def test_empty_string_rejected(self):
        assert _is_valid_entity("") is False

    def test_blocklisted_artefact_hui_biao_rejected(self):
        """「會表」 is a known boundary artefact from 「記者會表達」."""
        assert _is_valid_entity("會表") is False

    def test_blocklisted_artefact_jin_chong_rejected(self):
        """「謹衝」 is a known boundary artefact from 「嚴謹衝擊」."""
        assert _is_valid_entity("謹衝") is False

    def test_blocklisted_heisiang_rejected(self):
        """「黑箱」 as a standalone ORG is a common NER hallucination."""
        assert _is_valid_entity("黑箱") is False

    def test_leading_verb_prefix_biaoda_rejected(self):
        """An entity starting with 「表」 (表達) is almost certainly an artefact."""
        assert _is_valid_entity("表達立場") is False

    def test_trailing_function_word_de_rejected(self):
        """An entity ending with the function particle 「的」 is an artefact."""
        assert _is_valid_entity("立法院的") is False

    def test_punctuation_stripped_before_length_check(self):
        """Punctuation-wrapped single char should still be rejected after stripping."""
        assert _is_valid_entity("「院」") is False

    def test_english_name_valid(self):
        assert _is_valid_entity("Taiwan") is True

    def test_two_char_valid(self):
        assert _is_valid_entity("警察") is True


# ─── Tests: NER modifier extraction ──────────────────────────────────────────


class TestExtractNerModifiers:
    def test_noise_label_cardinal_skipped(self):
        """CARDINAL entities must be silently dropped."""
        ent = _make_span([], "三", "CARDINAL")
        doc = _make_doc(ents=[ent])
        results = _extract_ner_modifiers(doc)
        assert results == []

    def test_noise_label_date_skipped(self):
        ent = _make_span([], "昨天", "DATE")
        doc = _make_doc(ents=[ent])
        assert _extract_ner_modifiers(doc) == []

    def test_noise_label_time_skipped(self):
        ent = _make_span([], "下午三點", "TIME")
        doc = _make_doc(ents=[ent])
        assert _extract_ner_modifiers(doc) == []

    def test_artefact_entity_rejected(self):
        """「會表」 labelled PERSON must be filtered out."""
        ent = _make_span([], "會表", "PERSON")
        doc = _make_doc(ents=[ent])
        results = _extract_ner_modifiers(doc)
        assert results == []

    def test_short_entity_rejected(self):
        """Single-character entity must be filtered out regardless of label."""
        ent = _make_span([], "李", "PERSON")
        doc = _make_doc(ents=[ent])
        assert _extract_ner_modifiers(doc) == []

    def test_valid_org_extracted(self):
        """A valid ORG entity with no modifiers should appear in the output."""
        root = _make_token("立法院", pos_="NOUN", dep_="ROOT", i=0)
        ent = _make_span([root], "立法院", "ORG")
        doc = _make_doc(ents=[ent])
        results = _extract_ner_modifiers(doc)
        assert len(results) == 1
        assert results[0].entity_name == "立法院"
        assert results[0].entity_type == "ORG"

    def test_amod_modifier_captured(self):
        """Adjectival modifier (amod) child of entity root should be collected."""
        # "嚴重 立法院" – 嚴重 is amod of 立法院
        adj = _make_token("嚴重", pos_="ADJ", dep_="amod", i=0)
        root = _make_token("立法院", pos_="NOUN", dep_="ROOT", i=1, children=[adj])
        adj.head = root
        ent = _make_span([root], "立法院", "ORG")
        doc = _make_doc(ents=[ent])
        results = _extract_ner_modifiers(doc)
        assert len(results) == 1
        assert "嚴重" in results[0].modifiers

    def test_duplicate_entities_deduplicated(self):
        """Same entity text+label appearing twice should yield one record."""
        root = _make_token("政府", pos_="NOUN", dep_="ROOT", i=0)
        ent1 = _make_span([root], "政府", "ORG")
        ent2 = _make_span([root], "政府", "ORG")
        doc = _make_doc(ents=[ent1, ent2])
        results = _extract_ner_modifiers(doc)
        assert len(results) == 1

    def test_per_label_accepted(self):
        """PER (alternate label used by some zh models) should be accepted."""
        root = _make_token("陳水扁", pos_="NOUN", dep_="ROOT", i=0)
        ent = _make_span([root], "陳水扁", "PER")
        doc = _make_doc(ents=[ent])
        results = _extract_ner_modifiers(doc)
        assert len(results) == 1
        assert results[0].entity_type == "PER"


# ─── Tests: verb-action modifier extraction ───────────────────────────────────


class TestExtractVerbActionModifiers:
    def test_verb_with_advmod_captured(self):
        """草率 (advmod) + 掏空 (VERB) → VERB_ACTION entry with modifier 草率."""
        adv = _make_token("草率", pos_="ADV", dep_="advmod", i=0)
        verb = _make_token("掏空", pos_="VERB", dep_="ROOT", i=1, children=[adv])
        adv.head = verb

        doc = _make_doc(tokens=[adv, verb])
        results = _extract_verb_action_modifiers(doc)

        assert len(results) == 1
        assert results[0].entity_name == "掏空"
        assert results[0].entity_type == "VERB_ACTION"
        assert "草率" in results[0].modifiers

    def test_verb_with_neg_captured(self):
        """不 (neg) attached to a verb should be recorded."""
        neg = _make_token("不", pos_="ADV", dep_="neg", i=0)
        verb = _make_token("承認", pos_="VERB", dep_="ROOT", i=1, children=[neg])
        neg.head = verb

        doc = _make_doc(tokens=[neg, verb])
        results = _extract_verb_action_modifiers(doc)

        assert len(results) == 1
        assert "不" in results[0].modifiers

    def test_verb_with_no_advmod_excluded(self):
        """A VERB with only object children should not be recorded."""
        obj_ = _make_token("法案", pos_="NOUN", dep_="dobj", i=1)
        verb = _make_token("通過", pos_="VERB", dep_="ROOT", i=0, children=[obj_])
        obj_.head = verb

        doc = _make_doc(tokens=[verb, obj_])
        results = _extract_verb_action_modifiers(doc)
        assert results == []

    def test_noun_not_included_in_verb_pass(self):
        """A NOUN token should never appear in verb-action results."""
        adv = _make_token("快速", pos_="ADV", dep_="advmod", i=0)
        noun = _make_token("決議", pos_="NOUN", dep_="ROOT", i=1, children=[adv])
        adv.head = noun

        doc = _make_doc(tokens=[adv, noun])
        results = _extract_verb_action_modifiers(doc)
        # noun is not a VERB, so nothing recorded
        assert results == []

    def test_duplicate_verb_deduplicated(self):
        """Same verb appearing twice in the doc should yield one record."""
        adv1 = _make_token("粗暴", pos_="ADV", dep_="advmod", i=0)
        verb1 = _make_token("破壞", pos_="VERB", dep_="ROOT", i=1, children=[adv1])
        adv1.head = verb1

        adv2 = _make_token("粗暴", pos_="ADV", dep_="advmod", i=2)
        verb2 = _make_token("破壞", pos_="VERB", dep_="ROOT", i=3, children=[adv2])
        adv2.head = verb2

        doc = _make_doc(tokens=[adv1, verb1, adv2, verb2])
        results = _extract_verb_action_modifiers(doc)
        assert len(results) == 1

    def test_english_violently_crash(self):
        """'violently' (advmod) on 'clash' (VERB) captured for English text."""
        adv = _make_token("violently", pos_="ADV", dep_="advmod", i=0)
        verb = _make_token("clash", pos_="VERB", dep_="ROOT", i=1, children=[adv])
        adv.head = verb

        doc = _make_doc(tokens=[adv, verb])
        results = _extract_verb_action_modifiers(doc)
        assert len(results) == 1
        assert results[0].entity_name == "clash"
        assert "violently" in results[0].modifiers


# ─── Tests: event-noun modifier extraction ────────────────────────────────────


class TestExtractEventNounModifiers:
    def test_noun_with_amod_captured(self):
        """合理 (amod) + 分配 (NOUN) → EVENT_NOUN entry."""
        adj = _make_token("合理", pos_="ADJ", dep_="amod", i=0)
        noun = _make_token("分配", pos_="NOUN", dep_="ROOT", i=1, children=[adj])
        adj.head = noun

        doc = _make_doc(tokens=[adj, noun])
        results = _extract_event_noun_modifiers(doc, ner_texts=frozenset())

        assert len(results) == 1
        assert results[0].entity_name == "分配"
        assert results[0].entity_type == "EVENT_NOUN"
        assert "合理" in results[0].modifiers

    def test_noun_already_in_ner_skipped(self):
        """A noun already captured as an NER entity should not be double-counted."""
        adj = _make_token("合理", pos_="ADJ", dep_="amod", i=0)
        noun = _make_token("分配", pos_="NOUN", dep_="ROOT", i=1, children=[adj])
        adj.head = noun

        doc = _make_doc(tokens=[adj, noun])
        # Pretend the NER pass already captured 「分配」
        results = _extract_event_noun_modifiers(doc, ner_texts=frozenset({"分配"}))
        assert results == []

    def test_noun_with_compound_captured(self):
        """A noun with a ``compound`` dep modifier is an EVENT_NOUN."""
        comp = _make_token("黑箱", pos_="NOUN", dep_="compound", i=0)
        noun = _make_token("立法", pos_="NOUN", dep_="ROOT", i=1, children=[comp])
        comp.head = noun

        doc = _make_doc(tokens=[comp, noun])
        results = _extract_event_noun_modifiers(doc, ner_texts=frozenset())
        assert len(results) == 1
        assert results[0].entity_name == "立法"
        assert "黑箱" in results[0].modifiers

    def test_noun_without_modifier_excluded(self):
        """A bare NOUN with no qualifying child dep should be excluded."""
        noun = _make_token("法案", pos_="NOUN", dep_="ROOT", i=0)
        doc = _make_doc(tokens=[noun])
        results = _extract_event_noun_modifiers(doc, ner_texts=frozenset())
        assert results == []

    def test_verb_not_included_in_noun_pass(self):
        """A VERB token should never appear in event-noun results."""
        adj = _make_token("unlawful", pos_="ADJ", dep_="amod", i=0)
        verb = _make_token("assemble", pos_="VERB", dep_="ROOT", i=1, children=[adj])
        adj.head = verb

        doc = _make_doc(tokens=[adj, verb])
        results = _extract_event_noun_modifiers(doc, ner_texts=frozenset())
        assert results == []

    def test_english_unlawful_assembly(self):
        """'unlawful' (amod) + 'assembly' (NOUN) → EVENT_NOUN."""
        adj = _make_token("unlawful", pos_="ADJ", dep_="amod", i=0)
        noun = _make_token("assembly", pos_="NOUN", dep_="ROOT", i=1, children=[adj])
        adj.head = noun

        doc = _make_doc(tokens=[adj, noun])
        results = _extract_event_noun_modifiers(doc, ner_texts=frozenset())
        assert len(results) == 1
        assert results[0].entity_name == "assembly"
        assert "unlawful" in results[0].modifiers


# ─── Tests: combined public API ───────────────────────────────────────────────


class TestExtractEntityModifiers:
    def test_returns_list(self):
        doc = _make_doc()
        result = extract_entity_modifiers(doc)
        assert isinstance(result, list)

    def test_empty_doc_returns_empty(self):
        doc = _make_doc()
        assert extract_entity_modifiers(doc) == []

    def test_combined_ner_and_verb_and_noun(self):
        """Smoke test: a doc with one NER entity, one verb, one noun all contribute."""
        # NER: 立法院 (ORG)
        ent_root = _make_token("立法院", pos_="NOUN", dep_="ROOT", i=2)
        ent_span = _make_span([ent_root], "立法院", "ORG")

        # Verb: 草率 (advmod) → 掏空 (VERB)
        adv = _make_token("草率", pos_="ADV", dep_="advmod", i=3)
        verb = _make_token("掏空", pos_="VERB", dep_="ROOT", i=4, children=[adv])
        adv.head = verb

        # Noun: 合理 (amod) → 分配 (NOUN)
        adj = _make_token("合理", pos_="ADJ", dep_="amod", i=0)
        noun = _make_token("分配", pos_="NOUN", dep_="ROOT", i=1, children=[adj])
        adj.head = noun

        doc = _make_doc(ents=[ent_span], tokens=[adj, noun, ent_root, adv, verb])
        results = extract_entity_modifiers(doc)

        types = {r.entity_type for r in results}
        names = {r.entity_name for r in results}

        assert "ORG" in types
        assert "VERB_ACTION" in types
        assert "EVENT_NOUN" in types
        assert "立法院" in names
        assert "掏空" in names
        assert "分配" in names

    def test_artefact_entity_excluded_from_combined(self):
        """Artefact entity 「會表」 must be absent from final combined output."""
        ent = _make_span([], "會表", "PERSON")
        doc = _make_doc(ents=[ent])
        results = extract_entity_modifiers(doc)
        names = [r.entity_name for r in results]
        assert "會表" not in names

    def test_noise_label_excluded_from_combined(self):
        """CARDINAL entity must be absent from combined output."""
        ent = _make_span([], "三十", "CARDINAL")
        doc = _make_doc(ents=[ent])
        results = extract_entity_modifiers(doc)
        assert results == []

    def test_all_results_are_entity_modifier_instances(self):
        """Every result must be an EntityModifier for backwards compatibility."""
        adv = _make_token("堅定", pos_="ADV", dep_="advmod", i=0)
        verb = _make_token("完成", pos_="VERB", dep_="ROOT", i=1, children=[adv])
        adv.head = verb
        doc = _make_doc(tokens=[adv, verb])
        results = extract_entity_modifiers(doc)
        assert all(isinstance(r, EntityModifier) for r in results)

    def test_ner_entity_not_duplicated_in_noun_pass(self):
        """A noun that is already an NER entity should not appear as EVENT_NOUN."""
        adj = _make_token("重大", pos_="ADJ", dep_="amod", i=0)
        root = _make_token("事件", pos_="NOUN", dep_="ROOT", i=1, children=[adj])
        adj.head = root
        ent_span = _make_span([root], "事件", "EVENT")

        doc = _make_doc(ents=[ent_span], tokens=[adj, root])
        results = extract_entity_modifiers(doc)

        event_noun_results = [r for r in results if r.entity_type == "EVENT_NOUN"]
        assert all(r.entity_name != "事件" for r in event_noun_results)
