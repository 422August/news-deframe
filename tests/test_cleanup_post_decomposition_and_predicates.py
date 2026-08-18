"""Comprehensive test suite for post-decomposition atomic proposition eligibility and predicate quality."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from news_deframe.analysis.claim_verifier import (
    AtomicProposition,
    ClaimEligibility,
    SentenceProposition,
    check_atomic_proposition_eligibility,
    check_claim_eligibility,
    extract_atomic_propositions,
    extract_structured_quantities,
    is_atomic_proposition_eligible,
    _has_actionable_predicate,
    _reattach_proposition_fragments,
)
from news_deframe.parser.predicate_normalization import (
    is_valid_predicate_token,
    normalize_predicate_text,
    extract_normalized_predicate,
)
from news_deframe.parser.svo import extract_svo
from news_deframe.parser.spacy_loader import get_nlp_for_lang
from news_deframe.schemas import ParsedArticle


def _make_token(
    text: str,
    pos_: str = "NOUN",
    dep_: str = "nsubj",
    lemma_: str = "",
    i: int = 0,
    children: list | None = None,
    head: MagicMock | None = None,
    doc: list | None = None,
) -> MagicMock:
    """Build a mock spaCy Token for unit testing."""
    _children_list: list = list(children or [])

    class _FakeToken(MagicMock):
        @property
        def children(self):
            return iter(_children_list)

        @children.setter
        def children(self, value):
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
    tok.doc = doc
    return tok


class TestPostDecompositionAtomicEligibility:
    """Test Target A: Post-decomposition atomic proposition eligibility and fragment reattachment."""

    def test_valid_proposition_plus_invalid_discourse_fragment_zh(self):
        """Discourse connector '對此' must not become an orphaned claim proposition."""
        sentence = "立法院昨天三讀通過總預算，對此，行政院表達遺憾。"
        props = extract_atomic_propositions("outlet_a", 0, sentence)
        prop_texts = [p.proposition_text for p in props]
        assert len(props) == 2
        assert any("三讀通過總預算" in t for t in prop_texts)
        assert any("行政院表達遺憾" in t for t in prop_texts)
        assert not any(t.strip() == "對此" for t in prop_texts)

    def test_valid_proposition_plus_invalid_discourse_fragment_en(self):
        """English discourse transitions must not become standalone claim propositions."""
        sentence = "The regulatory body approved the license. However, local groups objected."
        props = extract_atomic_propositions("outlet_b", 0, sentence)
        prop_texts = [p.proposition_text for p in props]
        assert len(props) == 2
        assert not any(t.strip() in {"However", "However."} for t in prop_texts)

    def test_detached_attribution_prefix_reattachment_zh(self):
        """Attribution intro '黃國昌受訪表示' forward reattaches to reported clause."""
        sentence = "民眾黨主席黃國昌今天受訪表示，中央總預算審查刪減總金額480億元。"
        props = extract_atomic_propositions("outlet_c", 0, sentence)
        assert len(props) == 1
        assert "480億元" in props[0].proposition_text
        assert is_atomic_proposition_eligible(props[0].proposition_text)

    def test_detached_attribution_prefix_reattachment_en(self):
        """English attribution intro reattaches forward to reported clause."""
        sentence = "The spokesperson stated, overall spending was reduced by 4 billion dollars."
        props = extract_atomic_propositions("outlet_d", 0, sentence)
        assert len(props) == 1
        assert "4 billion dollars" in props[0].proposition_text
        assert is_atomic_proposition_eligible(props[0].proposition_text)

    def test_isolated_connector_rejected_zh(self):
        """Isolated Chinese discourse connectors fail atomic proposition eligibility."""
        for text in ("對此", "此外", "另外", "因此", "不過", "但是", "甚至"):
            assert not is_atomic_proposition_eligible(text)

    def test_isolated_connector_rejected_en(self):
        """Isolated English connectors fail atomic proposition eligibility."""
        for text in ("However", "In addition", "Furthermore", "Moreover", "Therefore"):
            assert not is_atomic_proposition_eligible(text)

    def test_rhetorical_scaffold_rejected_zh(self):
        """Pure rhetorical questions and conversational scaffolding are rejected."""
        rhetorical_cases = [
            "昨天立院民進黨團總召不是在朝野協商上面簽名了嗎？",
            "審查的延宕何以致之、孰以致之？",
            "為什麼他要這樣說？",
            "各方講法太多",
            "不予置評",
            "不予評論",
            "到底誰該負責",
        ]
        for text in rhetorical_cases:
            assert not is_atomic_proposition_eligible(text), f"Expected '{text}' to be rejected"

    def test_rhetorical_scaffold_rejected_en(self):
        """English rhetorical scaffolding is rejected."""
        rhetorical_cases = [
            "Why would the committee delay the vote?",
            "Who knows what will happen next?",
            "What would happen to the market?",
        ]
        for text in rhetorical_cases:
            assert not is_atomic_proposition_eligible(text), f"Expected '{text}' to be rejected"

    def test_legitimate_short_factual_proposition_zh(self):
        """Short factual Chinese propositions are preserved."""
        factual_cases = [
            "油價調漲三元。",
            "立法院依法行使職權。",
            "卓榮泰薪資凍結",
            "115年總預算減列480億元",
            "審議結果共計減列480億元。",
        ]
        for text in factual_cases:
            assert is_atomic_proposition_eligible(text), f"Expected '{text}' to be eligible"

    def test_legitimate_short_factual_proposition_en(self):
        """Short factual English propositions are preserved."""
        factual_cases = [
            "Prime Minister resigned.",
            "The factory closed.",
            "Police arrested three protesters.",
            "Fire broke out at midnight.",
            "Stock market suffered heavy losses.",
        ]
        for text in factual_cases:
            assert is_atomic_proposition_eligible(text), f"Expected '{text}' to be eligible"

    def test_quantity_with_valid_semantic_target_zh_and_en(self):
        """Quantities with semantic targets are extracted and eligible."""
        zh_text = "115年度中央政府總預算案原列歲出總額為3兆349億元。"
        en_text = "Overall spending was reduced by 4 billion dollars."
        sq_zh = extract_structured_quantities(zh_text)
        sq_en = extract_structured_quantities(en_text)
        assert len(sq_zh) >= 1
        assert len(sq_en) >= 1
        assert is_atomic_proposition_eligible(zh_text)
        assert is_atomic_proposition_eligible(en_text)

    def test_dependent_fragment_reattachment_zh(self):
        """Topic prefix and quantifier continuation reattach cleanly in Chinese."""
        chunks = ["最受矚目的歲出部分", "原列歲出總額為3兆349億元", "占總預算80%"]
        reattached = _reattach_proposition_fragments(chunks)
        assert len(reattached) == 1
        assert "最受矚目的歲出部分" in reattached[0]
        assert "3兆349億元" in reattached[0]
        assert "占總預算80%" in reattached[0]

    def test_dependent_fragment_reattachment_en(self):
        """Coordinate clause in English decomposes into independent clean propositions."""
        sentence = "The bill passed, and overall spending was reduced by 4 billion dollars."
        props = extract_atomic_propositions("art_x", 0, sentence)
        assert len(props) == 2
        assert props[0].proposition_text == "The bill passed"
        assert props[1].proposition_text == "overall spending was reduced by 4 billion dollars."


class TestPredicateQualityAndReconstruction:
    """Test Target B: Predicate validation, normalization, and actor-predicate link integrity."""

    def test_malformed_noun_tagged_as_verb_rejected(self):
        """Event nouns and budget nouns misclassified as verbs are rejected."""
        malformed = ["大戰", "表決大戰", "人事費", "特別費", "歲出總額", "審議結果", "憲政僵局"]
        for text in malformed:
            assert not is_valid_predicate_token(None, text, lang="zh"), f"Expected '{text}' to be rejected"

    def test_valid_chinese_compound_predicate(self):
        """Standard Chinese compound verbs pass predicate validation."""
        valid_compounds = ["通過", "審查", "表決", "三讀", "凍結", "刪除", "減列", "增列", "執行", "副署", "簽名"]
        for text in valid_compounds:
            assert is_valid_predicate_token(None, text, lang="zh"), f"Expected '{text}' to be valid"

    def test_valid_short_predicate(self):
        """Single-character CJK verbs pass validation."""
        valid_singles = ["說", "稱", "砍", "刪", "凍", "簽", "提", "查", "批", "遭"]
        for text in valid_singles:
            assert is_valid_predicate_token(None, text, lang="zh"), f"Expected '{text}' to be valid"

    def test_reporting_predicate(self):
        """Reporting verbs in both Chinese and English pass validation."""
        for zh_verb in ("表示", "指出", "強調", "呼籲", "認為", "宣布"):
            assert is_valid_predicate_token(None, zh_verb, lang="zh")
        for en_verb in ("said", "stated", "urged", "demanded", "reported", "announced"):
            assert is_valid_predicate_token(None, en_verb, lang="en")

    def test_chinese_compound_verb_reconstruction(self):
        """Split Chinese verb tokens are reconstructed into full compound verbs."""
        # Test syntactic child reconstruction
        t_kan = _make_token("砍", pos_="VERB", dep_="ROOT", i=0)
        t_shan = _make_token("刪", pos_="VERB", dep_="conj", i=1, head=t_kan)
        t_kan.children = [t_shan]
        norm = normalize_predicate_text("砍", head_token=t_kan, lang="zh")
        assert norm == "砍刪"

        # Test string fallback reconstruction
        norm_fallback = normalize_predicate_text("執", sentence="立法院執行公務", lang="zh")
        assert norm_fallback == "執行"

    def test_english_phrasal_verb(self):
        """English phrasal verbs attach their particle and normalize lemma."""
        t_break = _make_token("broke", pos_="VERB", dep_="ROOT", lemma_="break", i=0)
        t_through = _make_token("through", pos_="ADP", dep_="prt", i=1, head=t_break)
        t_break.children = [t_through]
        norm = normalize_predicate_text("broke", head_token=t_break, lang="en")
        assert norm == "break through"

    def test_passive_predicate_resolution(self):
        """Passive Chinese predicates resolve to their lexical action verb."""
        t_bei = _make_token("遭", pos_="VERB", dep_="ROOT", i=0)
        t_daibu = _make_token("逮捕", pos_="VERB", dep_="ccomp", i=1, head=t_bei)
        t_bei.children = [t_daibu]
        norm = normalize_predicate_text("遭", head_token=t_bei, lang="zh")
        assert norm == "逮捕"

    def test_parser_noise_rejection(self):
        """Parser noise, title prefixes, conjunction endings, and discourse openers are rejected."""
        noise_tokens = ["長韓", "席黃", "長卓", "例與", "費以", "定讓", "出總", "請問", "對此", "將表"]
        for token in noise_tokens:
            assert not is_valid_predicate_token(None, token, lang="zh"), f"Expected '{token}' to be rejected as noise"

    def test_actor_predicate_link_integrity(self):
        """Embedded clause action is not attributed to reporting speaker."""
        nlp_zh = get_nlp_for_lang("zh")
        doc = nlp_zh("黃國昌受訪說，行政院長卓榮泰未依法副署法律。")
        records = extract_svo(doc, lang="zh")
        rec_shuo = next((r for r in records if "說" in r.verb), None)
        rec_fushu = next((r for r in records if "副署" in r.verb), None)
        if rec_shuo:
            assert any("黃國昌" in s for s in rec_shuo.subjects)
        if rec_fushu:
            assert not any("黃國昌" in s for s in rec_fushu.subjects)
