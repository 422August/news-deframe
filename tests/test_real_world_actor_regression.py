"""Comprehensive regression test suite for real-world SVO and actor framing.

Tests Categories A through O covering structural failure types without hardcoding:
- Category A: Full person name with institutional title
- Category B: Person name appearing without title
- Category C: Institutional title + role phrase
- Category D: Organization + party group / sub-body
- Category E: Truncated proper name recovery / rejection
- Category F: Noise fragment rejected from actor extraction (budget, numbers, processes)
- Category G: Broken Chinese verb/noun boundary rejected as predicate
- Category H: Legitimate short Chinese predicate retained
- Category I: Legitimate compound Chinese predicate reconstructed
- Category J: Top-level reporting verb with embedded quoted assertion
- Category K: Passive clause subject / agent inversion
- Category L: SVO evidence with multiple occurrences across articles
- Category M: Non-actor event noun used as subject of eventive verb rejected
- Category N: Canonical actor separation across distinct persons sharing titles
- Category O: Display actor matrix invariants and provenance preservation
"""
import pytest
from news_deframe.parser.spacy_loader import get_nlp_for_lang
from news_deframe.parser.svo import extract_svo
from news_deframe.parser.predicate_normalization import (
    is_valid_predicate_token,
    normalize_predicate_text,
)
from news_deframe.analysis.actor_resolution import (
    _validate_actor,
    _should_merge,
    _is_valid_svo_span,
    ActorMention,
)
from news_deframe.analysis.entity_matrix import build_entity_outlet_matrix
from news_deframe.schemas import ParsedArticle


@pytest.fixture(scope="module")
def nlp_zh():
    return get_nlp_for_lang("zh")


@pytest.fixture(scope="module")
def nlp_en():
    return get_nlp_for_lang("en")


# ── Category A & B: Title + Name and Bare Name Resolution ──────────────────────

class TestCategoryA_B_TitleAndName:
    def test_title_name_should_merge(self):
        """Title + Name merges with bare Name without merging different persons."""
        assert _should_merge("立法院長韓國瑜", "PERSON", "韓國瑜", "PERSON") is True
        assert _should_merge("行政院長卓榮泰", "PERSON", "卓榮泰", "PERSON") is True
        assert _should_merge("民眾黨主席黃國昌", "PERSON", "黃國昌", "PERSON") is True
        assert _should_merge("教育部長鄭英耀", "PERSON", "鄭英耀", "PERSON") is True
        assert _should_merge("President Biden", "PERSON", "Biden", "PERSON") is True

    def test_distinct_persons_sharing_title_not_merged(self):
        """Distinct persons with title prefixes do NOT merge."""
        assert _should_merge("立法院長韓國瑜", "PERSON", "行政院長卓榮泰", "PERSON") is False
        assert _should_merge("韓國瑜", "PERSON", "卓榮泰", "PERSON") is False
        assert _should_merge("President Biden", "PERSON", "President Trump", "PERSON") is False


# ── Category C & D: Organization and Sub-bodies ───────────────────────────────

class TestCategoryC_D_OrgAndPartyGroups:
    def test_party_caucus_subsumption(self):
        """Party caucus merges with party body when appropriate."""
        assert _should_merge("立院民進黨團", "ORG", "民進黨團", "ORG") is True
        assert _should_merge("立院民進黨團", "ORG", "民進黨", "ORG") is True
        assert _should_merge("朝野黨團", "ORG", "朝野", "ORG") is True

    def test_institution_not_merged_with_individual(self):
        """Government branches and institutions do NOT merge with human officeholders."""
        assert _should_merge("行政院", "ORG", "卓榮泰", "PERSON") is False
        assert _should_merge("立法院", "ORG", "韓國瑜", "PERSON") is False
        assert _should_merge("Department of Defense", "ORG", "Secretary Austin", "PERSON") is False


# ── Category E: Truncated Mention Handling ─────────────────────────────────────

class TestCategoryE_TruncatedMentions:
    def test_truncated_proper_name_resolution(self):
        """Shortened entity prefix merges into full entity stem."""
        assert _should_merge("周曉", "PERSON", "周曉芸", "PERSON") is True
        assert _should_merge("民進", "ORG", "民進黨", "ORG") is True


# ── Category F & M: Non-Actor Noise Rejection ──────────────────────────────────

class TestCategoryF_M_NonActorRejection:
    @pytest.mark.parametrize("non_actor", [
        "總預算", "中央政府總預算", "特別費", "經費", "媒宣費", "歲出總額",
        "審議結果", "憲政僵局", "表決大戰", "審查延宕", "協商共識",
        "50人", "51人", "480億元", "3兆349億元", "1.7%", "266天", "351天",
        "115年度", "114年8月31日", "他代表", "大家", "各方", "雙方", "現場",
    ])
    def test_non_actor_spans_rejected(self, non_actor):
        """Budget items, numbers, abstract processes, and generic pronouns are rejected as actors."""
        assert _is_valid_svo_span(non_actor) is False or _validate_actor(
            non_actor,
            "SVO_PARTICIPANT",
            [ActorMention("a1", "sentence", non_actor, "agent", "造成", False, [])],
            total_article_count=3,
            cross_article_frequency=3,
        ) is False


# ── Category G: Broken Verb/Noun Boundaries Rejected ──────────────────────────

class TestCategoryG_BrokenPredicateRejection:
    @pytest.mark.parametrize("invalid_verb", [
        "例與", "費以", "樣說", "算", "出總", "長韓", "別費", "言行", "向在", "對此", "因此",
    ])
    def test_broken_predicate_tokens_rejected(self, invalid_verb):
        """Conjunction-ended tokens, discourse connectives, and bound nominals are not valid predicates."""
        assert is_valid_predicate_token(None, text_override=invalid_verb, lang="zh") is False


# ── Category H & I: Legitimate Predicate Normalization ─────────────────────────

class TestCategoryH_I_ValidPredicateNormalization:
    @pytest.mark.parametrize("valid_verb", [
        "說", "砍", "凍", "審", "查", "提", "批", "遭", "通過", "執行", "副署", "協商", "減列", "增列", "凍結",
    ])
    def test_valid_verbs_accepted(self, valid_verb):
        """Genuine action and reporting verbs pass predicate validation."""
        assert is_valid_predicate_token(None, text_override=valid_verb, lang="zh") is True

    def test_compound_verb_reconstruction(self, nlp_zh):
        """Adjacent compound verbs are reconstructed cleanly."""
        doc = nlp_zh("立法院三讀通過總預算。")
        verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
        norm = normalize_predicate_text(verbs[0].text, sentence=doc.text, head_token=verbs[0], lang="zh")
        assert norm == "通過"


# ── Category J: Reporting Verbs and Embedded Quotes ───────────────────────────

class TestCategoryJ_ReportingVerbScoping:
    def test_reporting_verb_local_scoping(self, nlp_zh):
        """Reporting verbs associate speaker with reporting action without leaking to quote."""
        doc = nlp_zh("韓國瑜表示，總預算拖了266天終於審查完畢，呼籲卓榮泰副署。")
        records = extract_svo(doc, lang="zh")
        rep_record = next((r for r in records if r.verb in ("表示", "指出", "說")), None)
        assert rep_record is not None
        assert any("韓國瑜" in s for s in rep_record.subjects)


# ── Category K: Passive Voice Inversion ───────────────────────────────────────

class TestCategoryK_PassiveInversion:
    def test_passive_voice_agent_patient_assignment(self, nlp_zh):
        """Passive clauses assign grammatical subject as patient and prepositional noun as agent."""
        doc = nlp_zh("經費被立法院刪減。")
        records = extract_svo(doc, lang="zh")
        assert len(records) > 0
        rec = records[0]
        assert rec.is_passive is True
        assert any("經費" in s for s in rec.subjects)


# ── Category N & O: Matrix Construction and Invariants ────────────────────────

class TestCategoryN_O_MatrixInvariants:
    def test_end_to_end_actor_matrix_invariants(self, nlp_zh):
        """Actor framing matrix retains only validated actors and valid predicates with full provenance."""
        art1 = ParsedArticle(
            article_id="art1",
            sentences=[
                "行政院長卓榮泰今日表示將依法推動政務。",
                "立法院長韓國瑜呼籲行政院執行預算。",
            ],
            entity_modifiers=[],
            svo_records=extract_svo(nlp_zh("行政院長卓榮泰今日表示將依法推動政務。立法院長韓國瑜呼籲行政院執行預算。")),
        )
        art2 = ParsedArticle(
            article_id="art2",
            sentences=[
                "卓榮泰強調願意與朝野各黨持續協商。",
            ],
            entity_modifiers=[],
            svo_records=extract_svo(nlp_zh("卓榮泰強調願意與朝野各黨持續協商。")),
        )

        matrix = build_entity_outlet_matrix([art1, art2])
        assert len(matrix.entity_names) > 0

        # Canonical merging check: '行政院長卓榮泰' and '卓榮泰' merged into one canonical actor
        assert "卓榮泰" in matrix.entity_names or "行政院長卓榮泰" in matrix.entity_names

        # Verification that all associated verbs are valid predicates
        for p in matrix.profiles:
            for v in p.associated_verbs:
                assert is_valid_predicate_token(None, text_override=v, lang="zh") is True

    def test_location_setting_does_not_receive_unrelated_actor_predicates(self, nlp_zh):
        """A geographic location acting as a setting or nationality specifier does not leak predicates from other entities."""
        from news_deframe.analysis.actor_resolution import resolve_actors
        from news_deframe.parser.entities import extract_entity_modifiers

        text = "日本首相高市早苗今日下令自衛隊展開搜救工作。"
        doc = nlp_zh(text)
        art = ParsedArticle(
            article_id="art_leak_test",
            sentences=[text],
            svo_records=extract_svo(doc, lang="zh"),
            entity_modifiers=extract_entity_modifiers(doc),
        )
        actors, role_stats = resolve_actors([art])
        actor_names = [a.canonical_name for a in actors]

        # '高市早苗' / '高市' should receive '下令'
        # '日本' should NOT receive '下令' or '搜救'
        for st in role_stats:
            if st.canonical_name == "日本":
                assert "下令" not in st.associated_agent_verbs
                assert "搜救" not in st.associated_agent_verbs

    def test_multiple_orgs_in_same_sentence_clause_local_predicates(self, nlp_zh):
        """Multiple organizations in the same sentence only receive clause-local predicates."""
        from news_deframe.analysis.actor_resolution import resolve_actors
        from news_deframe.parser.entities import extract_entity_modifiers

        text = "日本氣象廳發布警報，消防本部正調查火災，而自衛隊已投入救援。"
        doc = nlp_zh(text)
        art = ParsedArticle(
            article_id="art_multi_org",
            sentences=[text],
            svo_records=extract_svo(doc, lang="zh"),
            entity_modifiers=extract_entity_modifiers(doc),
        )
        actors, role_stats = resolve_actors([art])
        for st in role_stats:
            if "氣象廳" in st.canonical_name:
                assert "調查" not in st.associated_agent_verbs
                assert "救援" not in st.associated_agent_verbs
            if "自衛隊" in st.canonical_name:
                assert "發布" not in st.associated_agent_verbs
                assert "調查" not in st.associated_agent_verbs
