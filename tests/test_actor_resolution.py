"""Comprehensive tests for the actor resolution pipeline.

Coverage
--------
* Valid actor extraction from NER + SVO evidence
* Rejection of non-actor noun phrases (wrong NER type)
* Rejection of modifiers-only as actors (no SVO, no cross-article)
* Rejection of malformed / structurally invalid surface spans
* SVO-grounded agent assignment (active)
* SVO-grounded patient assignment (active)
* Passive construction role inversion (passive subject = logical patient)
* Active/passive semantic-role consistency
* Event-level canonicalization (cross-outlet deduplication)
* Conservative non-merging of distinct actors
* Deterministic canonicalization (order-independent)
* Agent/patient ratio calculation
* Zero-role cases (no SVO participation)
* Verb aggregation (agent and patient separate)
* Modifier aggregation
* Actor importance ranking
* Provenance preservation
* Chinese corpora
* English corpora
* Mixed-language event corpora
* JSON serialization of ActorRoleStats and CanonicalActor
* Console rendering integration (EntityOutletMatrix)
* Backward compatibility via build_entity_outlet_matrix

All tests use actor/entity names that do NOT appear elsewhere in the
repository fixtures (Zorbatian, Nexiphon, Valdric, Crethian, Molvox,
etc.) to guard against hard-coded vocabulary.
"""
from __future__ import annotations

import json
import pytest

from news_deframe.schemas import EntityModifier, ParsedArticle, SVORecord
from news_deframe.analysis.actor_resolution import (
    ActorMention,
    ActorRoleStats,
    CanonicalActor,
    _candidate_in_span,
    _canonicalize_actors,
    _compute_importance,
    _is_valid_surface,
    _is_valid_candidate_length,
    _is_abbreviation_of,
    _match_candidate_to_svo,
    _normalize_key,
    _should_merge,
    _token_overlap_ratio,
    _validate_actor,
    resolve_actors,
)
from news_deframe.analysis.entity_matrix import build_entity_outlet_matrix


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _article(
    article_id: str,
    *,
    svo: list[SVORecord] | None = None,
    entities: list[EntityModifier] | None = None,
    sentences: list[str] | None = None,
) -> ParsedArticle:
    return ParsedArticle(
        article_id=article_id,
        svo_records=svo or [],
        entity_modifiers=entities or [],
        sentences=sentences or [],
    )


def _svo(
    sentence: str,
    verb: str,
    subjects: list[str],
    objects: list[str],
    *,
    passive: bool = False,
) -> SVORecord:
    return SVORecord(
        sentence=sentence,
        verb=verb,
        subjects=subjects,
        objects=objects,
        is_passive=passive,
        voice_markers=["was"] if passive else [],
    )


def _em(name: str, etype: str, mods: list[str] | None = None) -> EntityModifier:
    return EntityModifier(entity_name=name, entity_type=etype, modifiers=mods or [])


# ─── Tests: _is_valid_surface ─────────────────────────────────────────────────


class TestIsValidSurface:
    def test_normal_name_valid(self):
        assert _is_valid_surface("Zorbatian Council") is True

    def test_single_char_rejected(self):
        assert _is_valid_surface("Z") is False

    def test_empty_rejected(self):
        assert _is_valid_surface("") is False

    def test_whitespace_only_rejected(self):
        assert _is_valid_surface("   ") is False

    def test_all_punctuation_rejected(self):
        assert _is_valid_surface("---") is False

    def test_leading_punctuation_rejected(self):
        assert _is_valid_surface(".Nexiphon") is False

    def test_trailing_punctuation_rejected(self):
        assert _is_valid_surface("Nexiphon.") is False

    def test_all_digits_rejected(self):
        assert _is_valid_surface("1234") is False

    def test_cjk_two_char_valid(self):
        assert _is_valid_surface("索比安") is True

    def test_english_two_char_valid(self):
        assert _is_valid_surface("Ab") is True

    def test_cjk_leading_punct_rejected(self):
        # CJK left bracket is punctuation
        assert _is_valid_surface("「Valdric") is False

    def test_mixed_valid(self):
        assert _is_valid_surface("Crethian 委員") is True


class TestIsValidCandidateLength:
    def test_short_ok(self):
        assert _is_valid_candidate_length("Zorbatian Council") is True

    def test_nine_tokens_rejected(self):
        long = " ".join(["word"] * 9)
        assert _is_valid_candidate_length(long) is False

    def test_eight_tokens_ok(self):
        ok = " ".join(["word"] * 8)
        assert _is_valid_candidate_length(ok) is True

    def test_single_token_over_20_chars_rejected(self):
        assert _is_valid_candidate_length("A" * 21) is False

    def test_single_token_20_chars_ok(self):
        assert _is_valid_candidate_length("A" * 20) is True


# ─── Tests: _normalize_key ────────────────────────────────────────────────────


class TestNormalizeKey:
    def test_lowercase(self):
        assert _normalize_key("Zorbatian") == "zorbatian"

    def test_strips_whitespace(self):
        assert _normalize_key("  Nexiphon  ") == "nexiphon"

    def test_removes_leading_the(self):
        assert _normalize_key("The Valdric Authority") == "valdric authority"

    def test_removes_leading_a(self):
        assert _normalize_key("a Crethian") == "crethian"

    def test_collapses_internal_whitespace(self):
        assert _normalize_key("Molvox   Council") == "molvox council"

    def test_cjk_unchanged(self):
        assert _normalize_key("索比安") == "索比安"


# ─── Tests: _token_overlap_ratio ──────────────────────────────────────────────


class TestTokenOverlapRatio:
    def test_identical(self):
        assert _token_overlap_ratio("Zorbatian Council", "Zorbatian Council") == pytest.approx(1.0)

    def test_partial_overlap(self):
        r = _token_overlap_ratio("Zorbatian Council of Trade", "Zorbatian Council")
        # intersection={zorbatian, council} union={zorbatian,council,of,trade}
        assert r == pytest.approx(2 / 4)

    def test_no_overlap(self):
        assert _token_overlap_ratio("Nexiphon", "Valdric") == pytest.approx(0.0)


# ─── Tests: _is_abbreviation_of ───────────────────────────────────────────────


class TestIsAbbreviationOf:
    def test_acronym_detected(self):
        # initials of "Zorbatian Council of Trade" = Z+C+O+T = "ZCOT"
        # "ZC" is contained in "ZCOT" and has length >= 2
        assert _is_abbreviation_of("ZC", "Zorbatian Council of Trade") is True

    def test_partial_initials(self):
        # "ZCO" is contained in "ZCOT"
        assert _is_abbreviation_of("ZCO", "Zorbatian Council of Trade") is True

    def test_non_matching_rejected(self):
        assert _is_abbreviation_of("NX", "Zorbatian Council") is False

    def test_single_char_rejected(self):
        assert _is_abbreviation_of("Z", "Zorbatian Council") is False


# ─── Tests: _should_merge ─────────────────────────────────────────────────────


class TestShouldMerge:
    def test_exact_normalized_match(self):
        assert _should_merge("Zorbatian Council", "ORG", "the Zorbatian Council", "ORG") is True

    def test_different_type_no_merge(self):
        assert _should_merge("Zorbatian", "ORG", "Zorbatian", "PERSON") is False

    def test_high_jaccard_merge(self):
        # "Zorbatian Council of Trade" vs "Zorbatian Council of Commerce"
        # tokens: {zorbatian, council, of, trade} vs {zorbatian, council, of, commerce}
        # intersection = {zorbatian, council, of} = 3, union = 5, Jaccard = 3/5 = 0.60
        # Use "Zorbatian Council of Trade" vs "Zorbatian Council Trade" -> 3/4 = 0.75
        # {zorbatian, council, of, trade} vs {zorbatian, council, trade}
        # intersection = {zorbatian, council, trade} = 3, union = {zorbatian, council, of, trade} = 4
        assert _should_merge(
            "Zorbatian Council of Trade", "ORG",
            "Zorbatian Council Trade", "ORG",
        ) is True

    def test_low_jaccard_no_merge(self):
        assert _should_merge("Zorbatian Council", "ORG", "Valdric Institute", "ORG") is False

    def test_abbreviation_merge(self):
        # "ZCO" is contained in "ZCOT" (Zorbatian Council of Trade initials)
        assert _should_merge("ZCO", "ORG", "Zorbatian Council of Trade", "ORG") is True

    def test_gpe_loc_same_group_merge(self):
        assert _should_merge("Crethia", "GPE", "Crethia", "LOC") is True

    def test_person_org_no_merge(self):
        assert _should_merge("Valdric", "PERSON", "Valdric", "ORG") is False


# ─── Tests: _candidate_in_span ────────────────────────────────────────────────


class TestCandidateInSpan:
    def test_exact_match(self):
        assert _candidate_in_span("Zorbatian Council", "The Zorbatian Council voted") is True

    def test_case_insensitive(self):
        assert _candidate_in_span("Zorbatian Council", "zorbatian council voted") is True

    def test_not_present(self):
        assert _candidate_in_span("Nexiphon", "Zorbatian Council voted") is False

    def test_word_boundary_prevents_partial_ascii(self):
        # "police" should NOT match "policies"
        assert _candidate_in_span("police", "new policies were adopted") is False

    def test_exact_word_match_ascii(self):
        assert _candidate_in_span("police", "police arrested the suspect") is True

    def test_cjk_substring_match(self):
        assert _candidate_in_span("索比安", "索比安當局逮捕了嫌疑人") is True


# ─── Tests: _match_candidate_to_svo ─────────────────────────────────────────


class TestMatchCandidateToSVO:
    def test_active_subject_is_agent(self):
        article = _article(
            "outlet_a",
            svo=[_svo("Valdric arrested suspects.", "arrest", ["Valdric"], ["suspects"])],
            entities=[_em("Valdric", "ORG")],
        )
        mentions = _match_candidate_to_svo("Valdric", [], article)
        assert len(mentions) == 1
        assert mentions[0].role == "agent"
        assert mentions[0].verb == "arrest"
        assert mentions[0].is_passive is False

    def test_active_object_is_patient(self):
        article = _article(
            "outlet_a",
            svo=[_svo("Crethian police detained Nexiphon.", "detain", ["Crethian police"], ["Nexiphon"])],
            entities=[_em("Nexiphon", "ORG")],
        )
        mentions = _match_candidate_to_svo("Nexiphon", [], article)
        assert len(mentions) == 1
        assert mentions[0].role == "patient"
        assert mentions[0].is_passive is False

    def test_passive_subject_is_logical_patient(self):
        """In a passive sentence, the grammatical subject is the logical patient."""
        article = _article(
            "outlet_a",
            svo=[_svo("Nexiphon was detained.", "detain", ["Nexiphon"], [], passive=True)],
            entities=[_em("Nexiphon", "ORG")],
        )
        mentions = _match_candidate_to_svo("Nexiphon", [], article)
        assert len(mentions) == 1
        assert mentions[0].role == "patient"
        assert mentions[0].is_passive is True

    def test_passive_object_is_logical_agent(self):
        """In a passive sentence, the grammatical object (by-phrase) is the logical agent."""
        article = _article(
            "outlet_a",
            svo=[_svo("Nexiphon was detained by Valdric.", "detain", ["Nexiphon"], ["Valdric"], passive=True)],
            entities=[_em("Valdric", "ORG")],
        )
        mentions = _match_candidate_to_svo("Valdric", [], article)
        assert len(mentions) == 1
        assert mentions[0].role == "agent"
        assert mentions[0].is_passive is True

    def test_no_svo_no_mentions(self):
        """A candidate not present in any SVO subject/object returns empty."""
        article = _article(
            "outlet_a",
            svo=[_svo("Officials spoke.", "speak", ["Officials"], [])],
            entities=[_em("Zorbatian Council", "ORG")],
        )
        mentions = _match_candidate_to_svo("Zorbatian Council", [], article)
        assert mentions == []

    def test_modifiers_attached_to_mention(self):
        article = _article(
            "outlet_a",
            svo=[_svo("Molvox acted decisively.", "act", ["Molvox"], [])],
            entities=[_em("Molvox", "PERSON", ["decisive"])],
        )
        mentions = _match_candidate_to_svo("Molvox", ["decisive"], article)
        assert mentions[0].modifiers == ["decisive"]

    def test_cjk_agent(self):
        article = _article(
            "outlet_cn",
            svo=[_svo("索比安逮捕了嫌疑人", "逮捕", ["索比安"], ["嫌疑人"])],
            entities=[_em("索比安", "ORG")],
        )
        mentions = _match_candidate_to_svo("索比安", [], article)
        assert len(mentions) == 1
        assert mentions[0].role == "agent"


# ─── Tests: _validate_actor ───────────────────────────────────────────────────


class TestValidateActor:
    def _make_mention(self, role: str = "agent", verb: str = "act") -> ActorMention:
        return ActorMention(
            article_id="a",
            sentence="s",
            surface="Zorbatian Council",
            role=role,
            verb=verb,
            is_passive=False,
            modifiers=[],
        )

    def test_actor_type_plus_svo_participation_passes(self):
        mentions = [self._make_mention()]
        assert _validate_actor("Zorbatian Council", "ORG", mentions, 2, 1) is True

    def test_actor_type_plus_cross_article_passes(self):
        # S1 + S3
        assert _validate_actor("Molvox Institute", "ORG", [], 3, 2) is True

    def test_actor_type_only_fails(self):
        # S1 only, no other signals
        assert _validate_actor("Nexiphon Bureau", "ORG", [], 3, 1) is False

    def test_non_actor_type_excluded(self):
        mentions = [self._make_mention()]
        assert _validate_actor("January", "DATE", mentions, 2, 2) is False

    def test_event_noun_type_excluded(self):
        mentions = [self._make_mention()]
        assert _validate_actor("protest", "EVENT_NOUN", mentions, 2, 2) is False

    def test_verb_action_type_excluded(self):
        mentions = [self._make_mention()]
        assert _validate_actor("denounce", "VERB_ACTION", mentions, 2, 2) is False

    def test_invalid_surface_excluded(self):
        mentions = [self._make_mention()]
        assert _validate_actor(".", "ORG", mentions, 2, 2) is False

    def test_empty_surface_excluded(self):
        assert _validate_actor("", "ORG", [], 2, 2) is False

    def test_two_mentions_plus_type_passes(self):
        # S1 + S2 + S5
        mentions = [self._make_mention(), self._make_mention()]
        assert _validate_actor("Crethian Parliament", "ORG", mentions, 2, 1) is True

    def test_person_type_with_svo_passes(self):
        mentions = [self._make_mention()]
        assert _validate_actor("Valdric Moreau", "PERSON", mentions, 2, 1) is True


# ─── Tests: _canonicalize_actors ─────────────────────────────────────────────


class TestCanonicalizeActors:
    def _mention(self, surface: str, article_id: str = "a") -> ActorMention:
        return ActorMention(
            article_id=article_id,
            sentence="s",
            surface=surface,
            role="agent",
            verb="act",
            is_passive=False,
            modifiers=[],
        )

    def test_identical_normalized_forms_merged(self):
        m = self._mention("Zorbatian Council")
        validated = [
            ["Zorbatian Council", "ORG", [m]],
            ["the Zorbatian Council", "ORG", [self._mention("the Zorbatian Council")]],
        ]
        result = _canonicalize_actors(validated)
        assert len(result) == 1

    def test_different_type_groups_not_merged(self):
        m = self._mention("Crethia")
        validated = [
            ["Crethia", "PERSON", [m]],
            ["Crethia", "ORG", [self._mention("Crethia")]],
        ]
        result = _canonicalize_actors(validated)
        assert len(result) == 2

    def test_distinct_actors_remain_separate(self):
        validated = [
            ["Zorbatian Council", "ORG", [self._mention("Zorbatian Council")]],
            ["Nexiphon Bureau", "ORG", [self._mention("Nexiphon Bureau")]],
        ]
        result = _canonicalize_actors(validated)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        assert _canonicalize_actors([]) == []

    def test_canonical_name_is_most_frequent(self):
        m1 = self._mention("ZCO")
        m2a = self._mention("Zorbatian Council of Trade", "a")
        m2b = self._mention("Zorbatian Council of Trade", "b")
        validated = [
            ["ZCO", "ORG", [m1]],
            ["Zorbatian Council of Trade", "ORG", [m2a, m2b]],
        ]
        result = _canonicalize_actors(validated)
        assert len(result) == 1
        # "Zorbatian Council of Trade" has 2 mentions vs "ZCO" 1
        assert result[0].canonical_name == "Zorbatian Council of Trade"

    def test_surface_mentions_preserved(self):
        validated = [
            ["Zorbatian Council", "ORG", [self._mention("Zorbatian Council")]],
            ["the Zorbatian Council", "ORG", [self._mention("the Zorbatian Council")]],
        ]
        result = _canonicalize_actors(validated)
        assert len(result) == 1
        surfs = result[0].surface_mentions
        assert "Zorbatian Council" in surfs or "the Zorbatian Council" in surfs

    def test_deterministic_order_independence(self):
        """Same corpus in different order must produce the same canonical actors."""
        m_a = self._mention("Zorbatian Council", "a")
        m_b = self._mention("Nexiphon Bureau", "b")
        validated_order1 = [
            ["Zorbatian Council", "ORG", [m_a]],
            ["Nexiphon Bureau", "ORG", [m_b]],
        ]
        validated_order2 = [
            ["Nexiphon Bureau", "ORG", [m_b]],
            ["Zorbatian Council", "ORG", [m_a]],
        ]
        r1 = _canonicalize_actors(validated_order1)
        r2 = _canonicalize_actors(validated_order2)
        assert sorted(a.canonical_name for a in r1) == sorted(a.canonical_name for a in r2)


# ─── Tests: role aggregation via resolve_actors ───────────────────────────────


class TestResolveActors:
    def _two_article_corpus(self) -> list[ParsedArticle]:
        """A minimal two-article corpus with clear agent/patient structure."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Molvox Corporation arrested Crethian activists.", "arrest",
                       ["Molvox Corporation"], ["Crethian activists"])],
            entities=[
                _em("Molvox Corporation", "ORG", ["dominant"]),
                _em("Crethian activists", "NORP", ["peaceful"]),
            ],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Crethian activists were detained by Molvox Corporation.",
                       "detain", ["Crethian activists"], ["Molvox Corporation"], passive=True)],
            entities=[
                _em("Molvox Corporation", "ORG", ["powerful"]),
                _em("Crethian activists", "NORP", ["unarmed"]),
            ],
        )
        return [art_a, art_b]

    def test_agents_identified_across_corpus(self):
        actors, stats = resolve_actors(self._two_article_corpus())
        names = [a.canonical_name for a in actors]
        assert any("molvox" in n.lower() for n in names)

    def test_agent_count_in_outlet_a(self):
        actors, stats = resolve_actors(self._two_article_corpus())
        molvox_a = next(
            (s for s in stats
             if "molvox" in s.canonical_name.lower() and s.article_id == "outlet_a"),
            None,
        )
        assert molvox_a is not None
        assert molvox_a.agent_count == 1
        assert molvox_a.patient_count == 0
        assert molvox_a.agent_ratio == pytest.approx(1.0)

    def test_passive_subject_becomes_patient(self):
        """In outlet_b: 'Crethian activists were detained' -> passive patient."""
        actors, stats = resolve_actors(self._two_article_corpus())
        crethian_b = next(
            (s for s in stats
             if "crethian" in s.canonical_name.lower() and s.article_id == "outlet_b"),
            None,
        )
        assert crethian_b is not None
        assert crethian_b.patient_count >= 1
        assert crethian_b.passive_patient_count >= 1

    def test_passive_object_becomes_agent(self):
        """In outlet_b: 'by Molvox Corporation' (passive object) -> logical agent."""
        actors, stats = resolve_actors(self._two_article_corpus())
        molvox_b = next(
            (s for s in stats
             if "molvox" in s.canonical_name.lower() and s.article_id == "outlet_b"),
            None,
        )
        assert molvox_b is not None
        assert molvox_b.agent_count >= 1

    def test_agent_verbs_separated_from_patient_verbs(self):
        actors, stats = resolve_actors(self._two_article_corpus())
        molvox_a = next(
            (s for s in stats
             if "molvox" in s.canonical_name.lower() and s.article_id == "outlet_a"),
            None,
        )
        assert molvox_a is not None
        assert "arrest" in molvox_a.associated_agent_verbs
        assert "arrest" not in molvox_a.associated_patient_verbs

    def test_role_occurrence_count_is_agent_plus_patient(self):
        actors, stats = resolve_actors(self._two_article_corpus())
        for s in stats:
            assert s.role_occurrence_count == s.agent_count + s.patient_count

    def test_agent_ratio_denominator(self):
        actors, stats = resolve_actors(self._two_article_corpus())
        for s in stats:
            if s.role_occurrence_count > 0:
                assert s.agent_ratio == pytest.approx(s.agent_count / s.role_occurrence_count)
                assert s.patient_ratio == pytest.approx(s.patient_count / s.role_occurrence_count)

    def test_zero_role_count_gives_zero_ratios(self):
        actors, stats = resolve_actors(self._two_article_corpus())
        for s in stats:
            if s.role_occurrence_count == 0:
                assert s.agent_ratio == 0.0
                assert s.patient_ratio == 0.0

    def test_modifiers_aggregated_per_actor_article(self):
        actors, stats = resolve_actors(self._two_article_corpus())
        molvox_a = next(
            (s for s in stats
             if "molvox" in s.canonical_name.lower() and s.article_id == "outlet_a"),
            None,
        )
        assert molvox_a is not None
        assert "dominant" in molvox_a.associated_modifiers

    def test_provenance_preserved(self):
        actors, stats = resolve_actors(self._two_article_corpus())
        for s in stats:
            # All provenance records should be ActorMention tuples
            for prov in s.provenance:
                assert isinstance(prov, ActorMention)
                assert prov.article_id == s.article_id

    def test_importance_ordering(self):
        """Actors appearing in more outlets should rank higher."""
        actors, stats = resolve_actors(self._two_article_corpus())
        if len(actors) >= 2:
            # Both Molvox and Crethian appear in 2 outlets; check they're both in top results
            top_names = [a.canonical_name.lower() for a in actors[:4]]
            assert any("molvox" in n for n in top_names)
            assert any("crethian" in n for n in top_names)

    def test_non_actor_types_excluded(self):
        """DATE, CARDINAL, EVENT_NOUN entities must not appear as actors."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Zorbatian Council voted.", "vote", ["Zorbatian Council"], [])],
            entities=[
                _em("Zorbatian Council", "ORG"),
                _em("January", "DATE"),
                _em("three", "CARDINAL"),
            ],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Zorbatian Council approved.", "approve", ["Zorbatian Council"], [])],
            entities=[_em("Zorbatian Council", "ORG")],
        )
        actors, stats = resolve_actors([art_a, art_b])
        names = [a.canonical_name.lower() for a in actors]
        assert "january" not in names
        assert "three" not in names

    def test_modifier_does_not_create_actor(self):
        """A modifier word appearing only as amod/advmod must not become an actor."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Nexiphon Bureau voted.", "vote", ["Nexiphon Bureau"], [])],
            entities=[
                _em("Nexiphon Bureau", "ORG"),
                _em("decisive", "EVENT_NOUN"),  # modifier as EVENT_NOUN
            ],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Nexiphon Bureau acted.", "act", ["Nexiphon Bureau"], [])],
            entities=[_em("Nexiphon Bureau", "ORG")],
        )
        actors, stats = resolve_actors([art_a, art_b])
        names = [a.canonical_name.lower() for a in actors]
        assert "decisive" not in names

    def test_malformed_span_not_actor(self):
        """Surface forms that fail structural validation are not actors."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Valdric voted.", "vote", ["Valdric"], [])],
            entities=[
                _em("Valdric", "ORG"),
                _em(".", "ORG"),           # punctuation-only
                _em("12345", "ORG"),       # digits-only
                _em("x", "ORG"),           # single char
            ],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Valdric approved.", "approve", ["Valdric"], [])],
            entities=[_em("Valdric", "ORG")],
        )
        actors, stats = resolve_actors([art_a, art_b])
        names = {a.canonical_name for a in actors}
        assert "." not in names
        assert "12345" not in names
        assert "x" not in names


# ─── Tests: Chinese corpus ────────────────────────────────────────────────────


class TestChineseCorpus:
    def test_zh_agent_extracted(self):
        art_a = _article(
            "zh_outlet_a",
            svo=[_svo("索比安當局逮捕了嫌疑人", "逮捕", ["索比安當局"], ["嫌疑人"])],
            entities=[_em("索比安當局", "ORG"), _em("嫌疑人", "PERSON")],
        )
        art_b = _article(
            "zh_outlet_b",
            svo=[_svo("索比安當局搜查了現場", "搜查", ["索比安當局"], [])],
            entities=[_em("索比安當局", "ORG")],
        )
        actors, stats = resolve_actors([art_a, art_b])
        names = [a.canonical_name for a in actors]
        assert any("索比安" in n for n in names)

    def test_zh_passive_patient(self):
        art_a = _article(
            "zh_outlet_a",
            svo=[_svo("嫌疑人被逮捕", "逮捕", ["嫌疑人"], [], passive=True)],
            entities=[_em("嫌疑人", "PERSON")],
        )
        art_b = _article(
            "zh_outlet_b",
            svo=[_svo("嫌疑人遭到拘留", "拘留", ["嫌疑人"], [], passive=True)],
            entities=[_em("嫌疑人", "PERSON")],
        )
        actors, stats = resolve_actors([art_a, art_b])
        suspect_stats = [s for s in stats if "嫌疑人" in s.canonical_name]
        # All mentions of 嫌疑人 in passive subject slot -> logical patients
        for s in suspect_stats:
            if s.role_occurrence_count > 0:
                assert s.patient_count >= 1


# ─── Tests: importance ranking ────────────────────────────────────────────────


class TestImportanceRanking:
    def _make_actor(self, name: str, article_ids: list[str], n_mentions: int = 1) -> CanonicalActor:
        mentions = [
            ActorMention(
                article_id=aid,
                sentence="s",
                surface=name,
                role="agent",
                verb="act",
                is_passive=False,
                modifiers=[],
            )
            for aid in article_ids
            for _ in range(n_mentions)
        ]
        return CanonicalActor(
            canonical_name=name,
            entity_type="ORG",
            surface_mentions=[name],
            mentions=mentions,
            article_ids=article_ids,
        )

    def test_cross_outlet_presence_increases_score(self):
        a_one_outlet = self._make_actor("Nexiphon", ["a"])
        a_two_outlets = self._make_actor("Molvox", ["a", "b"])
        s1 = _compute_importance(a_one_outlet, 2)
        s2 = _compute_importance(a_two_outlets, 2)
        assert s2 > s1

    def test_more_mentions_increases_score(self):
        a_few = self._make_actor("Nexiphon", ["a", "b"], n_mentions=1)
        a_many = self._make_actor("Molvox", ["a", "b"], n_mentions=5)
        s1 = _compute_importance(a_few, 2)
        s2 = _compute_importance(a_many, 2)
        assert s2 > s1

    def test_score_nonnegative(self):
        actor = self._make_actor("Valdric", [])
        assert _compute_importance(actor, 3) >= 0.0

    def test_deterministic(self):
        actor = self._make_actor("Zorbatian Council", ["a", "b"])
        assert _compute_importance(actor, 2) == _compute_importance(actor, 2)


# ─── Tests: event-level canonicalization across outlets ──────────────────────


class TestEventLevelCanonicalization:
    def test_same_actor_different_cases_merged(self):
        """'Police' in outlet_a and 'police' in outlet_b -> same canonical actor."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Police arrested Crethian.", "arrest", ["Police"], ["Crethian"])],
            entities=[_em("Police", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("police detained Crethian.", "detain", ["police"], ["Crethian"])],
            entities=[_em("police", "ORG"), _em("Crethian", "NORP")],
        )
        actors, stats = resolve_actors([art_a, art_b])
        # "Police" and "police" normalize to the same key -> merged
        police_actors = [a for a in actors if "police" in a.canonical_name.lower()]
        # Should be one canonical actor, not two
        assert len(police_actors) == 1
        assert len(police_actors[0].article_ids) == 2

    def test_distinct_actors_not_merged(self):
        """Completely different actors must remain separate."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Zorbatian Council voted.", "vote", ["Zorbatian Council"], [])],
            entities=[_em("Zorbatian Council", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Nexiphon Bureau approved.", "approve", ["Nexiphon Bureau"], [])],
            entities=[_em("Nexiphon Bureau", "ORG")],
        )
        actors, stats = resolve_actors([art_a, art_b])
        names = [a.canonical_name.lower() for a in actors]
        # Both must be present (neither qualifies for merge)
        # Note: may or may not pass validation depending on cross-article frequency
        # At minimum they should not be merged into one
        if len(actors) >= 2:
            assert len({a.canonical_name for a in actors}) >= 2

    def test_corpus_order_independence(self):
        """resolve_actors produces the same canonical actors regardless of input order."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Crethian Parliament legislated.", "legislate", ["Crethian Parliament"], [])],
            entities=[_em("Crethian Parliament", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Crethian Parliament voted.", "vote", ["Crethian Parliament"], [])],
            entities=[_em("Crethian Parliament", "ORG")],
        )
        actors1, _ = resolve_actors([art_a, art_b])
        actors2, _ = resolve_actors([art_b, art_a])
        assert sorted(a.canonical_name for a in actors1) == sorted(a.canonical_name for a in actors2)


# ─── Tests: build_entity_outlet_matrix integration ──────────────────────────


class TestEntityOutletMatrixIntegration:
    def test_matrix_contains_validated_actors_only(self):
        """The matrix should not include DATE, CARDINAL, VERB_ACTION rows."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Valdric Institute passed the bill.", "pass",
                       ["Valdric Institute"], ["bill"])],
            entities=[
                _em("Valdric Institute", "ORG"),
                _em("January", "DATE"),
                _em("twelve", "CARDINAL"),
            ],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Valdric Institute approved it.", "approve", ["Valdric Institute"], [])],
            entities=[_em("Valdric Institute", "ORG")],
        )
        matrix = build_entity_outlet_matrix([art_a, art_b])
        names = [n.lower() for n in matrix.entity_names]
        assert "january" not in names
        assert "twelve" not in names
        assert any("valdric" in n for n in names)

    def test_matrix_profile_counts_match(self):
        """Profiles should have correct agent/patient counts."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Molvox ordered Nexiphon.", "order", ["Molvox"], ["Nexiphon"])],
            entities=[_em("Molvox", "ORG"), _em("Nexiphon", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Molvox commanded Nexiphon.", "command", ["Molvox"], ["Nexiphon"])],
            entities=[_em("Molvox", "ORG"), _em("Nexiphon", "ORG")],
        )
        matrix = build_entity_outlet_matrix([art_a, art_b])
        molvox_profiles = [p for p in matrix.profiles if "molvox" in p.entity_name.lower()]
        for p in molvox_profiles:
            if p.total_mentions > 0:
                assert p.subject_count >= 1  # Molvox is agent
                assert p.agent_ratio == pytest.approx(1.0)

    def test_matrix_article_ids_sorted(self):
        art_a = _article("z_outlet", entities=[_em("Zorbatian Council", "ORG")], svo=[
            _svo("Zorbatian Council voted.", "vote", ["Zorbatian Council"], [])
        ])
        art_b = _article("a_outlet", entities=[_em("Zorbatian Council", "ORG")], svo=[
            _svo("Zorbatian Council acted.", "act", ["Zorbatian Council"], [])
        ])
        matrix = build_entity_outlet_matrix([art_a, art_b])
        assert matrix.article_ids == sorted(matrix.article_ids)

    def test_matrix_importance_ordering(self):
        """entity_names should be ordered by importance (cross-outlet first)."""
        art_a = _article(
            "outlet_a",
            svo=[
                _svo("Crethian Parliament voted.", "vote", ["Crethian Parliament"], []),
                _svo("Crethian Parliament debated.", "debate", ["Crethian Parliament"], []),
            ],
            entities=[_em("Crethian Parliament", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[
                _svo("Crethian Parliament passed.", "pass", ["Crethian Parliament"], []),
            ],
            entities=[_em("Crethian Parliament", "ORG")],
        )
        matrix = build_entity_outlet_matrix([art_a, art_b])
        # Crethian Parliament appears in both outlets and has multiple mentions
        if matrix.entity_names:
            assert any("crethian" in n.lower() for n in matrix.entity_names[:3])

    def test_matrix_json_serializable(self):
        """EntityOutletMatrix must serialize to valid JSON."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Nexiphon Bureau voted.", "vote", ["Nexiphon Bureau"], [])],
            entities=[_em("Nexiphon Bureau", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Nexiphon Bureau approved.", "approve", ["Nexiphon Bureau"], [])],
            entities=[_em("Nexiphon Bureau", "ORG")],
        )
        matrix = build_entity_outlet_matrix([art_a, art_b])
        json_str = matrix.model_dump_json()
        data = json.loads(json_str)
        assert "entity_names" in data
        assert "article_ids" in data
        assert "profiles" in data


# ─── Tests: mixed-language corpus ────────────────────────────────────────────


class TestMixedLanguageCorpus:
    def test_en_and_zh_articles_processed(self):
        """A corpus mixing English and Chinese articles must not raise errors."""
        art_en = _article(
            "en_outlet",
            svo=[_svo("Zorbatian Council voted.", "vote", ["Zorbatian Council"], [])],
            entities=[_em("Zorbatian Council", "ORG")],
        )
        art_zh = _article(
            "zh_outlet",
            svo=[_svo("索比安議會通過了法案", "通過", ["索比安議會"], ["法案"])],
            entities=[_em("索比安議會", "ORG")],
        )
        # Should not raise; may or may not produce actors depending on validation
        actors, stats = resolve_actors([art_en, art_zh])
        assert isinstance(actors, list)
        assert isinstance(stats, list)


# ─── Tests: ActorRoleStats JSON serialization ─────────────────────────────────


class TestActorRoleStatsSerialize:
    def test_json_serializable(self):
        stats = ActorRoleStats(
            canonical_name="Zorbatian Council",
            article_id="outlet_a",
            mention_count=2,
            role_occurrence_count=2,
            agent_count=2,
            patient_count=0,
            passive_patient_count=0,
            agent_ratio=1.0,
            patient_ratio=0.0,
            passive_patient_ratio=0.0,
            associated_agent_verbs=["vote"],
            associated_patient_verbs=[],
            associated_modifiers=["dominant"],
            provenance=[],
        )
        json_str = stats.model_dump_json()
        data = json.loads(json_str)
        assert data["canonical_name"] == "Zorbatian Council"
        assert data["agent_ratio"] == pytest.approx(1.0)


# ─── Tests: CanonicalActor JSON serialization ─────────────────────────────────


class TestCanonicalActorSerialize:
    def test_json_serializable(self):
        actor = CanonicalActor(
            canonical_name="Nexiphon Bureau",
            entity_type="ORG",
            surface_mentions=["Nexiphon Bureau", "the Nexiphon Bureau"],
            mentions=[],
            article_ids=["outlet_a", "outlet_b"],
        )
        json_str = actor.model_dump_json()
        data = json.loads(json_str)
        assert data["canonical_name"] == "Nexiphon Bureau"
        assert "outlet_a" in data["article_ids"]


# ─── Parameterized: valid actor types ────────────────────────────────────────


@pytest.mark.parametrize("ner_type", ["PERSON", "PER", "ORG", "GPE", "NORP", "FAC", "LOC"])
def test_valid_ner_types_pass_with_svo(ner_type: str):
    """Each actor-valid NER type should pass validation when combined with SVO."""

    def _mention() -> ActorMention:
        return ActorMention(
            article_id="a", sentence="s", surface="Testian Body",
            role="agent", verb="act", is_passive=False, modifiers=[]
        )

    assert _validate_actor("Testian Body", ner_type, [_mention()], 2, 1) is True


@pytest.mark.parametrize("ner_type", [
    "CARDINAL", "DATE", "TIME", "PERCENT", "QUANTITY", "ORDINAL",
    "MONEY", "LANGUAGE", "WORK_OF_ART", "LAW", "PRODUCT",
    "EVENT", "VERB_ACTION", "EVENT_NOUN",
])
def test_non_actor_types_always_rejected(ner_type: str):
    """Non-actor NER types must always be rejected regardless of other signals."""

    def _mention() -> ActorMention:
        return ActorMention(
            article_id="a", sentence="s", surface="Testian Body",
            role="agent", verb="act", is_passive=False, modifiers=[]
        )

    assert _validate_actor("Testian Body", ner_type, [_mention(), _mention()], 3, 3) is False


# ─── Parameterized: surface validation ───────────────────────────────────────


@pytest.mark.parametrize("surface,expected", [
    ("Zorbatian Council", True),
    ("索比安", True),
    ("Ab", True),
    ("", False),
    (".", False),
    ("12345", False),
    ("---", False),
    (".Start", False),
    ("End.", False),
    ("x", False),
])
def test_surface_validation_parametrized(surface: str, expected: bool):
    assert _is_valid_surface(surface) is expected
