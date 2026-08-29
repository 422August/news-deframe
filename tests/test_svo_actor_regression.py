"""Regression tests for the SVO-based actor extraction pipeline.

These tests address the structural failures described in the bug report:
- Invalid linguistic spans (adjectives, verb fragments) becoming actors
- Meaningful event participants being lost because NER fails to tag them
- Passive construction role assignment
- Participant span vs. syntactic head distinction
- Canonicalization correctness
- Ranking correctness

All actor names and sentence structures use vocabulary that does NOT appear
in the existing fixture files (Grethovi, Solvanic, Muraxian, Telphas, etc.).

Tests assert semantic/structural properties rather than fixed vocabulary lists.
"""
from __future__ import annotations

import pytest

from news_deframe.schemas import EntityModifier, ParsedArticle, SVORecord
from news_deframe.analysis.actor_resolution import (
    _extract_ner_candidates,
    _extract_svo_candidates,
    _is_valid_svo_span,
    _is_valid_surface,
    _validate_actor,
    _should_merge,
    resolve_actors,
    ActorMention,
    _SVO_DERIVED_TYPE,
)
from news_deframe.analysis.entity_matrix import build_entity_outlet_matrix


# ── Helpers ────────────────────────────────────────────────────────────────────


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


def _mention(surface: str, role: str = "agent", verb: str = "act",
             article_id: str = "a") -> ActorMention:
    return ActorMention(
        article_id=article_id,
        sentence="s",
        surface=surface,
        role=role,
        verb=verb,
        is_passive=False,
        modifiers=[],
    )


# ── Phase 2, Test 1: Adjective must never become an actor ─────────────────────


class TestAdjAdverbNotActor:
    """An adjective or adverb appearing in SVO or NER must not become an actor."""

    def test_adjective_ner_candidate_rejected_from_actor_validation(self):
        """An NER entity typed as a non-actor label is rejected regardless of SVO."""
        # "rapidly" typed as EVENT_NOUN (which is in _NON_ACTOR_NER_LABELS)
        art_a = _article(
            "grethovi_daily",
            svo=[_svo("Solvanic Guard seized rapidly.", "seize",
                      ["Solvanic Guard"], ["rapidly"])],
            entities=[
                _em("Solvanic Guard", "ORG"),
                _em("rapidly", "EVENT_NOUN"),
            ],
        )
        art_b = _article(
            "muraxian_press",
            svo=[_svo("Solvanic Guard detained Telphas.", "detain",
                      ["Solvanic Guard"], ["Telphas"])],
            entities=[_em("Solvanic Guard", "ORG")],
        )
        actors, _ = resolve_actors([art_a, art_b])
        names_lower = [a.canonical_name.lower() for a in actors]
        assert "rapidly" not in names_lower

    def test_svo_only_adverb_single_article_rejected(self):
        """An adverb-like string from SVO in a single article lacks enough signals."""
        art_a = _article(
            "grethovi_daily",
            svo=[_svo("Officials acted swiftly.", "act", ["Officials"], ["swiftly"])],
            entities=[_em("Officials", "ORG")],
        )
        art_b = _article(
            "muraxian_press",
            svo=[_svo("Officials intervened.", "intervene", ["Officials"], [])],
            entities=[_em("Officials", "ORG")],
        )
        actors, _ = resolve_actors([art_a, art_b])
        names_lower = [a.canonical_name.lower() for a in actors]
        # "swiftly" has only 1 SVO mention, 1 article → fails 3-signal threshold
        assert "swiftly" not in names_lower

    def test_is_valid_svo_span_rejects_pure_punctuation(self):
        assert _is_valid_svo_span("...") is False

    def test_is_valid_svo_span_rejects_single_char(self):
        assert _is_valid_svo_span("X") is False

    def test_is_valid_svo_span_rejects_excessively_long_span(self):
        long_span = "a " * 7 + "more"  # 8 space-separated tokens
        assert _is_valid_svo_span(long_span) is False

    def test_is_valid_svo_span_accepts_noun_phrase(self):
        assert _is_valid_svo_span("Solvanic Guard") is True

    def test_is_valid_svo_span_accepts_cjk_span(self):
        assert _is_valid_svo_span("格雷托維當局") is True


# ── Phase 2, Test 2: SVO path recovers NER-missed participants ─────────────────


class TestSVOPathRecovery:
    """Participants that NER fails to tag must be recoverable via the SVO path."""

    def test_svo_subject_without_ner_becomes_actor_cross_article(self):
        """A recurring SVO subject with no NER label becomes an actor via SVO path."""
        # "Grethovi Guards" has no NER entity — only SVO subject position
        art_a = _article(
            "outlet_a",
            svo=[_svo("Grethovi Guards arrested three protesters.", "arrest",
                      ["Grethovi Guards"], ["three protesters"])],
            entities=[],  # NO NER entity for Grethovi Guards
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Grethovi Guards detained the demonstrators.", "detain",
                      ["Grethovi Guards"], ["demonstrators"])],
            entities=[],  # NO NER entity for Grethovi Guards
        )
        actors, _ = resolve_actors([art_a, art_b])
        names_lower = [a.canonical_name.lower() for a in actors]
        # Must appear because: S2 (SVO), S3 (cross-article), S4 (verb), S5 (2+ mentions)
        assert any("grethovi guards" in n for n in names_lower)

    def test_svo_derived_candidate_type_is_sentinel(self):
        """SVO-derived candidates without NER use the SVO_PARTICIPANT sentinel type."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Muraxian Council voted.", "vote", ["Muraxian Council"], [])],
            entities=[],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Muraxian Council decided.", "decide", ["Muraxian Council"], [])],
            entities=[],
        )
        actors, _ = resolve_actors([art_a, art_b])
        muraxian = [a for a in actors if "muraxian" in a.canonical_name.lower()]
        assert len(muraxian) == 1
        assert muraxian[0].entity_type == _SVO_DERIVED_TYPE

    def test_ner_backed_type_preferred_over_svo_sentinel(self):
        """When both NER and SVO paths find the same participant, NER type wins."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Telphas Authority approved.", "approve",
                      ["Telphas Authority"], [])],
            entities=[_em("Telphas Authority", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Telphas Authority confirmed.", "confirm",
                      ["Telphas Authority"], [])],
            entities=[_em("Telphas Authority", "ORG")],
        )
        actors, _ = resolve_actors([art_a, art_b])
        telphas = [a for a in actors if "telphas" in a.canonical_name.lower()]
        assert len(telphas) == 1
        # NER-backed ORG must be chosen over SVO_PARTICIPANT
        assert telphas[0].entity_type == "ORG"

    def test_extract_svo_candidates_extracts_subjects_and_objects(self):
        """_extract_svo_candidates must extract unique subjects and objects."""
        art = _article(
            "outlet_a",
            svo=[
                _svo("Grethovi Guards arrested protesters.", "arrest",
                     ["Grethovi Guards"], ["protesters"]),
                _svo("Grethovi Guards issued a statement.", "issue",
                     ["Grethovi Guards"], ["statement"]),
            ],
            entities=[],
        )
        candidates = _extract_svo_candidates(art)
        surfaces = [c[0] for c in candidates]
        # Grethovi Guards should appear exactly once (deduplication)
        grethovi_count = sum(1 for s in surfaces if "grethovi" in s.lower())
        assert grethovi_count == 1
        # protesters and statement should also be candidates
        assert any("protesters" in s.lower() for s in surfaces)

    def test_extract_svo_candidates_type_is_svo_sentinel(self):
        """All SVO-derived candidates must have the SVO_PARTICIPANT type."""
        art = _article(
            "outlet_a",
            svo=[_svo("Solvanic Bureau voted.", "vote", ["Solvanic Bureau"], [])],
            entities=[],
        )
        candidates = _extract_svo_candidates(art)
        assert all(c[1] == _SVO_DERIVED_TYPE for c in candidates)


# ── Phase 2, Test 3: Abstract/event noun not actor solely from bad SVO ─────────


class TestAbstractNounNotActor:
    """Abstract nouns that appear in SVO only due to parser errors must not become actors."""

    def test_abstract_noun_single_article_fails_validation(self):
        """An abstract noun from SVO in only one article cannot reach actor threshold."""
        # "procedure" appears as SVO object but only in one article
        art_a = _article(
            "outlet_a",
            svo=[_svo("Officials followed procedure.", "follow",
                      ["Officials"], ["procedure"])],
            entities=[_em("Officials", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Officials acted.", "act", ["Officials"], [])],
            entities=[_em("Officials", "ORG")],
        )
        actors, _ = resolve_actors([art_a, art_b])
        names_lower = [a.canonical_name.lower() for a in actors]
        # "procedure" has no NER (no S1), only 1 article (no S3), only 1 mention (no S5)
        # Score: S2=1, S4=1 → total 2, need 3 → rejected
        assert "procedure" not in names_lower

    def test_non_actor_ner_type_excluded_regardless_of_svo(self):
        """EVENT, DATE, CARDINAL etc. must be rejected even with many SVO mentions."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("The regulation was enacted.", "enact",
                      ["regulation"], [], passive=True)],
            entities=[_em("regulation", "EVENT")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("The regulation was adopted.", "adopt",
                      ["regulation"], [], passive=True)],
            entities=[_em("regulation", "EVENT")],
        )
        actors, _ = resolve_actors([art_a, art_b])
        names_lower = [a.canonical_name.lower() for a in actors]
        assert "regulation" not in names_lower


# ── Phase 2, Test 4: Human/group participants lost must now be found ──────────


class TestImportantParticipantsFound:
    """Critical event participants that appear in SVO subject positions must survive."""

    def test_human_group_without_ner_cross_article(self):
        """A human group mentioned in SVO subjects across articles must become an actor."""
        art_a = _article(
            "outlet_a",
            svo=[
                _svo("Demonstrators marched to the square.", "march",
                     ["Demonstrators"], ["square"]),
                _svo("Demonstrators resisted police orders.", "resist",
                     ["Demonstrators"], ["police orders"]),
            ],
            entities=[],
        )
        art_b = _article(
            "outlet_b",
            svo=[
                _svo("Demonstrators were arrested.", "arrest",
                     ["Demonstrators"], [], passive=True),
                _svo("Demonstrators rejected the demand.", "reject",
                     ["Demonstrators"], ["demand"]),
            ],
            entities=[],
        )
        actors, _ = resolve_actors([art_a, art_b])
        names_lower = [a.canonical_name.lower() for a in actors]
        assert any("demonstrators" in n for n in names_lower)

    def test_institutional_actor_no_ner_cross_article(self):
        """An institutional actor (e.g. Grethovi Police) appears without NER."""
        art_a = _article(
            "solvanic_news",
            svo=[
                _svo("Grethovi Police intervened in the protest.", "intervene",
                     ["Grethovi Police"], ["protest"]),
                _svo("Grethovi Police issued a warning.", "issue",
                     ["Grethovi Police"], ["warning"]),
            ],
            entities=[],
        )
        art_b = _article(
            "muraxian_times",
            svo=[
                _svo("Grethovi Police dispersed the crowd.", "disperse",
                     ["Grethovi Police"], ["crowd"]),
            ],
            entities=[],
        )
        actors, _ = resolve_actors([art_a, art_b])
        names_lower = [a.canonical_name.lower() for a in actors]
        assert any("grethovi police" in n for n in names_lower)


# ── Phase 2, Test 5: Passive role assignment ──────────────────────────────────


class TestPassiveRoleAssignment:
    """Active and passive constructions should produce semantically consistent roles."""

    def test_active_subject_is_agent(self):
        """In active sentence, grammatical subject = semantic agent."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Telphas Council voted.", "vote", ["Telphas Council"], [])],
            entities=[_em("Telphas Council", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Telphas Council approved.", "approve", ["Telphas Council"], [])],
            entities=[_em("Telphas Council", "ORG")],
        )
        _, stats = resolve_actors([art_a, art_b])
        telphas_a = next(
            (s for s in stats if "telphas" in s.canonical_name.lower()
             and s.article_id == "outlet_a"), None
        )
        assert telphas_a is not None
        assert telphas_a.agent_count >= 1
        assert telphas_a.patient_count == 0

    def test_passive_subject_is_semantic_patient(self):
        """In passive sentence, grammatical subject = semantic patient."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Solvanic protesters were detained.", "detain",
                      ["Solvanic protesters"], [], passive=True)],
            entities=[_em("Solvanic protesters", "NORP")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Solvanic protesters were arrested.", "arrest",
                      ["Solvanic protesters"], [], passive=True)],
            entities=[_em("Solvanic protesters", "NORP")],
        )
        _, stats = resolve_actors([art_a, art_b])
        solvanic_a = next(
            (s for s in stats if "solvanic" in s.canonical_name.lower()
             and s.article_id == "outlet_a"), None
        )
        assert solvanic_a is not None
        assert solvanic_a.patient_count >= 1
        assert solvanic_a.passive_patient_count >= 1

    def test_active_and_passive_same_participant_consistent_role(self):
        """Equivalent active and passive constructions produce consistent roles.

        Outlet A: active   → 'Muraxian Guards arrested Grethovi activists'
        Outlet B: passive  → 'Grethovi activists were arrested by Muraxian Guards'

        Expected: Muraxian Guards = agent in both. Grethovi activists = patient in both.
        """
        art_a = _article(
            "outlet_a",
            svo=[_svo(
                "Muraxian Guards arrested Grethovi activists.",
                "arrest",
                ["Muraxian Guards"], ["Grethovi activists"],
            )],
            entities=[
                _em("Muraxian Guards", "ORG"),
                _em("Grethovi activists", "NORP"),
            ],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo(
                "Grethovi activists were arrested by Muraxian Guards.",
                "arrest",
                ["Grethovi activists"], ["Muraxian Guards"],
                passive=True,
            )],
            entities=[
                _em("Muraxian Guards", "ORG"),
                _em("Grethovi activists", "NORP"),
            ],
        )
        _, stats = resolve_actors([art_a, art_b])

        muraxian_a = next(
            (s for s in stats if "muraxian" in s.canonical_name.lower()
             and s.article_id == "outlet_a"), None
        )
        muraxian_b = next(
            (s for s in stats if "muraxian" in s.canonical_name.lower()
             and s.article_id == "outlet_b"), None
        )
        grethovi_a = next(
            (s for s in stats if "grethovi" in s.canonical_name.lower()
             and s.article_id == "outlet_a"), None
        )
        grethovi_b = next(
            (s for s in stats if "grethovi" in s.canonical_name.lower()
             and s.article_id == "outlet_b"), None
        )

        assert muraxian_a is not None and muraxian_a.agent_count >= 1
        assert muraxian_b is not None and muraxian_b.agent_count >= 1  # by-phrase → agent
        assert grethovi_a is not None and grethovi_a.patient_count >= 1
        assert grethovi_b is not None and grethovi_b.patient_count >= 1  # passive subj → patient


# ── Phase 2, Test 6: Canonicalization preserves valid actors ──────────────────


class TestCanonicalizationPreservesActors:
    """Canonicalization must not destroy or incorrectly replace valid actors."""

    def test_two_distinct_actors_not_merged(self):
        """Canonicalization must not merge Grethovi Police and Solvanic Press."""
        art_a = _article(
            "outlet_a",
            svo=[
                _svo("Grethovi Police arrived.", "arrive", ["Grethovi Police"], []),
                _svo("Solvanic Press reported.", "report", ["Solvanic Press"], []),
            ],
            entities=[
                _em("Grethovi Police", "ORG"),
                _em("Solvanic Press", "ORG"),
            ],
        )
        art_b = _article(
            "outlet_b",
            svo=[
                _svo("Grethovi Police dispersed.", "disperse", ["Grethovi Police"], []),
                _svo("Solvanic Press published.", "publish", ["Solvanic Press"], []),
            ],
            entities=[
                _em("Grethovi Police", "ORG"),
                _em("Solvanic Press", "ORG"),
            ],
        )
        actors, _ = resolve_actors([art_a, art_b])
        police = [a for a in actors if "police" in a.canonical_name.lower()]
        press = [a for a in actors if "press" in a.canonical_name.lower()]
        assert len(police) >= 1
        assert len(press) >= 1
        # Should be separate actors
        assert police[0].canonical_name != press[0].canonical_name

    def test_same_actor_different_surface_forms_merged(self):
        """'Telphas Agency' and 'the Telphas Agency' should canonicalize to one actor."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Telphas Agency decided.", "decide", ["Telphas Agency"], [])],
            entities=[_em("Telphas Agency", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("The Telphas Agency confirmed.", "confirm",
                      ["The Telphas Agency"], [])],
            entities=[_em("the Telphas Agency", "ORG")],
        )
        actors, _ = resolve_actors([art_a, art_b])
        telphas = [a for a in actors if "telphas" in a.canonical_name.lower()]
        assert len(telphas) == 1
        assert len(telphas[0].article_ids) == 2

    def test_canonical_name_is_most_frequent_surface(self):
        """Canonical name should be the most frequently mentioned surface form."""
        art_a = _article(
            "outlet_a",
            svo=[
                _svo("Muraxian Bureau voted.", "vote", ["Muraxian Bureau"], []),
                _svo("Muraxian Bureau approved.", "approve", ["Muraxian Bureau"], []),
            ],
            entities=[_em("Muraxian Bureau", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("MB decided.", "decide", ["MB"], [])],
            entities=[_em("MB", "ORG")],
        )
        actors, _ = resolve_actors([art_a, art_b])
        # "Muraxian Bureau" has more mentions than "MB" → should be canonical
        muraxian = [a for a in actors if "muraxian" in a.canonical_name.lower()
                    or a.canonical_name == "MB"]
        if muraxian and muraxian[0].surface_mentions and len(muraxian[0].surface_mentions) > 1:
            # If merged, canonical should be the more frequent one
            assert "muraxian" in muraxian[0].canonical_name.lower()


# ── Phase 2, Test 7: Ranking preserves structurally important actors ──────────


class TestRankingCorrectness:
    """High-signal participants must rank above single-occurrence noise."""

    def test_cross_outlet_actor_ranks_above_single_outlet(self):
        """An actor in all 3 outlets must score higher than one in a single outlet."""
        art_a = _article(
            "outlet_a",
            svo=[
                _svo("Grethovi Guards arrested protesters.", "arrest",
                     ["Grethovi Guards"], ["protesters"]),
            ],
            entities=[_em("Grethovi Guards", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[
                _svo("Grethovi Guards detained activists.", "detain",
                     ["Grethovi Guards"], ["activists"]),
                _svo("Muraxian Consul attended.", "attend", ["Muraxian Consul"], []),
            ],
            entities=[
                _em("Grethovi Guards", "ORG"),
                _em("Muraxian Consul", "PERSON"),
            ],
        )
        art_c = _article(
            "outlet_c",
            svo=[
                _svo("Grethovi Guards dispersed the crowd.", "disperse",
                     ["Grethovi Guards"], ["crowd"]),
            ],
            entities=[_em("Grethovi Guards", "ORG")],
        )
        actors, _ = resolve_actors([art_a, art_b, art_c])
        names = [a.canonical_name.lower() for a in actors]
        grethovi_rank = next(
            (i for i, n in enumerate(names) if "grethovi guards" in n), None
        )
        muraxian_rank = next(
            (i for i, n in enumerate(names) if "muraxian consul" in n), None
        )
        if grethovi_rank is not None and muraxian_rank is not None:
            assert grethovi_rank < muraxian_rank

    def test_more_verb_diverse_actor_ranks_higher(self):
        """An actor with more distinct associated verbs must rank above one with fewer."""
        art_a = _article(
            "outlet_a",
            svo=[
                _svo("Telphas Authority arrested.", "arrest", ["Telphas Authority"], []),
                _svo("Telphas Authority warned.", "warn", ["Telphas Authority"], []),
                _svo("Telphas Authority investigated.", "investigate",
                     ["Telphas Authority"], []),
            ],
            entities=[_em("Telphas Authority", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[
                _svo("Telphas Authority acted.", "act", ["Telphas Authority"], []),
                _svo("Solvanic Observer reported.", "report", ["Solvanic Observer"], []),
            ],
            entities=[
                _em("Telphas Authority", "ORG"),
                _em("Solvanic Observer", "ORG"),
            ],
        )
        actors, _ = resolve_actors([art_a, art_b])
        names = [a.canonical_name.lower() for a in actors]
        telphas_rank = next(
            (i for i, n in enumerate(names) if "telphas" in n), None
        )
        solvanic_rank = next(
            (i for i, n in enumerate(names) if "solvanic observer" in n), None
        )
        if telphas_rank is not None and solvanic_rank is not None:
            assert telphas_rank <= solvanic_rank


# ── Phase 2, Test 8: Matrix integration ──────────────────────────────────────


class TestMatrixIntegrationStructural:
    """Entity × Outlet Matrix must not contain pure noise rows."""

    def test_matrix_excludes_event_noun_type(self):
        """EVENT_NOUN entities must not appear in the matrix."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Grethovi Bureau voted.", "vote", ["Grethovi Bureau"], [])],
            entities=[
                _em("Grethovi Bureau", "ORG"),
                _em("deliberation", "EVENT_NOUN"),
            ],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Grethovi Bureau approved.", "approve",
                      ["Grethovi Bureau"], [])],
            entities=[_em("Grethovi Bureau", "ORG")],
        )
        matrix = build_entity_outlet_matrix([art_a, art_b])
        assert "deliberation" not in matrix.entity_names

    def test_matrix_includes_svo_recovered_participant(self):
        """A participant recovered via SVO path (no NER) should appear in the matrix."""
        art_a = _article(
            "outlet_a",
            svo=[
                _svo("Muraxian Officers detained activists.", "detain",
                     ["Muraxian Officers"], ["activists"]),
                _svo("Muraxian Officers issued a statement.", "issue",
                     ["Muraxian Officers"], []),
            ],
            entities=[],  # No NER for Muraxian Officers
        )
        art_b = _article(
            "outlet_b",
            svo=[
                _svo("Muraxian Officers intervened.", "intervene",
                     ["Muraxian Officers"], []),
                _svo("Muraxian Officers warned demonstrators.", "warn",
                     ["Muraxian Officers"], ["demonstrators"]),
            ],
            entities=[],  # No NER for Muraxian Officers
        )
        matrix = build_entity_outlet_matrix([art_a, art_b])
        names_lower = [n.lower() for n in matrix.entity_names]
        assert any("muraxian officers" in n for n in names_lower)

    def test_matrix_ratio_consistency(self):
        """agent_ratio + patient_ratio must equal 1.0 when role_occurrence_count > 0."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Solvanic Bureau arrested Telphas activists.", "arrest",
                      ["Solvanic Bureau"], ["Telphas activists"])],
            entities=[
                _em("Solvanic Bureau", "ORG"),
                _em("Telphas activists", "NORP"),
            ],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Solvanic Bureau detained Telphas activists.", "detain",
                      ["Solvanic Bureau"], ["Telphas activists"])],
            entities=[
                _em("Solvanic Bureau", "ORG"),
                _em("Telphas activists", "NORP"),
            ],
        )
        matrix = build_entity_outlet_matrix([art_a, art_b])
        for p in matrix.profiles:
            if p.total_mentions > 0:
                assert abs(p.agent_ratio + p.patient_ratio - 1.0) < 1e-4, (
                    f"Ratio invariant violated for {p.entity_name} in {p.article_id}: "
                    f"ag={p.agent_ratio} pt={p.patient_ratio}"
                )


# ── Phase 2, Test 9: Chinese corpus SVO recovery ─────────────────────────────


class TestChineseSVORecovery:
    """SVO-based recovery must work for Chinese participants too."""

    def test_zh_institutional_actor_no_ner_recovered(self):
        """A Chinese institutional participant without NER must be recovered via SVO."""
        art_a = _article(
            "zh_outlet_a",
            svo=[
                _svo("格雷托維當局逮捕了三名示威者", "逮捕",
                     ["格雷托維當局"], ["三名示威者"]),
                _svo("格雷托維當局發表聲明", "發表",
                     ["格雷托維當局"], ["聲明"]),
            ],
            entities=[],  # No NER
        )
        art_b = _article(
            "zh_outlet_b",
            svo=[
                _svo("格雷托維當局驅散了人群", "驅散",
                     ["格雷托維當局"], ["人群"]),
            ],
            entities=[],  # No NER
        )
        actors, _ = resolve_actors([art_a, art_b])
        names = [a.canonical_name for a in actors]
        assert any("格雷托維當局" in n for n in names)

    def test_zh_passive_subject_becomes_patient(self):
        """Chinese passive: grammatical subject is logical patient."""
        art_a = _article(
            "zh_outlet_a",
            svo=[_svo("索爾瓦尼示威者被逮捕", "逮捕",
                      ["索爾瓦尼示威者"], [], passive=True)],
            entities=[_em("索爾瓦尼示威者", "NORP")],
        )
        art_b = _article(
            "zh_outlet_b",
            svo=[_svo("索爾瓦尼示威者遭拘留", "拘留",
                      ["索爾瓦尼示威者"], [], passive=True)],
            entities=[_em("索爾瓦尼示威者", "NORP")],
        )
        _, stats = resolve_actors([art_a, art_b])
        solvanic = [s for s in stats if "索爾瓦尼" in s.canonical_name]
        for s in solvanic:
            if s.role_occurrence_count > 0:
                assert s.patient_count >= 1
                assert s.passive_patient_count >= 1


# ── Phase 2, Test 10: SVO-NER merge correctness ───────────────────────────────


class TestSVONERMerge:
    """When NER and SVO paths both find the same participant, merge correctly."""

    def test_same_surface_ner_and_svo_merged_to_one_actor(self):
        """If 'Telphas Guards' appears both as NER entity and SVO subject, → 1 actor."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Telphas Guards arrested protesters.", "arrest",
                      ["Telphas Guards"], ["protesters"])],
            entities=[_em("Telphas Guards", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Telphas Guards responded.", "respond",
                      ["Telphas Guards"], [])],
            entities=[_em("Telphas Guards", "ORG")],
        )
        actors, _ = resolve_actors([art_a, art_b])
        telphas = [a for a in actors if "telphas guards" in a.canonical_name.lower()]
        # Must produce exactly 1 canonical actor for Telphas Guards
        assert len(telphas) == 1

    def test_ner_surface_takes_precedence_when_duplicated(self):
        """When both paths find the same surface, NER-backed ner_type is preferred."""
        art_a = _article(
            "outlet_a",
            svo=[_svo("Muraxian Council voted.", "vote", ["Muraxian Council"], [])],
            entities=[_em("Muraxian Council", "ORG")],
        )
        art_b = _article(
            "outlet_b",
            svo=[_svo("Muraxian Council approved.", "approve",
                      ["Muraxian Council"], [])],
            entities=[_em("Muraxian Council", "ORG")],
        )
        actors, _ = resolve_actors([art_a, art_b])
        muraxian = [a for a in actors if "muraxian council" in a.canonical_name.lower()]
        assert len(muraxian) == 1
        assert muraxian[0].entity_type == "ORG"


# ── Phase 2, Test 11: Structural span validation ──────────────────────────────


class TestStructuralSpanValidation:
    """_is_valid_svo_span must correctly gate structural quality."""

    @pytest.mark.parametrize("span,expected", [
        ("Grethovi Guards", True),
        ("格雷托維當局", True),
        ("demonstrators", True),
        ("ab", True),
        ("X", False),           # single char
        ("", False),            # empty
        ("  ", False),          # whitespace only
        (".", False),           # punctuation only
        ("a b c d e f g", False),  # 7 tokens
        ("A" * 31, False),      # over 30 chars single word
    ])
    def test_is_valid_svo_span_parametrized(self, span: str, expected: bool):
        assert _is_valid_svo_span(span) is expected
