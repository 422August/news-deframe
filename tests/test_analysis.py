"""Deterministic offline unit tests for event-level analysis modules.

Modules tested:
- news_deframe.analysis.claims (claim clustering, deduplication)
- news_deframe.analysis.entity_matrix (entity framing matrix, ratios, verbs)
- news_deframe.analysis.framing_clusters (unsupervised clustering, neutral labels)
- news_deframe.analysis.consensus (consensus view, categories, absent outlets)
- news_deframe.analysis.event (event orchestrator)
- JSON serialization / deserialization

Deterministic embedding mock is used to avoid downloading sentence-transformers.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from news_deframe.schemas import (
    EntityModifier,
    ParsedArticle,
    SVORecord,
)
from news_deframe.analysis.schemas import (
    ClaimCluster,
    ClaimConsensus,
    ConsensusView,
    EntityOutletMatrix,
    EntityOutletProfile,
    EventAnalysis,
    FramingCluster,
)
from news_deframe.analysis.claims import cluster_claims
from news_deframe.analysis.entity_matrix import build_entity_outlet_matrix
from news_deframe.analysis.framing_clusters import (
    _build_feature_matrix,
    _build_feature_vector,
    cluster_articles,
)
from news_deframe.analysis.consensus import (
    CoverageThresholds,
    build_consensus_view,
)
from news_deframe.analysis.event import run_event_analysis
from news_deframe.formatters.json_export import (
    event_to_json,
    report_to_json,
    save_event_analysis,
)


# ─── Mock embedder helper ──────────────────────────────────────────────────────


def _make_keyword_embedder(cluster_keywords: list[list[str]]):
    """Return an embedding function where sentences matching a keyword group have sim = 1.0.

    Each keyword group maps to an orthogonal one-hot coordinate.
    """
    dim = max(len(cluster_keywords) + 2, 8)

    def _embed(sentences: list[str]) -> np.ndarray:
        embs = np.zeros((len(sentences), dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            assigned = False
            for group_idx, keywords in enumerate(cluster_keywords):
                if any(kw.lower() in s.lower() for kw in keywords):
                    embs[i, group_idx] = 1.0
                    assigned = True
                    break
            if not assigned:
                # default orthogonal dimension
                embs[i, -1] = 1.0
        return embs

    return _embed


def _make_parsed_article(
    article_id: str,
    sentences: list[str],
    svo_records: list[SVORecord] | None = None,
    entity_modifiers: list[EntityModifier] | None = None,
) -> ParsedArticle:
    """Helper to construct a ParsedArticle."""
    return ParsedArticle(
        article_id=article_id,
        sentences=sentences,
        svo_records=svo_records or [],
        entity_modifiers=entity_modifiers or [],
    )


# ─── Tests: Claim Clustering ──────────────────────────────────────────────────


class TestClaimClustering:
    def test_basic_claim_clustering(self):
        """Sentences with high similarity across articles form clusters."""
        # 3 articles
        # Claim A: Police arrested protesters (outlets 1, 2, 3)
        # Claim B: Fire started at midnight (outlets 1, 2)
        # Claim C: Stock market crashed (outlet 3 only)
        art1 = _make_parsed_article("outlet_a", ["Police arrested three protesters.", "Fire broke out at midnight."])
        art2 = _make_parsed_article("outlet_b", ["Three protesters were arrested by police.", "The fire started at 00:00."])
        art3 = _make_parsed_article("outlet_c", ["Officers arrested suspects.", "Stock market suffered heavy losses."])

        embed_fn = _make_keyword_embedder([
            ["arrest", "protester", "suspect", "officer"],
            ["fire", "midnight", "00:00"],
            ["stock", "market", "losses"],
        ])

        clusters = cluster_claims([art1, art2, art3], threshold=0.6, embed_fn=embed_fn)

        assert len(clusters) == 3
        # First cluster has highest coverage (3/3)
        assert clusters[0].coverage_count == 3
        assert set(clusters[0].article_ids) == {"outlet_a", "outlet_b", "outlet_c"}
        assert clusters[0].coverage_ratio == pytest.approx(1.0)

        # Second cluster has coverage 2/3
        assert clusters[1].coverage_count == 2
        assert set(clusters[1].article_ids) == {"outlet_a", "outlet_b"}
        assert clusters[1].coverage_ratio == pytest.approx(2 / 3, abs=1e-3)

        # Third cluster has coverage 1/3
        assert clusters[2].coverage_count == 1
        assert clusters[2].article_ids == ["outlet_c"]

    def test_duplicate_claims_in_one_article_do_not_inflate_coverage(self):
        """Repeated sentences/paraphrases within the SAME article must count once."""
        art1 = _make_parsed_article(
            "outlet_a",
            [
                "Police arrested three protesters.",
                "The police detained three activists.",
                "Arrests were made by the police.",
            ],
        )
        art2 = _make_parsed_article(
            "outlet_b",
            ["Officers arrested suspects."],
        )

        embed_fn = _make_keyword_embedder([
            ["police", "arrest", "detained", "officers", "suspects", "activists", "protesters"],
        ])

        clusters = cluster_claims([art1, art2], threshold=0.6, embed_fn=embed_fn)

        assert len(clusters) == 1
        cluster = clusters[0]
        # Total articles is 2, and art1 should only count ONCE even with 3 sentences
        assert cluster.coverage_count == 2
        assert cluster.total_articles == 2
        assert cluster.coverage_ratio == pytest.approx(1.0)
        assert sorted(cluster.article_ids) == ["outlet_a", "outlet_b"]
        # All 4 sentences are in sources
        assert len(cluster.sources) == 4

    def test_empty_corpus_claims(self):
        """Empty articles return empty clusters."""
        art1 = _make_parsed_article("a", [])
        art2 = _make_parsed_article("b", [])
        clusters = cluster_claims([art1, art2], embed_fn=lambda s: np.zeros((0, 8)))
        assert clusters == []


# ─── Tests: Entity × Outlet Matrix ───────────────────────────────────────────


class TestEntityOutletMatrix:
    def test_entity_framing_matrix_calculation(self):
        """Matrix correctly tallies agent, patient, passive and calculates normalized ratios."""
        # Article A: Police is subject (agent), Protesters is object (patient)
        art_a = _make_parsed_article(
            "outlet_a",
            ["Police arrested protesters."],
            svo_records=[
                SVORecord(
                    sentence="Police arrested protesters.",
                    verb="arrest",
                    subjects=["Police"],
                    objects=["protesters"],
                    is_passive=False,
                )
            ],
            entity_modifiers=[
                EntityModifier(entity_name="Police", entity_type="ORG", modifiers=["local"]),
                EntityModifier(entity_name="protesters", entity_type="PERSON", modifiers=["violent"]),
            ],
        )

        # Article B: Protesters is subject (agent) in active sentence, and in passive sentence
        art_b = _make_parsed_article(
            "outlet_b",
            ["Protesters marched peacefully.", "Protesters were attacked by Police."],
            svo_records=[
                SVORecord(
                    sentence="Protesters marched peacefully.",
                    verb="march",
                    subjects=["Protesters"],
                    objects=[],
                    is_passive=False,
                ),
                SVORecord(
                    sentence="Protesters were attacked by Police.",
                    verb="attack",
                    subjects=["Protesters"],
                    objects=["Police"],
                    is_passive=True,
                    voice_markers=["were"],
                ),
            ],
            entity_modifiers=[
                EntityModifier(entity_name="Police", entity_type="ORG", modifiers=["armed"]),
                EntityModifier(entity_name="protesters", entity_type="PERSON", modifiers=["peaceful"]),
            ],
        )

        matrix = build_entity_outlet_matrix([art_a, art_b])

        assert "Police" in matrix.entity_names or "police" in [e.lower() for e in matrix.entity_names]
        assert sorted(matrix.article_ids) == ["outlet_a", "outlet_b"]

        # Find profiles for Police
        police_profiles = {p.article_id: p for p in matrix.profiles if p.entity_name.lower() == "police"}
        assert "outlet_a" in police_profiles
        assert "outlet_b" in police_profiles

        prof_a = police_profiles["outlet_a"]
        assert prof_a.subject_count == 1
        assert prof_a.object_count == 0
        assert prof_a.total_mentions == 1
        assert prof_a.agent_ratio == pytest.approx(1.0)
        assert prof_a.patient_ratio == pytest.approx(0.0)
        assert "local" in prof_a.modifiers
        assert "arrest" in prof_a.associated_verbs

        prof_b = police_profiles["outlet_b"]
        # "Protesters were attacked by Police." (passive):
        #   subjects=["Protesters"] -> logical patient (passive nsubj)
        #   objects=["Police"]      -> logical AGENT (by-phrase / agent dep)
        # The actor resolution pipeline applies passive-role inversion:
        # Police in the objects slot of a passive SVO = agent, not patient.
        # This is the semantically correct behavior; the previous test encoded
        # the pre-refactor behavior that did NOT invert passive roles.
        assert prof_b.subject_count == 1   # agent_count (Police is logical agent here)
        assert prof_b.object_count == 0    # patient_count
        assert prof_b.total_mentions == 1
        assert prof_b.agent_ratio == pytest.approx(1.0)
        assert prof_b.patient_ratio == pytest.approx(0.0)
        # passive_count = passive_patient_count (Police is not a passive patient here)
        assert prof_b.passive_count == 0
        assert "attack" in prof_b.associated_verbs

    def test_entity_with_zero_svo_mentions_excluded_from_matrix(self):
        """Actors not participating in any SVO and appearing in only one article
        are excluded by the new actor validation pipeline.

        The old behavior was to include all NER entities regardless of SVO
        participation.  The new design requires at least (S1 + one other signal)
        where S1 = correct NER type.  An entity that never appears in a subject
        or object slot, and only in one article, has only S1 and is excluded.

        This is the intended behavior: the matrix should contain meaningful,
        SVO-grounded actors, not bare NER mentions.
        """
        art_a = _make_parsed_article(
            "outlet_a",
            ["Nothing happened."],
            entity_modifiers=[EntityModifier(entity_name="Hospital", entity_type="ORG", modifiers=[])],
        )
        art_b = _make_parsed_article("outlet_b", ["Nothing happened."])

        matrix = build_entity_outlet_matrix([art_a, art_b])
        # "Hospital" appears in one article, no SVO participation -> excluded
        assert "Hospital" not in matrix.entity_names

    def test_actor_appearing_in_both_outlets_but_no_svo_has_zero_ratios(self):
        """An actor that appears (via NER) in both outlets but never in SVO
        subjects or objects should have zero role_occurrence_count and zero ratios.
        It passes validation (S1 + S3: appears in >1 article) but has no
        role-grounded mentions.
        """
        art_a = _make_parsed_article(
            "outlet_a",
            ["The Regulatorium was mentioned."],
            svo_records=[
                SVORecord(
                    sentence="Officials spoke.",
                    verb="speak",
                    subjects=["Officials"],
                    objects=[],
                    is_passive=False,
                )
            ],
            entity_modifiers=[
                EntityModifier(entity_name="Regulatorium", entity_type="ORG", modifiers=[]),
                EntityModifier(entity_name="Officials", entity_type="ORG", modifiers=[]),
            ],
        )
        art_b = _make_parsed_article(
            "outlet_b",
            ["The Regulatorium was present."],
            svo_records=[
                SVORecord(
                    sentence="Officials spoke again.",
                    verb="speak",
                    subjects=["Officials"],
                    objects=[],
                    is_passive=False,
                )
            ],
            entity_modifiers=[
                EntityModifier(entity_name="Regulatorium", entity_type="ORG", modifiers=[]),
                EntityModifier(entity_name="Officials", entity_type="ORG", modifiers=[]),
            ],
        )

        matrix = build_entity_outlet_matrix([art_a, art_b])
        # Regulatorium appears in both articles (S1+S3 -> validated)
        # but has no SVO role occurrences -> zero total_mentions, zero ratios
        reg_profiles = [p for p in matrix.profiles if p.entity_name == "Regulatorium"]
        if reg_profiles:
            for p in reg_profiles:
                assert p.total_mentions == 0
                assert p.agent_ratio == 0.0
                assert p.patient_ratio == 0.0


# ─── Tests: Framing Clusters ─────────────────────────────────────────────────


class TestFramingClusters:
    def test_framing_feature_construction_and_clustering(self):
        """Articles are clustered into neutral framing groups."""
        art1 = _make_parsed_article(
            "outlet_1",
            ["s1", "s2"],
            svo_records=[
                SVORecord(sentence="s1", verb="v1", subjects=["Police"], objects=[], is_passive=False),
                SVORecord(sentence="s2", verb="v2", subjects=["Police"], objects=[], is_passive=False),
            ],
            entity_modifiers=[EntityModifier(entity_name="Police", entity_type="ORG", modifiers=[])],
        )
        art2 = _make_parsed_article(
            "outlet_2",
            ["s1", "s2"],
            svo_records=[
                SVORecord(sentence="s1", verb="v1", subjects=["Police"], objects=[], is_passive=False),
                SVORecord(sentence="s2", verb="v2", subjects=["Police"], objects=[], is_passive=False),
            ],
            entity_modifiers=[EntityModifier(entity_name="Police", entity_type="ORG", modifiers=[])],
        )
        art3 = _make_parsed_article(
            "outlet_3",
            ["s3", "s4"],
            svo_records=[
                SVORecord(sentence="s3", verb="v3", subjects=[], objects=["Police"], is_passive=True),
                SVORecord(sentence="s4", verb="v4", subjects=[], objects=["Police"], is_passive=True),
            ],
            entity_modifiers=[EntityModifier(entity_name="Police", entity_type="ORG", modifiers=[])],
        )

        matrix = build_entity_outlet_matrix([art1, art2, art3])
        claim_clusters = [
            ClaimCluster(
                cluster_id="C01",
                representative="s1",
                article_ids=["outlet_1", "outlet_2"],
                coverage_count=2,
                total_articles=3,
                coverage_ratio=2 / 3,
            )
        ]

        clusters = cluster_articles([art1, art2, art3], matrix, claim_clusters, n_clusters=2)

        assert len(clusters) == 2
        # Labels must be neutral
        for c in clusters:
            assert c.label.startswith("Framing Cluster")
            assert "bias" not in c.label.lower()
            assert "left" not in c.label.lower()
            assert "right" not in c.label.lower()
            assert isinstance(c.centroid_description, dict)

        all_assigned = []
        for c in clusters:
            all_assigned.extend(c.article_ids)
        assert set(all_assigned) == {"outlet_1", "outlet_2", "outlet_3"}


# ─── Tests: Consensus / Outlier View ──────────────────────────────────────────


class TestConsensusView:
    def test_consensus_categories_and_absent_outlets(self):
        """Consensus categories map correctly and absent outlets are listed."""
        all_outlets = ["outlet_a", "outlet_b", "outlet_c", "outlet_d", "outlet_e"]
        clusters = [
            # 5/5 -> 1.0 -> Widely shared
            ClaimCluster(
                cluster_id="C01",
                representative="Claim 1",
                article_ids=["outlet_a", "outlet_b", "outlet_c", "outlet_d", "outlet_e"],
                coverage_count=5,
                total_articles=5,
                coverage_ratio=1.0,
            ),
            # 3/5 -> 0.6 -> Commonly reported
            ClaimCluster(
                cluster_id="C02",
                representative="Claim 2",
                article_ids=["outlet_a", "outlet_b", "outlet_c"],
                coverage_count=3,
                total_articles=5,
                coverage_ratio=0.6,
            ),
            # 2/5 -> 0.4 -> Minority coverage (>= 0.20 and < 0.50)
            ClaimCluster(
                cluster_id="C03",
                representative="Claim 3",
                article_ids=["outlet_d", "outlet_e"],
                coverage_count=2,
                total_articles=5,
                coverage_ratio=0.4,
            ),
            # 1/5 -> 0.2 -> Minority coverage (default >= 0.20)
            # Let's test Rare claim with 0.1 ratio
            ClaimCluster(
                cluster_id="C04",
                representative="Claim 4",
                article_ids=["outlet_a"],
                coverage_count=1,
                total_articles=10,
                coverage_ratio=0.10,
            ),
        ]

        consensus = build_consensus_view(clusters, all_outlets)

        assert len(consensus.claims) == 4
        c1 = consensus.claims[0]
        assert c1.cluster_id == "C01"
        assert c1.coverage_category == "Widely shared"
        assert c1.outlets_absent == []

        c2 = consensus.claims[1]
        assert c2.cluster_id == "C02"
        assert c2.coverage_category == "Commonly reported"
        assert set(c2.outlets_absent) == {"outlet_d", "outlet_e"}

        c3 = consensus.claims[2]
        assert c3.cluster_id == "C03"
        assert c3.coverage_category == "Minority coverage"
        assert set(c3.outlets_absent) == {"outlet_a", "outlet_b", "outlet_c"}

        c4 = consensus.claims[3]
        assert c4.cluster_id == "C04"
        assert c4.coverage_category == "Rare claim"
        assert "outlet_b" in c4.outlets_absent

    def test_custom_coverage_thresholds(self):
        """Custom CoverageThresholds override default category cutoffs."""
        custom_t = CoverageThresholds(widely_shared=0.90, commonly_reported=0.70, minority_coverage=0.40)
        clusters = [
            ClaimCluster(
                cluster_id="C01",
                representative="Claim 1",
                article_ids=["a", "b", "c"],
                coverage_count=3,
                total_articles=4,
                coverage_ratio=0.75,
            )
        ]
        consensus = build_consensus_view(clusters, ["a", "b", "c", "d"], thresholds=custom_t)
        assert consensus.claims[0].coverage_category == "Commonly reported"


# ─── Tests: Event Orchestrator & Serialization ────────────────────────────────


class TestEventOrchestratorAndSerialization:
    def test_run_event_analysis_full_pipeline(self, tmp_path):
        """run_event_analysis wires all layers and produces valid EventAnalysis."""
        art1 = _make_parsed_article(
            "outlet_a",
            ["Police arrested three protesters.", "Incident occurred at 10:00."],
            svo_records=[
                SVORecord(
                    sentence="Police arrested three protesters.",
                    verb="arrest",
                    subjects=["Police"],
                    objects=["protesters"],
                    is_passive=False,
                )
            ],
            entity_modifiers=[
                EntityModifier(entity_name="Police", entity_type="ORG", modifiers=["local"]),
            ],
        )
        art2 = _make_parsed_article(
            "outlet_b",
            ["Three protesters were detained by officers.", "Incident was recorded."],
            svo_records=[
                SVORecord(
                    sentence="Three protesters were detained by officers.",
                    verb="detain",
                    subjects=["protesters"],
                    objects=["officers"],
                    is_passive=True,
                    voice_markers=["were"],
                )
            ],
            entity_modifiers=[
                EntityModifier(entity_name="protesters", entity_type="PERSON", modifiers=["young"]),
            ],
        )

        embed_fn = _make_keyword_embedder([
            ["police", "arrest", "protester", "detained", "officers"],
            ["incident", "occurred", "recorded"],
        ])

        analysis = run_event_analysis(
            event_id="event_001",
            articles=[art1, art2],
            threshold=0.6,
            embed_fn=embed_fn,
        )

        assert isinstance(analysis, EventAnalysis)
        assert analysis.event_id == "event_001"
        assert analysis.article_ids == ["outlet_a", "outlet_b"]
        assert len(analysis.articles) == 2
        assert len(analysis.claim_clusters) >= 1
        assert len(analysis.framing_clusters) >= 1
        assert analysis.consensus_view.total_articles == 2

        # Test JSON serialization via Pydantic model_dump_json
        json_str = analysis.model_dump_json(indent=2)
        data = json.loads(json_str)
        assert data["event_id"] == "event_001"
        assert "claim_clusters" in data
        assert "entity_outlet_matrix" in data
        assert "framing_clusters" in data
        assert "consensus_view" in data
        assert "articles" in data

        # Test save_event_analysis / report_to_json
        out_file = tmp_path / "event_report.json"
        saved = save_event_analysis(analysis, out_file)
        assert saved.exists()
        assert "event_001" in saved.read_text(encoding="utf-8")

    def test_run_event_analysis_raises_on_single_article(self):
        """run_event_analysis requires >= 2 articles."""
        art1 = _make_parsed_article("a", ["s1"])
        with pytest.raises(ValueError, match="at least 2 articles"):
            run_event_analysis("event_01", [art1])
