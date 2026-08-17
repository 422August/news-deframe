"""Deterministic unit tests for the event console formatter (news_deframe.formatters.event_console).

Tests verify:
- Default concise event output (compact header, merged claim table, actor-oriented blocks, research notes)
- Merged claim coverage table with summary counts
- Absence of duplicated consensus section
- --details mode: shows source sentences and outlet coverage for manual inspection
- --verbose mode: shows centroid values, denominators, and full actor diagnostics
- Combined --details --verbose mode: includes similarity scores and full technical details
- Framing centroids hidden by default, visible in verbose
- Claim source sentences hidden by default, visible in details
- Actor-oriented framing rendering and top-N filtering
- Small corpus graceful handling
- Narrow terminal width robustness
- Deterministic sorting of claims and actors
"""
from __future__ import annotations

import io
from rich.console import Console

from news_deframe.analysis.schemas import (
    ClaimCluster,
    ClaimConsensus,
    ConsensusView,
    EntityOutletMatrix,
    EntityOutletProfile,
    EventAnalysis,
    FramingCluster,
    SourceSentence,
)
from news_deframe.formatters.event_console import render_event_analysis
from news_deframe.schemas import ParsedArticle


def _build_test_analysis(
    *,
    n_articles: int = 3,
    n_claims: int = 4,
    all_single_clusters: bool = True,
    n_actors: int = 6,
) -> EventAnalysis:
    """Construct a synthetic EventAnalysis for deterministic formatter testing."""
    article_ids = [f"outlet_{chr(97 + i)}" for i in range(n_articles)]  # outlet_a, outlet_b, ...

    # Claim clusters
    claim_clusters: list[ClaimCluster] = []
    consensus_claims: list[ClaimConsensus] = []

    # C01: all outlets
    c01_sources = [
        SourceSentence(article_id=aid, text=f"Source sentence for C01 in {aid}", similarity=0.95 - 0.05 * i)
        for i, aid in enumerate(article_ids)
    ]
    claim_clusters.append(
        ClaimCluster(
            cluster_id="C01",
            representative="Representative claim 1 shared by all.",
            sources=c01_sources,
            article_ids=list(article_ids),
            coverage_count=n_articles,
            total_articles=n_articles,
            coverage_ratio=1.0,
        )
    )
    consensus_claims.append(
        ClaimConsensus(
            cluster_id="C01",
            representative="Representative claim 1 shared by all.",
            coverage_count=n_articles,
            total_articles=n_articles,
            coverage_ratio=1.0,
            coverage_category="Widely shared",
            outlets_present=list(article_ids),
            outlets_absent=[],
        )
    )

    # C02: majority (outlets 0 and 1)
    if n_claims >= 2 and n_articles >= 3:
        present_c02 = article_ids[:2]
        absent_c02 = article_ids[2:]
        claim_clusters.append(
            ClaimCluster(
                cluster_id="C02",
                representative="Representative claim 2 majority.",
                sources=[
                    SourceSentence(article_id=aid, text=f"Source sentence for C02 in {aid}", similarity=0.88)
                    for aid in present_c02
                ],
                article_ids=present_c02,
                coverage_count=len(present_c02),
                total_articles=n_articles,
                coverage_ratio=round(len(present_c02) / n_articles, 2),
            )
        )
        consensus_claims.append(
            ClaimConsensus(
                cluster_id="C02",
                representative="Representative claim 2 majority.",
                coverage_count=len(present_c02),
                total_articles=n_articles,
                coverage_ratio=round(len(present_c02) / n_articles, 2),
                coverage_category="Commonly reported",
                outlets_present=present_c02,
                outlets_absent=absent_c02,
            )
        )

    # C03: single outlet (outlet_a)
    if n_claims >= 3:
        present_c03 = [article_ids[0]]
        absent_c03 = article_ids[1:]
        claim_clusters.append(
            ClaimCluster(
                cluster_id="C03",
                representative="Representative claim 3 single.",
                sources=[
                    SourceSentence(article_id=article_ids[0], text="Source for C03", similarity=1.0)
                ],
                article_ids=present_c03,
                coverage_count=1,
                total_articles=n_articles,
                coverage_ratio=round(1 / n_articles, 2),
            )
        )
        consensus_claims.append(
            ClaimConsensus(
                cluster_id="C03",
                representative="Representative claim 3 single.",
                coverage_count=1,
                total_articles=n_articles,
                coverage_ratio=round(1 / n_articles, 2),
                coverage_category="Minority coverage",
                outlets_present=present_c03,
                outlets_absent=absent_c03,
            )
        )

    # Actor framing
    entity_names = [f"Actor_{i + 1}" for i in range(n_actors)]
    profiles: list[EntityOutletProfile] = []
    for ename in entity_names:
        for aid in article_ids:
            profiles.append(
                EntityOutletProfile(
                    entity_name=ename,
                    article_id=aid,
                    total_mentions=4,
                    subject_count=3,
                    object_count=1,
                    passive_count=1,
                    agent_ratio=0.75,
                    patient_ratio=0.25,
                    passive_ratio=1.0,
                    modifiers=["important"],
                    associated_verbs=["announced", "investigated"],
                )
            )

    matrix = EntityOutletMatrix(
        entity_names=entity_names,
        article_ids=article_ids,
        profiles=profiles,
    )

    # Framing clusters
    framing_clusters: list[FramingCluster] = []
    if all_single_clusters:
        for i, aid in enumerate(article_ids):
            framing_clusters.append(
                FramingCluster(
                    cluster_id=i + 1,
                    label=f"Framing Cluster {i + 1}",
                    article_ids=[aid],
                    centroid_description={
                        "passive_ratio": 0.12,
                        "mean_agent_ratio": 0.45,
                        "mean_patient_ratio": 0.25,
                        "entity_count_norm": 0.80,
                        "sentence_count_norm": 0.60,
                    },
                )
            )
    else:
        framing_clusters.append(
            FramingCluster(
                cluster_id=1,
                label="Framing Cluster 1",
                article_ids=article_ids[:2],
                centroid_description={"passive_ratio": 0.10, "mean_agent_ratio": 0.50},
            )
        )
        framing_clusters.append(
            FramingCluster(
                cluster_id=2,
                label="Framing Cluster 2",
                article_ids=article_ids[2:],
                centroid_description={"passive_ratio": 0.20, "mean_agent_ratio": 0.30},
            )
        )

    return EventAnalysis(
        event_id="test_event",
        articles=[ParsedArticle(article_id=aid) for aid in article_ids],
        article_ids=article_ids,
        claim_clusters=claim_clusters,
        entity_outlet_matrix=matrix,
        framing_clusters=framing_clusters,
        consensus_view=ConsensusView(total_articles=n_articles, claims=consensus_claims),
    )


def _render_to_string(analysis: EventAnalysis, *, details: bool = False, verbose: bool = False, width: int = 100) -> str:
    """Helper to render EventAnalysis to plain string using Rich Console."""
    string_io = io.StringIO()
    console = Console(file=string_io, width=width, color_system=None, force_terminal=False)
    render_event_analysis(analysis, details=details, verbose=verbose, console=console)
    return string_io.getvalue()


class TestEventConsoleFormatter:
    def test_default_concise_event_output(self):
        """Default output must be concise, research-oriented, and omit technical engineering diagnostics."""
        analysis = _build_test_analysis()
        out = _render_to_string(analysis, details=False, verbose=False)

        # 1. Compact header
        assert "Event:" in out
        assert "test_event" in out
        assert "Articles: 3" in out
        assert "Claim clusters: 3" in out
        assert "Framing clusters: 3" in out

        # 2. Merged Claim Coverage
        assert "Claim Coverage" in out
        assert "Shared by all outlets:     1" in out
        assert "Shared by majority:        1" in out
        assert "Single-outlet claims:      1" in out
        assert "C01" in out
        assert "C02" in out
        assert "C03" in out
        assert "Widely shared" in out
        assert "Commonly reported" in out
        assert "Single-outlet" in out

        # 3. Absence of duplicated consensus section
        assert "Consensus / Outliers" not in out

        # 4. Actor Framing
        assert "Actor Framing by Outlet" in out
        assert "Ratios are calculated from occurrences with an identifiable agent/patient role." in out
        assert "Actor: Actor_1" in out
        assert "Agent" in out
        assert "Patient" in out
        assert "Role observations" in out
        assert "Associated actions:" in out
        assert "announced, investigated" in out
        # Top 5 limit applied when 6 actors exist
        assert "Showing top 5 of 6 actors" in out
        assert "Actor_5" in out
        assert "Actor: Actor_6" not in out  # 6th actor omitted in default view

        # 5. Framing Clusters (Centroid hidden by default)
        assert "Framing Clusters" in out
        assert "Note: Each article forms a separate structural cluster in this corpus." in out
        assert "• Framing Cluster 1: outlet_a" in out
        assert "Centroid —" not in out
        assert "passive_ratio:" not in out

        # 6. Detailed claim evidence hidden by default
        assert "Detailed Claim Evidence" not in out
        assert "Source sentences:" not in out
        assert "sim=" not in out

        # 7. Research interpretation notes
        assert "Research Interpretation Notes" in out
        assert "Claim coverage reflects reporting frequency across outlets" in out
        assert "Absence of a claim indicates a reporting difference" in out

    def test_details_mode(self):
        """--details displays claim-level evidence and source sentences without technical similarity scores."""
        analysis = _build_test_analysis()
        out = _render_to_string(analysis, details=True, verbose=False)

        assert "Detailed Claim Evidence" in out
        assert "Representative:" in out
        assert "Coverage: 3/3" in out
        assert "Present outlets: outlet_a, outlet_b, outlet_c" in out
        assert "Absent outlets:  —" in out
        assert "Source sentences:" in out
        assert "[outlet_a] Source sentence for C01 in outlet_a" in out
        # In details mode alone, technical similarity scores are hidden
        assert "(sim=" not in out
        # Centroid values still hidden
        assert "Centroid —" not in out

    def test_verbose_mode(self):
        """--verbose displays centroids, exact denominators, all actors, and passive patient details."""
        analysis = _build_test_analysis()
        out = _render_to_string(analysis, details=False, verbose=True)

        # Centroid values visible
        assert "Centroid —" in out
        assert "passive_ratio: 0.12" in out
        assert "mean_agent_ratio: 0.45" in out

        # Exact denominator note visible
        assert "Exact denominator: role_occurrence_count" in out

        # Passive Pt column visible in actor table
        assert "Passive Pt" in out

        # All 6 actors shown (no top-5 capping in verbose)
        assert "Actor: Actor_6" in out

        # Evaluative modifiers shown in verbose
        assert "Evaluative modifiers:" in out

        # Since details=False, Detailed Claim Evidence is omitted
        assert "Detailed Claim Evidence" not in out

    def test_details_and_verbose_combined(self):
        """--details --verbose shows detailed claims with similarity scores and all verbose diagnostics."""
        analysis = _build_test_analysis()
        out = _render_to_string(analysis, details=True, verbose=True)

        assert "Detailed Claim Evidence" in out
        assert "Centroid —" in out
        assert "(sim=0.95)" in out
        assert "Actor: Actor_6" in out

    def test_multi_article_framing_clusters_rendering(self):
        """Multi-article clusters render membership without single-cluster note."""
        analysis = _build_test_analysis(all_single_clusters=False)
        out = _render_to_string(analysis, details=False, verbose=False)

        assert "• Framing Cluster 1: outlet_a, outlet_b" in out
        assert "• Framing Cluster 2: outlet_c" in out
        assert "Note: Each article forms a separate structural cluster" not in out

    def test_narrow_terminal_width(self):
        """Formatter renders without error on narrow terminals (e.g. 60 cols)."""
        analysis = _build_test_analysis()
        out = _render_to_string(analysis, details=True, verbose=True, width=60)
        assert "Event:" in out
        assert "Claim Coverage" in out
        assert "Detailed Claim Evidence" in out

    def test_empty_analysis_graceful_handling(self):
        """Empty analysis prints empty notices without raising exceptions."""
        empty_analysis = EventAnalysis(
            event_id="empty_event",
            articles=[],
            article_ids=[],
            claim_clusters=[],
            entity_outlet_matrix=EntityOutletMatrix(entity_names=[], article_ids=[], profiles=[]),
            framing_clusters=[],
            consensus_view=ConsensusView(total_articles=0, claims=[]),
        )
        out = _render_to_string(empty_analysis, details=True, verbose=False)
        assert "Event:" in out
        assert "No claim clusters found." in out
        assert "No actor framing data." in out
        assert "No framing clusters found." in out
        assert "No detailed claim data." in out
