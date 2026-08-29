"""Top-level event analysis orchestrator.

This module wires the individual analysis layers together into a single
:func:`run_event_analysis` call that accepts a list of
:class:`~news_deframe.schemas.ParsedArticle` objects and returns a fully
populated :class:`~news_deframe.analysis.schemas.EventAnalysis`.

The pipeline order is:

1.  Claim clustering       (``analysis.claims``)
2.  Entity \u00d7 outlet matrix (``analysis.entity_matrix``)
3.  Framing clusters       (``analysis.framing_clusters``)
4.  Consensus view         (``analysis.consensus``)
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from news_deframe.schemas import ParsedArticle
from news_deframe.analysis.claims import cluster_claims
from news_deframe.analysis.entity_matrix import build_entity_outlet_matrix
from news_deframe.analysis.framing_clusters import cluster_articles
from news_deframe.analysis.consensus import build_consensus_view, CoverageThresholds
from news_deframe.analysis.schemas import EventAnalysis


def run_event_analysis(
    event_id: str,
    articles: list[ParsedArticle],
    *,
    threshold: float = 0.60,
    n_framing_clusters: int | None = None,
    coverage_thresholds: CoverageThresholds | None = None,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> EventAnalysis:
    """Run the complete event-level analysis pipeline.

    Parameters
    ----------
    event_id:
        Identifier for this event corpus (e.g. the folder name).
    articles:
        Parsed articles.  Must contain at least 2 items.
    threshold:
        Cosine similarity threshold for claim clustering.
    n_framing_clusters:
        Number of framing clusters (defaults to ``min(3, len(articles))``).
    coverage_thresholds:
        Custom coverage category thresholds for the consensus view.
    embed_fn:
        Optional embedding function override for testing.

    Returns
    -------
    EventAnalysis
    """
    if len(articles) < 2:
        raise ValueError(
            f"run_event_analysis requires at least 2 articles; got {len(articles)}."
        )

    article_ids = [a.article_id for a in articles]

    # 1. Claim clustering
    claims = cluster_claims(articles, threshold=threshold, embed_fn=embed_fn)

    # 2. Entity × outlet matrix
    matrix = build_entity_outlet_matrix(articles)

    # 3. Framing clusters
    framing = cluster_articles(
        articles,
        matrix,
        claims,
        n_clusters=n_framing_clusters,
    )

    # 4. Consensus view
    consensus = build_consensus_view(claims, article_ids, thresholds=coverage_thresholds)

    return EventAnalysis(
        event_id=event_id,
        articles=articles,
        article_ids=article_ids,
        claim_clusters=claims,
        entity_outlet_matrix=matrix,
        framing_clusters=framing,
        consensus_view=consensus,
    )
