"""Consensus / outlier view for claim clusters.

This module classifies each claim cluster by its *coverage frequency* using
configurable thresholds.  The categories are purely descriptive:

+--------------------+----------------------------+
| Category           | Default coverage ratio     |
+====================+============================+
| Widely shared      | >= 0.80                    |
+--------------------+----------------------------+
| Commonly reported  | >= 0.50                    |
+--------------------+----------------------------+
| Minority coverage  | >= 0.20                    |
+--------------------+----------------------------+
| Rare claim         | < 0.20                     |
+--------------------+----------------------------+

Absence of a claim from an outlet is reported as a *coverage difference* \u2014
never as intentional omission, falsehood, or bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from news_deframe.analysis.schemas import (
    ClaimCluster,
    ClaimConsensus,
    ConsensusView,
)


@dataclass(frozen=True)
class CoverageThresholds:
    """Configurable thresholds for coverage frequency categories.

    Attributes
    ----------
    widely_shared:
        Minimum ratio for the "Widely shared" category.
    commonly_reported:
        Minimum ratio for the "Commonly reported" category.
    minority_coverage:
        Minimum ratio for the "Minority coverage" category.
    Below *minority_coverage* is classified as "Rare claim".
    """

    widely_shared: float = 0.80
    commonly_reported: float = 0.50
    minority_coverage: float = 0.20


_DEFAULT_THRESHOLDS = CoverageThresholds()


def _coverage_category(ratio: float, thresholds: CoverageThresholds) -> str:
    """Map a coverage ratio to a descriptive frequency category."""
    if ratio >= thresholds.widely_shared:
        return "Widely shared"
    if ratio >= thresholds.commonly_reported:
        return "Commonly reported"
    if ratio >= thresholds.minority_coverage:
        return "Minority coverage"
    return "Rare claim"


def build_consensus_view(
    clusters: list[ClaimCluster],
    all_article_ids: Sequence[str],
    *,
    thresholds: CoverageThresholds | None = None,
) -> ConsensusView:
    """Build a consensus / outlier view from claim clusters.

    Parameters
    ----------
    clusters:
        Claim clusters produced by :func:`~news_deframe.analysis.claims.cluster_claims`.
    all_article_ids:
        The complete ordered list of article IDs in the event corpus.  Used to
        determine which outlets are absent from each claim cluster.
    thresholds:
        Optional custom coverage thresholds.

    Returns
    -------
    ConsensusView
        Sorted by descending coverage ratio, then by cluster ID.
    """
    t = thresholds or _DEFAULT_THRESHOLDS
    total_articles = len(all_article_ids)
    all_ids = set(all_article_ids)

    claims: list[ClaimConsensus] = []
    for cluster in clusters:
        present = set(cluster.article_ids)
        absent = sorted(all_ids - present)
        category = _coverage_category(cluster.coverage_ratio, t)

        claims.append(
            ClaimConsensus(
                cluster_id=cluster.cluster_id,
                representative=cluster.representative,
                coverage_count=cluster.coverage_count,
                total_articles=total_articles,
                coverage_ratio=cluster.coverage_ratio,
                coverage_category=category,
                outlets_present=sorted(present),
                outlets_absent=absent,
            )
        )

    # Sort by descending coverage_ratio, then cluster_id for stability
    claims.sort(key=lambda c: (-c.coverage_ratio, c.cluster_id))

    return ConsensusView(total_articles=total_articles, claims=claims)
