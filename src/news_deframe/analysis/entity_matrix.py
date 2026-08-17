"""Entity x Outlet framing matrix construction.

This module is the integration point between the actor resolution pipeline and
the rest of the analysis stack.  It exposes the same public API as before
(``build_entity_outlet_matrix`` returning ``EntityOutletMatrix``) so that
callers and the console formatter require minimal changes.

Internal architecture
---------------------
1. ``resolve_actors`` (actor_resolution module) runs the full extraction,
   validation, canonicalization, and role-aggregation pipeline.
2. Results are projected into ``EntityOutletProfile`` objects for backward
   compatibility with existing consumers.
3. The ``entity_names`` list is ordered by importance score (most cross-outlet,
   most SVO-grounded actors first) rather than alphabetically, so that the
   console display cap of 20 rows shows the most useful actors.

Denomination contract (from ActorRoleStats, surfaced here)
----------------------------------------------------------
``total_mentions = agent_count + patient_count``  (role_occurrence_count)
``agent_ratio    = agent_count / total_mentions``  (0.0 when denom = 0)
``patient_ratio  = patient_count / total_mentions``  (0.0 when denom = 0)
``passive_ratio  = passive_patient_count / patient_count``  (0.0 when denom = 0)

The ``passive_ratio`` field has changed denominator from ``total_mentions``
to ``patient_count`` to give a more precise measure of passive-patient rate.
"""
from __future__ import annotations

from news_deframe.schemas import ParsedArticle
from news_deframe.analysis.schemas import (
    EntityOutletMatrix,
    EntityOutletProfile,
)
from news_deframe.analysis.actor_resolution import resolve_actors, ActorRoleStats


def _stats_to_profile(stats: ActorRoleStats) -> EntityOutletProfile:
    """Convert an ActorRoleStats into the backward-compatible EntityOutletProfile.

    Field mapping
    -------------
    entity_name      <- canonical_name
    total_mentions   <- role_occurrence_count  (agent + patient only)
    subject_count    <- agent_count
    object_count     <- patient_count
    passive_count    <- passive_patient_count
    agent_ratio      <- agent_ratio   (denom = role_occurrence_count)
    patient_ratio    <- patient_ratio (denom = role_occurrence_count)
    passive_ratio    <- passive_patient_ratio (denom = patient_count)
    modifiers        <- associated_modifiers
    associated_verbs <- agent verbs + patient verbs (deduplicated, order-stable)
    """
    agent_verbs = list(stats.associated_agent_verbs)
    patient_verbs = [v for v in stats.associated_patient_verbs if v not in agent_verbs]
    combined_verbs = agent_verbs + patient_verbs

    return EntityOutletProfile(
        entity_name=stats.canonical_name,
        article_id=stats.article_id,
        total_mentions=stats.role_occurrence_count,
        subject_count=stats.agent_count,
        object_count=stats.patient_count,
        passive_count=stats.passive_patient_count,
        agent_ratio=stats.agent_ratio,
        patient_ratio=stats.patient_ratio,
        passive_ratio=stats.passive_patient_ratio,
        modifiers=list(stats.associated_modifiers),
        associated_verbs=combined_verbs,
    )


def build_entity_outlet_matrix(
    articles: list[ParsedArticle],
    *,
    min_mentions: int = 0,
) -> EntityOutletMatrix:
    """Build the entity x outlet framing matrix for *articles*.

    Parameters
    ----------
    articles:
        Parsed articles in the event corpus.
    min_mentions:
        Minimum total role-grounded mentions (agent + patient) across all
        outlets for an actor to be included in the matrix.  Default 0 includes
        all validated actors.

    Returns
    -------
    EntityOutletMatrix
        entity_names ordered by importance score (most cross-outlet and
        most SVO-grounded actors first).
        profiles contains one EntityOutletProfile per (entity, article).
    """
    if not articles:
        return EntityOutletMatrix(entity_names=[], article_ids=[], profiles=[])

    article_ids = sorted(a.article_id for a in articles)

    canonical_actors, role_stats = resolve_actors(articles)

    # Filter by min_mentions (total role occurrences across all articles)
    actor_totals: dict[str, int] = {}
    for stats in role_stats:
        actor_totals[stats.canonical_name] = (
            actor_totals.get(stats.canonical_name, 0) + stats.role_occurrence_count
        )

    included_actors = {
        name for name, total in actor_totals.items() if total >= min_mentions
    }

    # Maintain importance ordering from resolve_actors (already sorted)
    ordered_names = [
        a.canonical_name
        for a in canonical_actors
        if a.canonical_name in included_actors
    ]

    # Build profiles only for included actors
    profiles: list[EntityOutletProfile] = [
        _stats_to_profile(stats)
        for stats in role_stats
        if stats.canonical_name in included_actors
    ]

    return EntityOutletMatrix(
        entity_names=ordered_names,
        article_ids=article_ids,
        profiles=profiles,
    )
