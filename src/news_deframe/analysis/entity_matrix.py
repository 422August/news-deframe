"""Entity \u00d7 Outlet framing matrix construction.

For each named entity found across the corpus, this module builds a profile
describing how each outlet structurally positions that entity using the SVO
records already extracted by the parser pipeline:

- *subject/agent* appearances: the entity fills the grammatical subject slot.
- *object/patient* appearances: the entity fills the grammatical object slot.
- *passive* appearances: the entity appears in a passive construction.
- *modifiers*: evaluative adjectives/adverbs attached to the entity.
- *associated verbs*: head verbs for which the entity acts as subject or object.

Only NER entities (PERSON, ORG, GPE, NORP, LOC, FAC, EVENT) are included in
the matrix.  VERB_ACTION and EVENT_NOUN descriptors from the entities pipeline
are excluded because they are not structural participants in the same sense.

Reporting is strictly descriptive.  The matrix exposes *what was found*;
interpretation (e.g. \u201cOutlet A is biased\u201d) is left to the human analyst.
"""
from __future__ import annotations

from collections import defaultdict

from news_deframe.schemas import ParsedArticle, SVORecord
from news_deframe.analysis.schemas import (
    EntityOutletMatrix,
    EntityOutletProfile,
)

# NER label types that represent structural participants
_NER_LABELS: frozenset[str] = frozenset(
    {"PERSON", "PER", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT"}
)


def _entity_names_from_article(article: ParsedArticle) -> set[str]:
    """Return the set of NER entity names from an article."""
    return {
        em.entity_name
        for em in article.entity_modifiers
        if em.entity_type in _NER_LABELS
    }


def _normalise(name: str) -> str:
    """Normalise an entity name for matching (lowercased, stripped)."""
    return name.strip().lower()


def _contains_entity(text: str, entity: str) -> bool:
    """Case-insensitive substring presence check."""
    return _normalise(entity) in _normalise(text)


def _build_profile(
    entity_name: str,
    article: ParsedArticle,
) -> EntityOutletProfile:
    """Construct a framing profile for *entity_name* in *article*."""
    subject_count = 0
    object_count = 0
    passive_count = 0
    associated_verbs: list[str] = []

    for record in article.svo_records:
        entity_as_subject = any(
            _contains_entity(subj, entity_name) for subj in record.subjects
        )
        entity_as_object = any(
            _contains_entity(obj, entity_name) for obj in record.objects
        )

        if entity_as_subject:
            subject_count += 1
            if record.verb:
                associated_verbs.append(record.verb)
        if entity_as_object:
            object_count += 1
            if record.verb:
                associated_verbs.append(record.verb)
        if record.is_passive and (entity_as_subject or entity_as_object):
            passive_count += 1

    total_mentions = subject_count + object_count
    agent_ratio = round(subject_count / total_mentions, 4) if total_mentions else 0.0
    patient_ratio = round(object_count / total_mentions, 4) if total_mentions else 0.0
    passive_ratio = round(passive_count / total_mentions, 4) if total_mentions else 0.0

    # Modifiers from entity_modifiers for this entity name
    modifiers: list[str] = []
    for em in article.entity_modifiers:
        if em.entity_type in _NER_LABELS and em.entity_name == entity_name:
            modifiers.extend(em.modifiers)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_mods: list[str] = []
    for m in modifiers:
        if m not in seen:
            seen.add(m)
            unique_mods.append(m)

    seen_v: set[str] = set()
    unique_verbs: list[str] = []
    for v in associated_verbs:
        if v not in seen_v:
            seen_v.add(v)
            unique_verbs.append(v)

    return EntityOutletProfile(
        entity_name=entity_name,
        article_id=article.article_id,
        total_mentions=total_mentions,
        subject_count=subject_count,
        object_count=object_count,
        passive_count=passive_count,
        agent_ratio=agent_ratio,
        patient_ratio=patient_ratio,
        passive_ratio=passive_ratio,
        modifiers=unique_mods,
        associated_verbs=unique_verbs,
    )


def build_entity_outlet_matrix(
    articles: list[ParsedArticle],
    *,
    min_mentions: int = 0,
) -> EntityOutletMatrix:
    """Build the entity \u00d7 outlet framing matrix for *articles*.

    Parameters
    ----------
    articles:
        Parsed articles in the event corpus.
    min_mentions:
        Minimum total mentions across all outlets for an entity to be included
        in the matrix.  Set to ``1`` to exclude entities with zero SVO
        participation (default ``0`` includes all NER entities).

    Returns
    -------
    EntityOutletMatrix
        Profiles sorted by entity name, then by article ID.
    """
    # Collect all distinct NER entity names across the corpus
    all_entity_names: set[str] = set()
    for article in articles:
        all_entity_names |= _entity_names_from_article(article)

    if not all_entity_names:
        return EntityOutletMatrix(
            entity_names=[],
            article_ids=[a.article_id for a in articles],
            profiles=[],
        )

    article_ids = sorted(a.article_id for a in articles)
    article_map = {a.article_id: a for a in articles}

    profiles: list[EntityOutletProfile] = []
    for entity_name in sorted(all_entity_names):
        entity_total = 0
        entity_profiles: list[EntityOutletProfile] = []
        for aid in article_ids:
            profile = _build_profile(entity_name, article_map[aid])
            entity_profiles.append(profile)
            entity_total += profile.total_mentions

        if entity_total >= min_mentions:
            profiles.extend(entity_profiles)

    included_entities = sorted(
        {p.entity_name for p in profiles}
    )

    return EntityOutletMatrix(
        entity_names=included_entities,
        article_ids=article_ids,
        profiles=profiles,
    )
