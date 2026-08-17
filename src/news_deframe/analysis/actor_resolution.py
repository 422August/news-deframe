"""Actor resolution pipeline for the Entity x Outlet Framing Matrix.

This module implements the layered pipeline:

    raw NER candidates  ──┐
                           ├─→ actor validation
    SVO-grounded spans  ──┘
    → actor mentions (SVO-grounded role assignment)
    → canonical actors (event-level deduplication)
    → framing statistics (typed, ratio-denominator-explicit)

Design principles
-----------------
* Two complementary candidate extraction paths:

  Path A (NER-based): Named entities whose NER label is in _ACTOR_NER_LABELS.
    Validated with structural checks to reject spans that the NER model may
    have misclassified (adjectives, verb fragments, short noise tokens).

  Path B (SVO-based): Subject and object spans from SVO records, validated
    by structural properties (surface length, character composition).
    This path recovers participants that NER fails to tag (e.g. generic
    institutional references in Chinese text such as 警方, 主辦團體).

* Actor validation uses multiple converging signals (NER type, SVO
  participation, frequency, recurrence) rather than any single heuristic.
* NER type supports actorhood but is not mandatory; structural SVO evidence
  can compensate for missing or incorrect NER labels.
* Canonicalization is conservative -- prefer false negatives over false merges.
* All statistics have explicit denominators and zero-denominator guards.
* No hard-coded vocabulary, blacklists, or fixture-specific aliases.
* Deterministic -- same corpus in any order produces the same canonical actors.
* Provenance is preserved throughout; every statistic is traceable to its
  source (article, sentence, mention, role, verb, passive flag, modifiers).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import NamedTuple

from pydantic import BaseModel, Field

from news_deframe.schemas import ParsedArticle


# -- NER label sets -----------------------------------------------------------

# Labels that represent human-scale actors or structured social entities.
_ACTOR_NER_LABELS: frozenset[str] = frozenset(
    {"PERSON", "PER", "ORG", "GPE", "NORP", "FAC", "LOC"}
)

# Labels that are never actors by type (temporal, quantitative, abstract, or verb).
_NON_ACTOR_NER_LABELS: frozenset[str] = frozenset(
    {
        "CARDINAL",
        "DATE",
        "TIME",
        "PERCENT",
        "QUANTITY",
        "ORDINAL",
        "MONEY",
        "LANGUAGE",
        "WORK_OF_ART",
        "LAW",
        "PRODUCT",
        "EVENT",
        "VERB_ACTION",
        "EVENT_NOUN",
    }
)

# spaCy non-actor NER labels (excludes synthetic modifier tags like EVENT_NOUN)
_SPACY_NON_ACTOR_NER_LABELS: frozenset[str] = frozenset(
    {
        "CARDINAL",
        "DATE",
        "TIME",
        "PERCENT",
        "QUANTITY",
        "ORDINAL",
        "MONEY",
        "LANGUAGE",
        "WORK_OF_ART",
        "LAW",
        "PRODUCT",
        "EVENT",
    }
)



# Sentinel NER type for SVO-derived candidates that lack a NER label.
_SVO_DERIVED_TYPE: str = "SVO_PARTICIPANT"


# -- Span structural validation -----------------------------------------------


def _is_valid_surface(text: str) -> bool:
    """Return True when *text* is structurally plausible as an actor name.

    Checks are based on linguistic/structural properties only -- no vocabulary.
    """
    stripped = text.strip()
    if len(stripped) < 2:
        return False

    # Must contain at least one letter or CJK character
    has_word_char = any(
        ch.isalpha() or (0x4E00 <= ord(ch) <= 0x9FFF) or (0x3400 <= ord(ch) <= 0x4DBF)
        for ch in stripped
    )
    if not has_word_char:
        return False

    # First and last char must not be pure punctuation/symbol
    first_cat = unicodedata.category(stripped[0])
    last_cat = unicodedata.category(stripped[-1])
    if first_cat.startswith("P") or first_cat.startswith("S"):
        return False
    if last_cat.startswith("P") or last_cat.startswith("S"):
        return False

    return True


def _is_valid_candidate_length(text: str, *, max_tokens: int = 8) -> bool:
    """Reject spans that are suspiciously long (likely parser fragments)."""
    token_count = len(text.split())
    if token_count == 1 and len(text) > 20:
        return False
    return token_count <= max_tokens


def _is_structurally_valid_ner_span(entity_name: str, entity_type: str) -> bool:
    """Structural validation for NER-derived candidates.

    Beyond type-level checks, validates that the surface form is plausible as
    an actor name using character-level structural properties only.
    Rejects very short spans, all-digit spans, and spans that contain no
    word characters.
    """
    text = entity_name.strip()
    if not _is_valid_surface(text):
        return False
    if not _is_valid_candidate_length(text):
        return False
    return True


def _is_valid_svo_span(span_text: str) -> bool:
    """Structural validation for SVO-derived participant spans.

    Returns True when *span_text* is plausible as a participant:
    - Contains at least one CJK character or alphabetic word char
    - Is not excessively long (protects against parser subtree artifacts)
    - Does not consist solely of digits or punctuation
    """
    text = span_text.strip()
    if len(text) < 2:
        return False

    # Character length limit for a participant span
    if len(text) > 30:
        return False

    # Token count limit (space-separated tokens for English; CJK is dense)
    token_count = len(text.split())
    if token_count > 6:
        return False

    has_word_char = any(
        ch.isalpha() or (0x4E00 <= ord(ch) <= 0x9FFF) or (0x3400 <= ord(ch) <= 0x4DBF)
        for ch in text
    )
    if not has_word_char:
        return False

    # First and last char must not be punctuation
    first_cat = unicodedata.category(text[0])
    last_cat = unicodedata.category(text[-1])
    if first_cat.startswith("P") or first_cat.startswith("S"):
        return False
    if last_cat.startswith("P") or last_cat.startswith("S"):
        return False

    return True


# -- Actor mention: a single SVO-grounded occurrence --------------------------


class ActorMention(NamedTuple):
    """A single occurrence of an actor in a specific grammatical role.

    Provenance fields allow every aggregated statistic to be traced to source.
    """

    article_id: str
    sentence: str
    surface: str       # original surface form from NER or SVO span
    role: str          # "agent", "patient", or "modifier_only"
    verb: str          # lemma of the associated verb (empty if modifier_only)
    is_passive: bool   # whether the containing SVO record is passive
    modifiers: list    # evaluative modifiers attached to this mention


# -- Canonical actor: event-level resolved identity ---------------------------


class CanonicalActor(BaseModel):
    """Event-level canonical actor with aggregated framing evidence."""

    canonical_name: str = Field(..., description="Chosen canonical surface form")
    entity_type: str = Field(..., description="NER label of the canonical form")
    surface_mentions: list[str] = Field(
        default_factory=list, description="All distinct surface forms mapped here"
    )
    mentions: list = Field(
        default_factory=list, description="All SVO-grounded occurrences (ActorMention)"
    )
    article_ids: list[str] = Field(
        default_factory=list, description="Distinct articles containing this actor"
    )


# -- Actor role statistics (per canonical actor x article) --------------------


class ActorRoleStats(BaseModel):
    """Strongly typed framing statistics for one canonical actor in one article.

    Denominator contract
    --------------------
    role_occurrence_count = agent_count + patient_count

    agent_ratio = agent_count / role_occurrence_count  (0.0 when denom = 0)
    patient_ratio = patient_count / role_occurrence_count  (0.0 when denom = 0)
    passive_patient_ratio = passive_patient_count / patient_count  (0.0 when denom = 0)
    """

    canonical_name: str
    article_id: str

    # Raw counts
    mention_count: int = Field(default=0, ge=0)
    role_occurrence_count: int = Field(
        default=0, ge=0, description="agent_count + patient_count"
    )
    agent_count: int = Field(default=0, ge=0)
    patient_count: int = Field(default=0, ge=0)
    passive_patient_count: int = Field(
        default=0, ge=0, description="patient occurrences in passive constructions"
    )

    # Normalized ratios (denominator = role_occurrence_count)
    agent_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    patient_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    passive_patient_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    # Verb evidence (lemmas, deduplicated, order-stable)
    associated_agent_verbs: list[str] = Field(default_factory=list)
    associated_patient_verbs: list[str] = Field(default_factory=list)

    # Modifier evidence
    associated_modifiers: list[str] = Field(default_factory=list)

    # Provenance
    provenance: list = Field(
        default_factory=list,
        description="Source ActorMention tuples this profile was aggregated from",
    )


# -- Stage 1a: NER candidate extraction with structural validation -------------


def _extract_ner_candidates(article: ParsedArticle) -> list:
    """Return (surface, ner_type, modifiers) triples for NER actor candidates.

    Applies structural validation in addition to type filtering to reject
    spans that the NER model has misclassified (e.g. adjectives tagged as
    PERSON or FAC by a low-accuracy Chinese model).
    """
    results = []
    for em in article.entity_modifiers:
        if em.entity_type in _NON_ACTOR_NER_LABELS:
            continue
        if em.entity_type not in _ACTOR_NER_LABELS:
            continue
        surface = em.entity_name.strip()
        if not _is_structurally_valid_ner_span(surface, em.entity_type):
            continue
        results.append((surface, em.entity_type, list(em.modifiers)))
    return results


def _extract_svo_candidates(article: ParsedArticle) -> list:
    """Return (surface, ner_type, modifiers) triples derived from SVO subjects/objects.

    This path recovers participants that NER fails to tag. The ner_type is
    set to _SVO_DERIVED_TYPE (a sentinel). These candidates are validated
    primarily by structural evidence (surface form, SVO recurrence across
    articles) rather than NER label.

    Spans that were explicitly identified by the entity extractor as non-actor
    entity types (DATE, TIME, CARDINAL, EVENT, LAW, PRODUCT, EVENT_NOUN, etc.)
    are excluded so that non-actor entities are not resurrected via SVO.

    Each unique surface form is emitted at most once per article.
    """
    non_actor_keys = {
        _normalize_key(em.entity_name)
        for em in article.entity_modifiers
        if em.entity_type in _SPACY_NON_ACTOR_NER_LABELS
    }

    seen: set[str] = set()
    results = []

    for record in article.svo_records:
        for span in record.subjects + record.objects:
            text = span.strip()
            if text in seen:
                continue
            if _normalize_key(text) in non_actor_keys:
                continue
            if not _is_valid_svo_span(text):
                continue
            seen.add(text)
            results.append((text, _SVO_DERIVED_TYPE, []))

    return results


# -- Stage 2: SVO matching ----------------------------------------------------


def _normalize_for_matching(text: str) -> str:
    """Lowercase and strip for matching."""
    return text.strip().lower()


def _candidate_in_span(candidate: str, span: str) -> bool:
    """Return True when candidate is present in span (case-insensitive).

    Uses word-boundary matching for ASCII to reduce false positives.
    Falls back to substring for CJK where word boundaries are not spaces.
    """
    norm_cand = _normalize_for_matching(candidate)
    norm_span = _normalize_for_matching(span)

    if norm_cand not in norm_span:
        return False

    # CJK: substring match is sufficient
    has_cjk = any(
        (0x4E00 <= ord(ch) <= 0x9FFF) or (0x3400 <= ord(ch) <= 0x4DBF)
        for ch in norm_cand
    )
    if has_cjk:
        return True

    # ASCII: word-boundary check to avoid partial matches
    pattern = r"(?<![a-z0-9_])" + re.escape(norm_cand) + r"(?![a-z0-9_])"
    return bool(re.search(pattern, norm_span))


def _match_candidate_to_svo(
    candidate: str,
    modifiers: list,
    article: ParsedArticle,
) -> list:
    """Produce SVO-grounded ActorMention records for candidate in article.

    Passive role inversion:
    - Passive grammatical subject -> logical patient
    - Passive grammatical object  -> logical agent
    """
    mentions = []

    for record in article.svo_records:
        in_subjects = any(_candidate_in_span(candidate, s) for s in record.subjects)
        in_objects = any(_candidate_in_span(candidate, o) for o in record.objects)

        if not (in_subjects or in_objects):
            continue

        verb = record.verb or ""

        if record.is_passive:
            if in_subjects:
                mentions.append(ActorMention(
                    article_id=article.article_id,
                    sentence=record.sentence,
                    surface=candidate,
                    role="patient",
                    verb=verb,
                    is_passive=True,
                    modifiers=list(modifiers),
                ))
            if in_objects:
                mentions.append(ActorMention(
                    article_id=article.article_id,
                    sentence=record.sentence,
                    surface=candidate,
                    role="agent",
                    verb=verb,
                    is_passive=True,
                    modifiers=list(modifiers),
                ))
        else:
            if in_subjects:
                mentions.append(ActorMention(
                    article_id=article.article_id,
                    sentence=record.sentence,
                    surface=candidate,
                    role="agent",
                    verb=verb,
                    is_passive=False,
                    modifiers=list(modifiers),
                ))
            if in_objects:
                mentions.append(ActorMention(
                    article_id=article.article_id,
                    sentence=record.sentence,
                    surface=candidate,
                    role="patient",
                    verb=verb,
                    is_passive=False,
                    modifiers=list(modifiers),
                ))

    return mentions


# -- Stage 3: Actor validation ------------------------------------------------


def _validate_actor(
    surface: str,
    ner_type: str,
    mentions: list,
    total_article_count: int,
    cross_article_frequency: int,
) -> bool:
    """Apply multi-signal actor validation.

    NER-backed candidates (S1):
        Require S1 plus at least one other signal (S2, S3, S4, or S5).

    SVO-only candidates (no S1):
        Require at least three other signals (S2, S3, S4, S5).
        This compensates for absence of NER evidence, reducing noise from
        parser artifacts that appear as SVO arguments.

    Signals
    -------
    S1 -- NER type in _ACTOR_NER_LABELS
    S2 -- Participates in >= 1 SVO subject/object slot
    S3 -- Appears in > 1 article in the corpus
    S4 -- Has >= 1 associated action verb (SVO-grounded)
    S5 -- Has >= 2 total SVO-grounded mentions
    """
    if not _is_valid_surface(surface):
        return False
    if ner_type in _NON_ACTOR_NER_LABELS:
        return False

    s1 = ner_type in _ACTOR_NER_LABELS
    s2 = len(mentions) > 0
    s3 = cross_article_frequency > 1
    s4 = any(m.verb for m in mentions)
    s5 = len(mentions) >= 2

    score = sum([s1, s2, s3, s4, s5])

    if s1 and score >= 2:
        return True
    if not s1 and score >= 3:
        return True

    return False


# -- Stage 4: Event-level canonicalization ------------------------------------


def _normalize_key(text: str) -> str:
    """Deterministic normalization key for canonicalization.

    Language-general transformations only:
    1. Strip whitespace
    2. Lowercase
    3. Remove leading English determiners (a/an/the)
    4. Collapse internal whitespace
    """
    text = text.strip().lower()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard token overlap between two surface forms."""
    set_a = set(_normalize_key(a).split())
    set_b = set(_normalize_key(b).split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _is_abbreviation_of(short: str, long_form: str) -> bool:
    """Return True when short is a plausible acronym of long_form (>= 2 chars)."""
    short_up = short.upper().strip()
    if len(short_up) < 2:
        return False
    words = [w for w in long_form.split() if w and w[0].isalpha()]
    initials = "".join(w[0].upper() for w in words)
    return short_up in initials and len(short_up) >= 2


def _type_group(t: str) -> str:
    """Map NER label to a broad compatibility group."""
    if t in {"PERSON", "PER"}:
        return "person"
    if t == "ORG":
        return "org"
    if t in {"GPE", "LOC"}:
        return "geo"
    if t == "NORP":
        return "group"
    if t == "FAC":
        return "fac"
    if t == _SVO_DERIVED_TYPE:
        return "svo"
    return t


def _should_merge(surface_a: str, type_a: str, surface_b: str, type_b: str) -> bool:
    """Return True when two candidates should share a canonical actor.

    Conservative: requires compatible NER types AND at least one strong
    lexical/abbreviation signal.

    SVO-derived candidates (_SVO_DERIVED_TYPE) can be merged with any other
    candidate when the surface forms match, because the SVO path may find
    the same participant that NER independently identified.
    """
    group_a = _type_group(type_a)
    group_b = _type_group(type_b)

    # Allow SVO-derived to merge with any candidate on exact normalized match
    svo_involved = _SVO_DERIVED_TYPE in {type_a, type_b}

    if not svo_involved and group_a != group_b:
        return False

    key_a = _normalize_key(surface_a)
    key_b = _normalize_key(surface_b)

    # Exact normalized match
    if key_a == key_b:
        return True

    # High token overlap (Jaccard >= 0.75)
    if _token_overlap_ratio(surface_a, surface_b) >= 0.75:
        return True

    # Abbreviation relationship
    if _is_abbreviation_of(surface_a, surface_b) or _is_abbreviation_of(surface_b, surface_a):
        return True

    return False


def _canonicalize_actors(validated: list) -> list:
    """Group validated (surface, ner_type, mentions) triples into CanonicalActors.

    Uses union-find over a deterministically sorted list of candidates.
    Canonical name chosen by mention frequency (tie-break: lexicographic).

    When a group contains both NER-backed and SVO-derived candidates, the
    canonical entity_type is chosen from NER-backed entries (majority vote
    among NER labels only; SVO_PARTICIPANT is used only as a fallback).
    """
    if not validated:
        return []

    # Sort for determinism
    sorted_cands = sorted(validated, key=lambda x: _normalize_key(x[0]))
    n = len(sorted_cands)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            surf_i, type_i, _ = sorted_cands[i]
            surf_j, type_j, _ = sorted_cands[j]
            if _should_merge(surf_i, type_i, surf_j, type_j):
                union(i, j)

    groups: dict = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    canonical_actors = []
    for root, indices in sorted(groups.items()):
        members = [sorted_cands[i] for i in indices]
        surfaces = [m[0] for m in members]

        # Choose canonical name by mention frequency
        surface_counter: Counter = Counter()
        for _, _, mention_list in members:
            for mention in mention_list:
                surface_counter[mention.surface] += 1
        for surf in surfaces:
            if surf not in surface_counter:
                surface_counter[surf] = 0

        canonical_name = max(
            surface_counter.keys(),
            key=lambda s: (surface_counter[s], -len(_normalize_key(s))),
        )

        all_mentions = []
        for _, _, mention_list in members:
            all_mentions.extend(mention_list)

        # Prefer NER-backed type over SVO_PARTICIPANT
        ner_backed_types = [m[1] for m in members if m[1] != _SVO_DERIVED_TYPE]
        if ner_backed_types:
            type_counts: Counter = Counter(ner_backed_types)
        else:
            type_counts = Counter(m[1] for m in members)
        entity_type = type_counts.most_common(1)[0][0]

        article_ids = sorted({m.article_id for m in all_mentions})

        canonical_actors.append(
            CanonicalActor(
                canonical_name=canonical_name,
                entity_type=entity_type,
                surface_mentions=sorted(set(surfaces)),
                mentions=all_mentions,
                article_ids=article_ids,
            )
        )

    return canonical_actors


# -- Stage 5: Role aggregation ------------------------------------------------


def _deduplicate_ordered(items: list) -> list:
    """Remove duplicates while preserving insertion order."""
    seen: set = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _aggregate_role_stats(
    actor: CanonicalActor,
    article_id: str,
    article_modifiers: list,
) -> ActorRoleStats:
    """Compute ActorRoleStats for actor in a specific article.

    Denominator: role_occurrence_count = agent_count + patient_count.
    All ratios are 0.0 when the denominator is zero.
    """
    article_mentions = [m for m in actor.mentions if m.article_id == article_id]

    mention_count = len(article_mentions)
    agent_count = sum(1 for m in article_mentions if m.role == "agent")
    patient_count = sum(1 for m in article_mentions if m.role == "patient")
    passive_patient_count = sum(
        1 for m in article_mentions if m.role == "patient" and m.is_passive
    )

    role_occurrence_count = agent_count + patient_count

    agent_ratio = round(agent_count / role_occurrence_count, 4) if role_occurrence_count > 0 else 0.0
    patient_ratio = round(patient_count / role_occurrence_count, 4) if role_occurrence_count > 0 else 0.0
    passive_patient_ratio = round(passive_patient_count / patient_count, 4) if patient_count > 0 else 0.0

    agent_verbs = _deduplicate_ordered(
        [m.verb for m in article_mentions if m.role == "agent" and m.verb]
    )
    patient_verbs = _deduplicate_ordered(
        [m.verb for m in article_mentions if m.role == "patient" and m.verb]
    )

    modifier_sources = list(article_modifiers)
    for m in article_mentions:
        modifier_sources.extend(m.modifiers)
    all_modifiers = _deduplicate_ordered(modifier_sources)

    return ActorRoleStats(
        canonical_name=actor.canonical_name,
        article_id=article_id,
        mention_count=mention_count,
        role_occurrence_count=role_occurrence_count,
        agent_count=agent_count,
        patient_count=patient_count,
        passive_patient_count=passive_patient_count,
        agent_ratio=agent_ratio,
        patient_ratio=patient_ratio,
        passive_patient_ratio=passive_patient_ratio,
        associated_agent_verbs=agent_verbs,
        associated_patient_verbs=patient_verbs,
        associated_modifiers=all_modifiers,
        provenance=article_mentions,
    )


# -- Stage 6: Importance ranking ----------------------------------------------


def _compute_importance(actor: CanonicalActor, total_articles: int) -> float:
    """Deterministic importance score based on structural signals only.

    Signals: cross-outlet presence, role-grounded mention count, verb diversity.
    No sentiment, political, or domain-specific weights.
    """
    outlet_fraction = len(actor.article_ids) / max(total_articles, 1)
    total_role_mentions = sum(1 for m in actor.mentions if m.role in {"agent", "patient"})
    distinct_verbs = len({m.verb for m in actor.mentions if m.verb})

    return (
        outlet_fraction * 4.0
        + min(total_role_mentions, 20) * 0.5
        + min(distinct_verbs, 10) * 0.3
    )


# -- Public API ---------------------------------------------------------------


def resolve_actors(
    articles: list,
    *,
    min_outlets: int = 1,
    max_actors_console: int = 20,
) -> tuple:
    """Run the full actor resolution pipeline over an event corpus.

    Parameters
    ----------
    articles : list[ParsedArticle]
        All parsed articles in the event corpus.
    min_outlets : int
        Minimum distinct articles an actor must appear in (default 1).
    max_actors_console : int
        Soft cap for console display; JSON export retains all.

    Returns
    -------
    (canonical_actors, role_stats)
        canonical_actors -- list[CanonicalActor] sorted by importance desc.
        role_stats       -- list[ActorRoleStats], one per (actor x article).
    """
    total_articles = len(articles)
    article_ids = sorted(a.article_id for a in articles)

    # Collect all surface keys across the corpus that were explicitly identified
    # as non-actor entity types (DATE, TIME, CARDINAL, EVENT, LAW, PRODUCT, etc.)
    corpus_non_actor_keys: set[str] = {
        _normalize_key(em.entity_name)
        for a in articles
        for em in a.entity_modifiers
        if em.entity_type in _SPACY_NON_ACTOR_NER_LABELS
    }


    raw_candidates: dict = defaultdict(list)
    key_article_set: dict = defaultdict(set)
    surface_modifiers: dict = defaultdict(list)

    for article in articles:
        # Path A: NER-backed candidates (with structural validation)
        ner_candidates = _extract_ner_candidates(article)
        # Path B: SVO-derived candidates (structural validation only)
        svo_candidates = _extract_svo_candidates(article)

        # Merge: if the same surface already comes from NER, skip the SVO
        # duplicate so that the NER-backed entry retains its type label.
        # Also exclude spans known across the corpus to be non-actor types.
        ner_surfaces: set[str] = {_normalize_key(c[0]) for c in ner_candidates}
        deduplicated_svo = [
            c for c in svo_candidates
            if _normalize_key(c[0]) not in ner_surfaces
            and _normalize_key(c[0]) not in corpus_non_actor_keys
        ]

        all_candidates = ner_candidates + deduplicated_svo


        for surface, ner_type, modifiers in all_candidates:
            nkey = _normalize_key(surface)
            mentions = _match_candidate_to_svo(surface, modifiers, article)
            raw_candidates[nkey].append((surface, ner_type, mentions))
            key_article_set[nkey].add(article.article_id)
            surface_modifiers[(article.article_id, surface)].extend(modifiers)

    validated = []
    seen_surf_type: set = set()

    for nkey, candidate_list in sorted(raw_candidates.items()):
        cross_article_freq = len(key_article_set[nkey])
        for surface, ner_type, mentions in candidate_list:
            key = (_normalize_key(surface), ner_type)
            if key in seen_surf_type:
                # Accumulate mentions into existing validated entry
                for entry in validated:
                    if _normalize_key(entry[0]) == _normalize_key(surface) and entry[1] == ner_type:
                        entry[2].extend(mentions)
                continue
            if _validate_actor(surface, ner_type, mentions, total_articles, cross_article_freq):
                seen_surf_type.add(key)
                validated.append([surface, ner_type, list(mentions)])

    canonical_actors = _canonicalize_actors(validated)

    canonical_actors = [
        a for a in canonical_actors if len(a.article_ids) >= min_outlets
    ]

    all_stats = []
    for actor in canonical_actors:
        for aid in article_ids:
            actor_modifiers = []
            for surf in actor.surface_mentions:
                actor_modifiers.extend(surface_modifiers.get((aid, surf), []))
            actor_modifiers = _deduplicate_ordered(actor_modifiers)
            stats = _aggregate_role_stats(actor, aid, actor_modifiers)
            all_stats.append(stats)

    canonical_actors.sort(
        key=lambda a: (-_compute_importance(a, total_articles), a.canonical_name)
    )

    return canonical_actors, all_stats
