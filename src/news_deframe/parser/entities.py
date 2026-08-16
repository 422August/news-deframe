"""Named Entity Recognition with modifier extraction.

For each entity found by spaCy's NER pipeline, this module traverses
the dependency tree to collect:

- ``amod``  tokens (adjectival modifiers)
- ``advmod`` tokens (adverbial modifiers)

These modifiers reveal descriptive tone differences between two articles
covering the same event.

Entity Filtering
----------------
Low-information entity types are excluded so that the output focuses on
agents, locations, and groups that are actually relevant to framing analysis:

**Kept** (high-signal):
    ``PERSON``, ``ORG``, ``GPE``, ``NORP``, ``FAC``, ``EVENT``

**Excluded** (noise):
    ``CARDINAL``, ``DATE``, ``TIME``, ``PERCENT``, ``QUANTITY``,
    ``ORDINAL``, ``MONEY``, ``LANGUAGE``, ``WORK_OF_ART``, ``LAW``,
    ``PRODUCT``

Both English (en_core_web_md) and Chinese (zh_core_web_md) model labels are
covered; any label not in the keep-list is silently skipped.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span, Token

from news_deframe.schemas import EntityModifier

# ── Entity type allow-list ────────────────────────────────────────────────────

#: Entity types considered informative for framing analysis.
#: Covers both spaCy English and Chinese model label sets.
_INFORMATIVE_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        # People, organisations, places, groups
        "PERSON",
        "PER",   # some zh models use PER
        "ORG",
        "GPE",
        "LOC",   # generic location used by some models
        "NORP",  # nationalities, religious or political groups
        "FAC",   # buildings, airports, etc.
        "EVENT",
    }
)

#: Entity types that carry little framing information and are suppressed.
_NOISE_ENTITY_TYPES: frozenset[str] = frozenset(
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
    }
)

# ── Modifier dependencies ─────────────────────────────────────────────────────

#: Dependency labels considered "descriptive" modifiers.
_MODIFIER_DEPS: frozenset[str] = frozenset({"amod", "advmod", "nmod", "compound"})


# ── Helpers ───────────────────────────────────────────────────────────────────


def _is_informative_entity(label: str) -> bool:
    """Return ``True`` if *label* should be included in the output.

    The check first looks at the explicit allow-list; if the label is not in
    either the allow-list or the noise-list it is **included** so that
    unknown labels from future models are not silently dropped.
    """
    if label in _INFORMATIVE_ENTITY_TYPES:
        return True
    if label in _NOISE_ENTITY_TYPES:
        return False
    # Unknown label → include conservatively
    return True


def _get_modifiers(entity: Span) -> list[str]:
    """Collect adjective / adverb modifiers associated with the entity span.

    The strategy:
    1.  For the entity's root token, collect all children with modifier deps.
    2.  Also check one level up (the root's head) for modifiers that govern
        the whole entity phrase.
    """
    root: Token = entity.root
    modifiers: list[str] = []

    # Direct children of the entity root
    for child in root.children:
        if child.dep_ in _MODIFIER_DEPS and child not in entity:
            modifiers.append(child.text)

    # Modifiers that govern the entity root from its head
    for sibling in root.head.children:
        if sibling.dep_ in _MODIFIER_DEPS and sibling.i < root.i:
            modifiers.append(sibling.text)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for m in modifiers:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    return unique


# ── Public API ────────────────────────────────────────────────────────────────


def extract_entity_modifiers(doc: Doc) -> list[EntityModifier]:
    """Extract named entities and their associated modifiers from *doc*.

    Only entities whose type is in the high-signal allow-list are included;
    numeric / temporal entities (``CARDINAL``, ``DATE``, ``TIME``, etc.) are
    silently skipped.

    Parameters
    ----------
    doc:
        A spaCy ``Doc`` with NER and dependency annotations.

    Returns
    -------
    list[EntityModifier]
        One record per unique entity (by text + label), filtered to
        informative entity types only.
    """
    seen_keys: set[tuple[str, str]] = set()
    results: list[EntityModifier] = []

    for ent in doc.ents:
        # Skip low-information entity types
        if not _is_informative_entity(ent.label_):
            continue

        key = (ent.text, ent.label_)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        modifiers = _get_modifiers(ent)
        results.append(
            EntityModifier(
                entity_name=ent.text,
                entity_type=ent.label_,
                modifiers=modifiers,
            )
        )

    return results
