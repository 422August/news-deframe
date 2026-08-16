"""Named Entity Recognition with modifier extraction.

For each entity found by spaCy's NER pipeline, this module traverses
the dependency tree to collect:

- ``amod``  tokens (adjectival modifiers)
- ``advmod`` tokens (adverbial modifiers)

These modifiers reveal descriptive tone differences between two articles
covering the same event.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span, Token

from news_deframe.schemas import EntityModifier

# Dependency labels considered "descriptive" modifiers
_MODIFIER_DEPS: frozenset[str] = frozenset({"amod", "advmod", "nmod", "compound"})


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


def extract_entity_modifiers(doc: Doc) -> list[EntityModifier]:
    """Extract named entities and their associated modifiers from *doc*.

    Parameters
    ----------
    doc:
        A spaCy ``Doc`` with NER and dependency annotations.

    Returns
    -------
    list[EntityModifier]
        One record per unique entity (by text + label).
    """
    seen_keys: set[tuple[str, str]] = set()
    results: list[EntityModifier] = []

    for ent in doc.ents:
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
