"""SVO (Subject-Verb-Object) extraction with passive voice detection.

Strategy
--------
For each sentence in a spaCy ``Doc``:

1.  Find every token whose POS is ``VERB`` and whose dep is ``ROOT``
    (or any secondary verb that heads a clause).
2.  Walk the immediate children to collect:
    - Subjects  → dep tags ``nsubj``, ``csubj``, ``nsubjpass``, ``csubjpass``
    - Objects   → dep tags ``dobj``, ``pobj``, ``obj``, ``iobj``
3.  Detect passive voice by:
    - Presence of Chinese passive markers 被/遭/受到/由 as children or in sentence.
    - spaCy's ``nsubjpass`` / ``csubjpass`` dependency on any token.
    - dep tag ``auxpass`` on a child of the verb.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import spacy
    from spacy.tokens import Doc, Span, Token

from news_deframe.schemas import SVORecord

# Chinese passive markers (surface forms)
_PASSIVE_CHARS: frozenset[str] = frozenset({"被", "遭", "受到", "由", "经", "为"})

# Dependency labels treated as subjects
_SUBJECT_DEPS: frozenset[str] = frozenset({"nsubj", "csubj", "nsubjpass", "csubjpass"})

# Dependency labels treated as objects
_OBJECT_DEPS: frozenset[str] = frozenset({"dobj", "pobj", "obj", "iobj"})

# Dependency / morphological markers for passive
_PASSIVE_DEP_MARKERS: frozenset[str] = frozenset({"nsubjpass", "csubjpass", "auxpass"})


def _collect_span_text(token: Token) -> str:
    """Return the full noun phrase / subtree text for a dependency head token."""
    # Expand to the full subtree for richer subject/object text
    subtree_tokens = sorted(token.subtree, key=lambda t: t.i)
    return "".join(t.text_with_ws for t in subtree_tokens).strip()


def _detect_passive(verb_token: Token, sent: Span) -> tuple[bool, list[str]]:
    """Return (is_passive, voice_markers) for a given verb token in a sentence."""
    markers: list[str] = []

    # 1. Check dependency labels on children
    for child in verb_token.children:
        if child.dep_ in _PASSIVE_DEP_MARKERS:
            markers.append(child.text)
        if child.text in _PASSIVE_CHARS:
            markers.append(child.text)

    # 2. Scan the raw sentence text for passive characters (covers pre-verb position)
    for char in _PASSIVE_CHARS:
        if char in sent.text and char not in markers:
            markers.append(char)

    is_passive = bool(markers)
    return is_passive, markers


def extract_svo(doc: Doc) -> list[SVORecord]:
    """Extract SVO records from every sentence in *doc*.

    Parameters
    ----------
    doc:
        A spaCy ``Doc`` object (must have been parsed with a dependency model).

    Returns
    -------
    list[SVORecord]
        One or more records per sentence (one per verbal head found).
    """
    records: list[SVORecord] = []

    for sent in doc.sents:
        # Collect all verbal roots in this sentence
        verb_tokens: list[Token] = [
            t for t in sent if t.pos_ in {"VERB"} and t.dep_ in {"ROOT", "conj", "relcl", "advcl", "ccomp"}
        ]

        if not verb_tokens:
            # Fallback: any verb
            verb_tokens = [t for t in sent if t.pos_ == "VERB"]

        if not verb_tokens:
            continue

        for verb in verb_tokens:
            subjects: list[str] = []
            objects: list[str] = []

            for child in verb.children:
                if child.dep_ in _SUBJECT_DEPS:
                    subjects.append(_collect_span_text(child))
                elif child.dep_ in _OBJECT_DEPS:
                    objects.append(_collect_span_text(child))

            # Also consider siblings sharing the same head for conjunction verbs
            if not subjects and verb.head != verb:
                for sibling in verb.head.children:
                    if sibling.dep_ in _SUBJECT_DEPS:
                        subjects.append(_collect_span_text(sibling))

            is_passive, voice_markers = _detect_passive(verb, sent)

            records.append(
                SVORecord(
                    sentence=sent.text.strip(),
                    verb=verb.lemma_ or verb.text,
                    subjects=subjects,
                    objects=objects,
                    is_passive=is_passive,
                    voice_markers=voice_markers,
                )
            )

    return records


def passive_ratio(records: list[SVORecord]) -> float:
    """Return the fraction of SVO records that are passive (0.0 if empty)."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.is_passive) / len(records)
