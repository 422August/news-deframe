"""Named Entity Recognition and evaluative modifier extraction.

This module performs three complementary extraction passes over a spaCy ``Doc``:

Pass 1 – NER entities
    High-signal named entities (PERSON, ORG, GPE, NORP, FAC, EVENT) are kept
    after strict boundary-artefact and length filtering.  For each surviving
    entity the dependency tree is searched for descriptive modifiers
    (``amod``, ``advmod``).

Pass 2 – Predicate verbs & event nouns (framing descriptors)
    Political / journalistic framing is often expressed through evaluative
    adverbs on action verbs (「草率掏空」, 「粗暴破壞」) and adjectives on key
    nouns (「合理分配」, 「黑箱立法」).  Both are captured even when the verb /
    noun is not a named entity.

    *Verb action*  – every ``VERB`` with at least one ``advmod`` or negation
                     child is recorded with ``entity_type = "VERB_ACTION"``.
    *Event noun*   – every ``NOUN`` or ``PROPN`` that is not already covered
                     by an NER span AND carries at least one ``amod`` or
                     compound modifier is recorded with
                     ``entity_type = "EVENT_NOUN"``.

Artefact filtering (Chinese NER boundary fixes)
------------------------------------------------
SpaCy's Chinese model frequently generates spurious spans such as:

* ``會表`` (taken from 「記者會表達」) labelled PERSON
* ``謹衝`` (from 「嚴謹衝擊」) labelled GPE
* ``黑箱`` (from 「黑箱立法」) labelled ORG  ← single-word compound fragment

These are rejected by the following rules (applied in order):

1. **Length gate**: any entity text shorter than 2 characters is dropped.
2. **Prefix verb gate**: entities whose first character is a common Chinese
   verb / function word are very likely to be boundary artefacts.
3. **Suffix function-word gate**: similarly for trailing function words.
4. **Stopword gate**: a curated set of known-bad spans (e.g. single
   compounds pulled loose from their host) is rejected outright.

Output
------
Both passes return :class:`~news_deframe.schemas.EntityModifier` instances so
that the rest of the pipeline (``ParsedArticle``, console formatter) needs no
changes.  The ``entity_type`` field encodes whether an entry is a classic NER
label or one of the synthetic tags ``VERB_ACTION`` / ``EVENT_NOUN``.
"""
from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span, Token

from news_deframe.schemas import EntityModifier

# ── Entity type allow-list ────────────────────────────────────────────────────

#: Only these NER labels are considered high-signal for framing analysis.
_ALLOWED_NER_LABELS: frozenset[str] = frozenset(
    {
        "PERSON",
        "PER",    # alternative label used by some zh models
        "ORG",
        "GPE",
        "LOC",    # generic location
        "NORP",   # nationalities, religious / political groups
        "FAC",    # facilities (buildings, airports …)
        "EVENT",
    }
)

#: Numeric / temporal / quantity labels – always dropped.
_NOISE_NER_LABELS: frozenset[str] = frozenset(
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

# ── Chinese artefact filtering ────────────────────────────────────────────────

#: Chinese characters that commonly appear as leading boundary artefacts when
#: the tokeniser cuts a verb off a conference / assembly noun.
_ZH_VERB_PREFIX_CHARS: frozenset[str] = frozenset(
    "表達說明表示提出指出呼籲宣布宣稱聲稱強調認為批評譴責"
)

#: Chinese trailing function / auxiliary characters that signal the entity ran
#: one token too far.
_ZH_FUNCTION_SUFFIX_CHARS: frozenset[str] = frozenset(
    "的地得了嗎呢啊吧哦唉也都還是"
)

#: Known-bad surface forms that the zh NER routinely hallucinates.
_ZH_ARTEFACT_BLOCKLIST: frozenset[str] = frozenset(
    {
        "會表",
        "謹衝",
        "黑箱",
        "程序",
        "結果",
        "情況",
        "影響",
        "問題",
        "方式",
    }
)

# ── Modifier dependency labels ────────────────────────────────────────────────

#: Dependency relations that signal a *descriptive* modifier on an NER entity.
_ENTITY_MODIFIER_DEPS: frozenset[str] = frozenset(
    {"amod", "advmod", "nmod", "compound"}
)

#: Dependency relations that signal an evaluative adverb on a *predicate verb*.
_VERB_ADVERB_DEPS: frozenset[str] = frozenset({"advmod", "neg"})

#: Dependency relations that signal a descriptive adjective on an *event noun*.
_NOUN_ADJ_DEPS: frozenset[str] = frozenset({"amod", "compound:nn", "compound"})


# ── Helpers ───────────────────────────────────────────────────────────────────


def _is_cjk(char: str) -> bool:
    """Return ``True`` if *char* is a CJK unified ideograph."""
    cp = ord(char)
    return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)


def _strip_punct(text: str) -> str:
    """Strip leading/trailing ASCII and Unicode punctuation from *text*."""
    _EXTRA_BRACKETS = set("「」『』【】〔〕《》〈〉")
    while text and (unicodedata.category(text[0]).startswith("P") or text[0] in _EXTRA_BRACKETS):
        text = text[1:]
    while text and (unicodedata.category(text[-1]).startswith("P") or text[-1] in _EXTRA_BRACKETS):
        text = text[:-1]
    return text


def _is_valid_entity(text: str) -> bool:
    """Return ``True`` if *text* passes all artefact-rejection heuristics.

    Rejection rules (applied in order; first match → reject):

    1. After stripping punctuation the text is empty.
    2. Text is shorter than 2 characters.
    3. The span is in the known-bad blocklist.
    4. The *first* character is a common Chinese verb prefix.
    5. The *last* character is a common Chinese function/auxiliary suffix.
    """
    text = _strip_punct(text)
    if not text:
        return False
    if len(text) < 2:
        return False
    if text in _ZH_ARTEFACT_BLOCKLIST:
        return False
    if _is_cjk(text[0]) and text[0] in _ZH_VERB_PREFIX_CHARS:
        return False
    if _is_cjk(text[-1]) and text[-1] in _ZH_FUNCTION_SUFFIX_CHARS:
        return False
    return True


def _collect_children_modifiers(
    root: "Token",
    dep_filter: frozenset[str],
    exclude_indices: frozenset[int] | None = None,
) -> list[str]:
    """Collect direct-child modifier tokens whose dep is in *dep_filter*.

    Tokens with an index in *exclude_indices* (e.g. the entity span itself)
    are skipped.
    """
    exclude = exclude_indices or frozenset()
    mods: list[str] = []
    for child in root.children:
        if child.dep_ in dep_filter and child.i not in exclude:
            mods.append(child.text)
    return mods


def _get_entity_modifiers(entity: "Span") -> list[str]:
    """Collect adjective/adverb modifiers for a *named entity* span.

    Strategy:
    1.  Collect ``amod``/``advmod``/``nmod``/``compound`` children of the
        entity root that lie *outside* the span.
    2.  Walk one level up to the root's syntactic head and collect pre-entity
        siblings with the same modifier deps.
    """
    root: "Token" = entity.root
    span_indices: frozenset[int] = frozenset(t.i for t in entity)

    modifiers: list[str] = _collect_children_modifiers(
        root, _ENTITY_MODIFIER_DEPS, span_indices
    )

    # Pre-entity modifiers governed by the syntactic head (e.g. "嚴重 [ORG]")
    if root.head is not root:
        for sibling in root.head.children:
            if sibling.dep_ in _ENTITY_MODIFIER_DEPS and sibling.i < root.i:
                modifiers.append(sibling.text)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for m in modifiers:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


# ── Pass 1: NER entity extraction ─────────────────────────────────────────────


def _extract_ner_modifiers(doc: "Doc") -> list[EntityModifier]:
    """Extract high-signal named entities and their evaluative modifiers.

    Only entities in ``_ALLOWED_NER_LABELS`` that pass boundary-artefact
    validation are included.  Unknown labels (neither in the allow- nor the
    noise-list) are included conservatively for future model compatibility.
    """
    seen_keys: set[tuple[str, str]] = set()
    results: list[EntityModifier] = []

    for ent in doc.ents:
        label = ent.label_

        # Drop low-signal numeric/temporal labels
        if label in _NOISE_NER_LABELS:
            continue

        # Drop anything not in the allow-list when it IS in the noise list
        # (belt-and-suspenders; unknown labels are kept for forward-compat)
        if label not in _ALLOWED_NER_LABELS and label not in _NOISE_NER_LABELS:
            # Unknown label → include conservatively
            pass
        elif label not in _ALLOWED_NER_LABELS:
            continue

        text = ent.text
        if not _is_valid_entity(text):
            continue

        key = (text, label)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        modifiers = _get_entity_modifiers(ent)
        results.append(
            EntityModifier(
                entity_name=text,
                entity_type=label,
                modifiers=modifiers,
            )
        )

    return results


# ── Pass 2: predicate verb adverbs ────────────────────────────────────────────


def _extract_verb_action_modifiers(doc: "Doc") -> list[EntityModifier]:
    """Extract action verbs that carry evaluative adverbs or negation markers.

    For each ``VERB`` token in the doc, collect ``advmod`` and ``neg``
    children.  Only verbs with at least one such modifier are recorded.

    The synthetic label ``VERB_ACTION`` is used for ``entity_type``.
    """
    seen_keys: set[tuple[str, str]] = set()
    results: list[EntityModifier] = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        mods = _collect_children_modifiers(token, _VERB_ADVERB_DEPS)
        if not mods:
            continue

        key = (token.text, "VERB_ACTION")
        if key in seen_keys:
            continue
        seen_keys.add(key)

        results.append(
            EntityModifier(
                entity_name=token.text,
                entity_type="VERB_ACTION",
                modifiers=mods,
            )
        )

    return results


# ── Pass 3: event / action noun adjectives ────────────────────────────────────

# POS tags that can form "event nouns" in Chinese political text.
_EVENT_NOUN_POS: frozenset[str] = frozenset({"NOUN", "PROPN"})


def _extract_event_noun_modifiers(
    doc: "Doc", ner_texts: frozenset[str]
) -> list[EntityModifier]:
    """Extract key nouns that carry adjectival / compound modifiers.

    Nouns already captured by the NER pass (identified via *ner_texts*) are
    skipped to avoid duplication.  Only nouns with at least one qualifying
    modifier are recorded.

    The synthetic label ``EVENT_NOUN`` is used for ``entity_type``.
    """
    seen_keys: set[tuple[str, str]] = set()
    results: list[EntityModifier] = []

    for token in doc:
        if token.pos_ not in _EVENT_NOUN_POS:
            continue
        if token.text in ner_texts:
            continue

        mods = _collect_children_modifiers(token, _NOUN_ADJ_DEPS)
        if not mods:
            continue

        key = (token.text, "EVENT_NOUN")
        if key in seen_keys:
            continue
        seen_keys.add(key)

        results.append(
            EntityModifier(
                entity_name=token.text,
                entity_type="EVENT_NOUN",
                modifiers=mods,
            )
        )

    return results


# ── Public API ────────────────────────────────────────────────────────────────


def extract_entity_modifiers(doc: "Doc") -> list[EntityModifier]:
    """Extract framing descriptors from *doc* via three complementary passes.

    The function combines:

    * **Pass 1 – NER entities**: high-signal named entities (PERSON, ORG, GPE,
      NORP, FAC, EVENT) after boundary-artefact filtering.
    * **Pass 2 – Verb actions**: action verbs carrying evaluative adverbs
      (``advmod``) or negation markers (``neg``).
    * **Pass 3 – Event nouns**: nouns / proper nouns with adjectival or
      compound modifiers that are *not* already covered by a NER span.

    Parameters
    ----------
    doc:
        A spaCy ``Doc`` with NER and dependency annotations.

    Returns
    -------
    list[EntityModifier]
        Combined results from all three passes.  Entries are ordered: NER
        entities first, then verb actions, then event nouns.  The
        ``entity_type`` field encodes provenance (NER label, ``VERB_ACTION``,
        or ``EVENT_NOUN``).
    """
    ner_results = _extract_ner_modifiers(doc)

    # Build the set of NER texts for dedup in pass 3
    ner_texts: frozenset[str] = frozenset(r.entity_name for r in ner_results)

    verb_results = _extract_verb_action_modifiers(doc)
    noun_results = _extract_event_noun_modifiers(doc, ner_texts)

    return ner_results + verb_results + noun_results
