"""Named Entity Recognition and evaluative modifier extraction.

This module performs three complementary extraction passes over a spaCy Doc:

Pass 1 – NER entities
    High-signal named entities (PERSON, ORG, GPE, NORP, FAC, EVENT) are retained
    after POS-purity, boundary-character, and length filtering. For each surviving
    entity, the dependency tree is searched for descriptive modifiers (amod, advmod).

Pass 2 – Predicate verbs (framing descriptors)
    Political / journalistic framing is frequently expressed through evaluative
    adverbs on action verbs (「草率掏空」, 「粗暴破壞」).
    Every VERB with at least one advmod or neg child is recorded with
    entity_type = "VERB_ACTION".

Pass 3 – Event / action nouns
    Evaluative framing also attaches to key nouns (「合理分配」, 「透明原則」).
    Every NOUN or PROPN that is not already covered by an NER span and carries
    at least one qualifying modifier (amod, compound) is recorded with
    entity_type = "EVENT_NOUN".
"""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span, Token

from news_deframe.schemas import EntityModifier

# ── Entity type allow-list ────────────────────────────────────────────────────

_ALLOWED_NER_LABELS: frozenset[str] = frozenset(
    {
        "PERSON",
        "PER",
        "ORG",
        "GPE",
        "LOC",
        "NORP",
        "FAC",
        "EVENT",
    }
)

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

# ── Chinese boundary heuristics ──────────────────────────────────────────────

_ZH_VERB_PREFIX_CHARS: frozenset[str] = frozenset(
    "表達說明表示提出指出呼籲宣布宣稱聲稱強調認為批評譴責"
)

_ZH_FUNCTION_SUFFIX_CHARS: frozenset[str] = frozenset(
    "的地得了嗎呢啊吧哦唉也都還是"
)

# ── Modifier dependency labels ────────────────────────────────────────────────

_ENTITY_MODIFIER_DEPS: frozenset[str] = frozenset(
    {"amod", "advmod", "nmod", "compound"}
)

_VERB_ADVERB_DEPS: frozenset[str] = frozenset({"advmod", "neg"})

_NOUN_ADJ_DEPS: frozenset[str] = frozenset({"amod", "compound:nn", "compound"})

_EVENT_NOUN_POS: frozenset[str] = frozenset({"NOUN", "PROPN"})

_INVALID_ENTITY_POS: frozenset[str] = frozenset(
    {"VERB", "ADP", "PUNCT", "CCONJ", "AUX", "SCONJ"}
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _is_cjk(char: str) -> bool:
    """Return True if char is a CJK unified ideograph."""
    cp = ord(char)
    return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)


def _strip_punct(text: str) -> str:
    """Strip leading/trailing ASCII and Unicode punctuation from text."""
    _EXTRA_BRACKETS = set("「」『』【】〔〕《》〈〉")
    while text and (
        unicodedata.category(text[0]).startswith("P") or text[0] in _EXTRA_BRACKETS
    ):
        text = text[1:]
    while text and (
        unicodedata.category(text[-1]).startswith("P") or text[-1] in _EXTRA_BRACKETS
    ):
        text = text[:-1]
    return text


def _is_valid_entity(ent: "Span") -> bool:
    """Validate named entity spans using POS purity, boundary chars, and length."""
    raw_text = ent.text.strip()
    text = _strip_punct(raw_text)

    if len(text) < 2:
        return False

    # 實體內部若跨越動詞、介系詞、連詞或標點，判定為斷詞破裂
    if any(token.pos_ in _INVALID_ENTITY_POS for token in ent):
        return False

    # 首尾字元虛詞與動作字首防護
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
    """Collect direct-child modifier tokens whose dep is in dep_filter."""
    exclude = exclude_indices or frozenset()
    mods: list[str] = []
    for child in root.children:
        if child.dep_ in dep_filter and child.i not in exclude:
            mods.append(child.text)
    return mods


def _get_entity_modifiers(entity: "Span") -> list[str]:
    """Collect adjective/adverb modifiers for a named entity span."""
    root: "Token" = entity.root
    span_indices: frozenset[int] = frozenset(t.i for t in entity)

    modifiers: list[str] = _collect_children_modifiers(
        root, _ENTITY_MODIFIER_DEPS, span_indices
    )

    if root.head is not root:
        for sibling in root.head.children:
            if sibling.dep_ in _ENTITY_MODIFIER_DEPS and sibling.i < root.i:
                modifiers.append(sibling.text)

    seen: set[str] = set()
    unique: list[str] = []
    for m in modifiers:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


# ── Pass 1: NER entity extraction ─────────────────────────────────────────────


def _extract_ner_modifiers(doc: "Doc") -> list[EntityModifier]:
    """Extract high-signal named entities and their evaluative modifiers."""
    seen_keys: set[tuple[str, str]] = set()
    results: list[EntityModifier] = []

    for ent in doc.ents:
        label = ent.label_

        if label in _NOISE_NER_LABELS:
            continue

        if label not in _ALLOWED_NER_LABELS and label not in _NOISE_NER_LABELS:
            pass
        elif label not in _ALLOWED_NER_LABELS:
            continue

        if not _is_valid_entity(ent):
            continue

        text = _strip_punct(ent.text.strip())
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
    """Extract action verbs carrying evaluative adverbs or negation markers."""
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


def _extract_event_noun_modifiers(
    doc: "Doc", ner_texts: frozenset[str]
) -> list[EntityModifier]:
    """Extract key nouns that carry adjectival or compound modifiers."""
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
    """Extract framing descriptors from doc via three complementary passes."""
    ner_results = _extract_ner_modifiers(doc)
    ner_texts: frozenset[str] = frozenset(r.entity_name for r in ner_results)

    verb_results = _extract_verb_action_modifiers(doc)
    noun_results = _extract_event_noun_modifiers(doc, ner_texts)

    return ner_results + verb_results + noun_results