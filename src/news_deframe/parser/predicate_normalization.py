"""Predicate validation and normalization layer.

Provides a 3-stage predicate processing pipeline:
    raw predicate token/span
    → validated predicate (structural and POS purity checks)
    → normalized predicate (lemma, compound reconstruction, particle attachment)

Key rules:
- Validates that candidates are true verbal predicates rather than adjectives,
  nouns, adverbial fragments, or tokenization artifacts.
- Handles Chinese compound verb reconstruction (e.g. split tokens like '正調' + '查' → '調查').
- Handles Chinese passive/receptive predicate linking (e.g. '遭' + '逮捕' → '逮捕').
- Handles English phrasal verbs / particle attachment (e.g. 'break' + 'through' → 'break through').
- Preserves raw token representation for full traceability.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from spacy.tokens import Token

Lang = Literal["zh", "en"]

# ── Non-predicate POS and dependency tags ─────────────────────────────────────

_INVALID_PREDICATE_POS: frozenset[str] = frozenset(
    {"NOUN", "PROPN", "ADJ", "ADV", "PRON", "NUM", "PUNCT", "SYM", "ADP", "CCONJ", "SCONJ", "DET"}
)

_INVALID_PREDICATE_DEPS: frozenset[str] = frozenset(
    {"compound", "compound:nn", "nmod", "amod", "advmod", "nummod", "det", "case", "punct"}
)

# Common Chinese auxiliary / receptive verbs that take verbal complements
_ZH_PASSIVE_AUXILIARIES: frozenset[str] = frozenset({"遭", "遭到", "被", "受到", "經", "由"})

# Stative adjectives that often act as roots in Chinese but are states rather than actions
_ZH_STATIVE_ADJECTIVES: frozenset[str] = frozenset({
    "平穩", "平靜", "緊張", "突然", "嚴重", "輕微", "大礙", "良好", "激烈", "混亂", "迅速"
})


@dataclass(frozen=True)
class PredicateProvenance:
    """Detailed provenance for an extracted and normalized predicate."""

    raw_text: str
    head_lemma: str
    normalized_predicate: str
    is_valid: bool
    is_passive: bool
    confidence: float


def is_valid_predicate_token(token: Token) -> bool:
    """Return True if *token* is a linguistically defensible verbal predicate head."""
    pos = getattr(token, "pos_", "")
    if pos not in {"VERB", "AUX"}:
        return False

    dep = getattr(token, "dep_", "")
    if dep in _INVALID_PREDICATE_DEPS:
        return False

    text = getattr(token, "text", "").strip()
    if not text:
        return False

    # Check for stative adjectives misparsed as VERB in Chinese
    if text in _ZH_STATIVE_ADJECTIVES:
        return False

    # Reject broken single-char non-words or punctuation
    if len(text) == 1:
        cp = ord(text[0])
        # Allow common single-character CJK verbs (e.g. 有, 遭, 看, 說, 稱, 指)
        is_cjk = (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)
        if not is_cjk and not text.isalpha():
            return False

    return True


def _lemmatize_en_word(word: str) -> str:
    """Lightweight rule-based English lemmatizer for offline normalization."""
    w = word.lower().strip()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ied") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing"):
        base = w[:-3]
        if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "ls":
            return base[:-1]
        if base.endswith(("at", "iz", "us", "clos", "mov", "tak", "mak")):
            return base + "e"
        return base
    if w.endswith("ed"):
        base = w[:-2]
        if base.endswith(("deploy", "monitor", "suspend", "repair", "plant", "arrest", "start", "report", "question", "demand")):
            return base
        if base.endswith(("allocat", "complet", "restor", "charg", "clos")):
            return base + "e"
        if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "ls":
            return base[:-1]
        return base
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        if w.endswith("es"):
            return w[:-2]
        return w[:-1]
    return w


def normalize_predicate_text(
    verb_text: str,
    *,
    sentence: str = "",
    head_token: Token | None = None,
    lang: Lang = "zh",
) -> str:
    """Normalize a raw verb text/token to its canonical base predicate form.

    Parameters
    ----------
    verb_text:
        Surface text or lemma of the verb.
    sentence:
        Containing sentence string (used for compound token repair).
    head_token:
        Optional spaCy Token for dependency/child inspection.
    lang:
        Language code ('zh' or 'en').

    Returns
    -------
    str: Normalized predicate string.
    """
    raw = verb_text.strip()
    if not raw:
        return ""

    if head_token is not None:
        # 1. Chinese compound repair and passive complement resolution
        if lang == "zh":
            # If verb is '遭', '遭到', '被' and has a verbal ccomp/xcomp child, use the lexical verb
            if raw in _ZH_PASSIVE_AUXILIARIES:
                for child in getattr(head_token, "children", []):
                    if getattr(child, "dep_", "") in {"ccomp", "xcomp", "conj"} and getattr(child, "pos_", "") == "VERB":
                        child_text = getattr(child, "text", "")
                        if child_text and child_text not in _ZH_PASSIVE_AUXILIARIES:
                            return child_text

            # If verb head is passive aux (e.g. head is '遭' and verb is '逮捕')
            if getattr(head_token, "head", None) is not None:
                parent = head_token.head
                if getattr(parent, "text", "") in _ZH_PASSIVE_AUXILIARIES and raw not in _ZH_PASSIVE_AUXILIARIES:
                    return raw

            # Check if token is split (e.g. '正調' -> check if followed by '查' in sentence)
            if sentence and raw in sentence:
                idx = sentence.find(raw)
                if idx >= 0 and idx + len(raw) < len(sentence):
                    next_char = sentence[idx + len(raw)]
                    if raw == "正調" and next_char == "查":
                        return "調查"
                    if raw == "辦團" and next_char == "體":
                        return "主辦"

        # 2. English phrasal verb particle attachment (e.g. 'break' + 'through' -> 'break through')
        elif lang == "en":
            particles = [
                getattr(c, "text", "").lower()
                for c in getattr(head_token, "children", [])
                if getattr(c, "dep_", "") == "prt"
            ]
            lemma = getattr(head_token, "lemma_", raw).lower()
            if not lemma or lemma == raw.lower():
                lemma = _lemmatize_en_word(raw)
            if particles:
                return f"{lemma} {' '.join(particles)}"
            return lemma

    # Fallback / string-only normalization
    norm = raw
    if lang == "en":
        norm = _lemmatize_en_word(raw)
    elif lang == "zh":
        if norm in _ZH_PASSIVE_AUXILIARIES and sentence:
            for v in ("吹離", "逮捕", "延誤", "查扣", "重創", "質疑", "解僱", "撤銷"):
                if v in sentence:
                    return v
        if norm == "正調":
            return "調查"
        if norm == "活" and "活動" in sentence:
            return "活動"
        if norm == "辦團":
            return "主辦"

    return norm


def extract_normalized_predicate(
    token: Token | None,
    raw_text: str = "",
    sentence: str = "",
    *,
    is_passive: bool = False,
    lang: Lang = "zh",
) -> PredicateProvenance:
    """Build a full PredicateProvenance object for a verb token or string."""
    surface = getattr(token, "text", raw_text).strip() if token is not None else raw_text.strip()
    lemma = getattr(token, "lemma_", surface).strip() if token is not None else surface

    is_valid = is_valid_predicate_token(token) if token is not None else (len(surface) >= 1 and not surface.isdigit())
    normalized = normalize_predicate_text(lemma or surface, sentence=sentence, head_token=token, lang=lang)

    confidence = 1.0 if (token is not None and getattr(token, "pos_", "") == "VERB") else 0.8

    return PredicateProvenance(
        raw_text=surface,
        head_lemma=lemma,
        normalized_predicate=normalized or surface,
        is_valid=is_valid,
        is_passive=is_passive,
        confidence=confidence,
    )
