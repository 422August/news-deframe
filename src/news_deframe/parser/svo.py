"""SVO (Subject-Verb-Object) extraction with bilingual passive voice detection.

Strategy
--------
For each sentence in a spaCy ``Doc``:

1.  Find every token whose POS is ``VERB`` and whose dep is ``ROOT``
    (or any secondary verb that heads a clause).
2.  Walk the immediate children to collect:
    - Subjects  → dep tags ``nsubj``, ``csubj``, ``nsubjpass``, ``csubjpass``,
                  ``nsubj:pass``
    - Objects   → dep tags ``dobj``, ``pobj``, ``obj``, ``iobj``
3.  Detect passive voice using language-aware rules:

    **Chinese rules**
    - Voice markers: character-level tokens ``被``, ``遭``, ``受到`` present
      as children of the verb or anywhere in the sentence.
    - Dependency tags containing ``pass`` (e.g. ``nsubjpass``, ``auxpass``).

    **English rules**
    - Auxiliary dependencies ``aux:pass`` (e.g. "was arrested").
    - Subject dependencies ``nsubj:pass`` on any child.
    - Prepositional agents introduced by ``agent`` dependency (e.g.
      "arrested by the police").

The language is communicated to ``extract_svo`` via the optional *lang*
parameter (``'zh'`` or ``'en'``).  When omitted the function infers it from
the doc text for convenience.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import spacy
    from spacy.tokens import Doc, Span, Token

from news_deframe.schemas import SVORecord

Lang = Literal["zh", "en"]

# ── Chinese passive markers (surface forms) ───────────────────────────────────

_ZH_PASSIVE_CHARS: frozenset[str] = frozenset({"被", "遭", "受到", "由", "经", "为"})

# ── Dependency labels treated as subjects (universal + spaCy zh/en tags) ──────

_SUBJECT_DEPS: frozenset[str] = frozenset(
    {"nsubj", "csubj", "nsubjpass", "csubjpass", "nsubj:pass", "csubj:pass"}
)

# ── Dependency labels treated as objects ──────────────────────────────────────

_OBJECT_DEPS: frozenset[str] = frozenset({"dobj", "pobj", "obj", "iobj"})

# ── Chinese passive dep markers ───────────────────────────────────────────────

_ZH_PASSIVE_DEP_MARKERS: frozenset[str] = frozenset(
    {"nsubjpass", "csubjpass", "auxpass"}
)

# ── English passive dep markers ───────────────────────────────────────────────

_EN_PASSIVE_DEP_MARKERS: frozenset[str] = frozenset(
    {"aux:pass", "nsubj:pass", "csubj:pass", "auxpass", "nsubjpass", "agent"}
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _collect_span_text(token: Token) -> str:
    """Return the full noun phrase / subtree text for a dependency head token."""
    subtree_tokens = sorted(token.subtree, key=lambda t: t.i)
    return "".join(t.text_with_ws for t in subtree_tokens).strip()


def _detect_lang_from_text(text: str) -> Lang:
    """Lightweight language guess from raw sentence text (no external deps)."""
    cjk = sum(
        1 for ch in text
        if (0x4E00 <= ord(ch) <= 0x9FFF) or (0x3400 <= ord(ch) <= 0x4DBF)
    )
    alpha = sum(1 for ch in text if ch.isalpha())
    if alpha == 0:
        return "zh"
    return "zh" if (cjk / alpha) >= 0.15 else "en"


def _detect_passive_zh(verb_token: Token, sent: Span) -> tuple[bool, list[str]]:
    """Chinese passive detection: character markers + dep tags."""
    markers: list[str] = []

    for child in verb_token.children:
        # dep-based markers
        if child.dep_ in _ZH_PASSIVE_DEP_MARKERS or "pass" in child.dep_:
            markers.append(child.text)
        # surface character markers
        if child.text in _ZH_PASSIVE_CHARS:
            if child.text not in markers:
                markers.append(child.text)

    # Scan raw sentence for pre-verb passive characters
    for char in _ZH_PASSIVE_CHARS:
        if char in sent.text and char not in markers:
            markers.append(char)

    return bool(markers), markers


def _detect_passive_en(verb_token: Token, sent: Span) -> tuple[bool, list[str]]:
    """English passive detection: aux:pass, nsubj:pass, agent (by …)."""
    markers: list[str] = []

    for child in verb_token.children:
        dep = child.dep_
        if dep in _EN_PASSIVE_DEP_MARKERS or "pass" in dep:
            markers.append(child.text)
        # "agent" dep catches "by the police" prepositional phrases
        if dep == "agent":
            markers.append(child.text)

    return bool(markers), markers


def _detect_passive(
    verb_token: Token, sent: Span, lang: Lang
) -> tuple[bool, list[str]]:
    """Dispatch to the language-specific passive detector."""
    if lang == "zh":
        return _detect_passive_zh(verb_token, sent)
    return _detect_passive_en(verb_token, sent)


# ── Public API ────────────────────────────────────────────────────────────────


def extract_svo(doc: Doc, lang: Lang | None = None) -> list[SVORecord]:
    """Extract SVO records from every sentence in *doc*.

    Parameters
    ----------
    doc:
        A spaCy ``Doc`` object (must have been parsed with a dependency model).
    lang:
        ``'zh'`` or ``'en'``.  When *None* (default) the language is inferred
        from ``doc.text`` using the same CJK-proportion heuristic as
        :func:`~news_deframe.parser.spacy_loader.detect_language`.

    Returns
    -------
    list[SVORecord]
        One or more records per sentence (one per verbal head found).
    """
    resolved_lang: Lang = lang or _detect_lang_from_text(doc.text)
    records: list[SVORecord] = []

    for sent in doc.sents:
        # Collect all verbal roots in this sentence
        verb_tokens: list[Token] = [
            t
            for t in sent
            if t.pos_ in {"VERB"}
            and t.dep_ in {"ROOT", "conj", "relcl", "advcl", "ccomp"}
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

            # Inherit subjects from head verb in conjunction chains
            if not subjects and verb.head != verb:
                for sibling in verb.head.children:
                    if sibling.dep_ in _SUBJECT_DEPS:
                        subjects.append(_collect_span_text(sibling))

            is_passive, voice_markers = _detect_passive(verb, sent, resolved_lang)

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
