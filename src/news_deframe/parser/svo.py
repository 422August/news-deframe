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

_PASSIVE_SUBJ_DEPS: frozenset[str] = frozenset(
    {"nsubjpass", "csubjpass", "nsubj:pass", "csubj:pass"}
)

_ACTIVE_SUBJ_DEPS: frozenset[str] = frozenset({"nsubj", "csubj"})

_CLAUSE_DEPS: frozenset[str] = frozenset(
    {"ccomp", "conj", "advcl", "xcomp", "acl", "relcl"}
)


# ── Helpers ───────────────────────────────────────────────────────────────────


# POS tags that are acceptable as the syntactic head of a participant span.
# A span headed by ADJ, VERB, ADV, or function words cannot be an event participant.
_PARTICIPANT_HEAD_POS: frozenset[str] = frozenset(
    {"NOUN", "PROPN", "PRON", "NUM", "X"}
)

_VALID_NOMINAL_DEPS: frozenset[str] = frozenset(
    {
        "compound",
        "compound:nn",
        "nmod",
        "nummod",
        "clf",
        "mark:clf",
        "det",
        "amod",
        "poss",
        "nmod:poss",
        "flat",
        "conj",
        "cc",
        "appos",
    }
)

_PRUNE_DEPS: frozenset[str] = frozenset(
    {
        "relcl",
        "acl",
        "advcl",
        "ccomp",
        "xcomp",
        "csubj",
        "csubjpass",
        "nmod:tmod",
        "obl:tmod",
        "advmod",
        "prep",
        "obl:loc",
        "punct",
    }
)

_TEMPORAL_MODIFIERS: frozenset[str] = frozenset(
    {
        "昨日", "今日", "明天", "傍晚", "晚間", "上午", "下午", "期間",
        "當時", "日前", "過去", "近期", "動期", "初期",
        "yesterday", "today", "tomorrow", "morning", "evening", "period",
    }
)


def _is_temporal_token(tok: Token) -> bool:
    """Return True if *tok* is a temporal noun or modifier."""
    text = getattr(tok, "text", "")
    dep = getattr(tok, "dep_", "")
    if text in _TEMPORAL_MODIFIERS or dep in {"nmod:tmod", "obl:tmod"}:
        return True
    if any(text.endswith(s) for s in ("期間", "傍晚", "晚間", "上午", "下午", "昨日", "日前")):
        return True
    return False


def _is_participant_head(token: Token) -> bool:
    """Return True when *token* is a POS-valid head for a participant span.

    Rejects adjectives, verbs, adverbs, and function words as argument heads,
    because these cannot represent event participants.
    """
    return getattr(token, "pos_", "") in _PARTICIPANT_HEAD_POS


def _collect_participant_span(token: Token) -> str | None:
    """Return the participant span text for a dependency head token, or None.

    Preserves meaningful noun phrases while structurally excluding unrelated
    temporal material, locative material, clausal material, and predicates.
    """
    if not _is_participant_head(token):
        return None

    collected_tokens = []

    def walk(tok: Token) -> None:
        collected_tokens.append(tok)
        for child in getattr(tok, "children", []):
            cdep = getattr(child, "dep_", "")
            cpos = getattr(child, "pos_", "")
            if cdep in _PRUNE_DEPS:
                continue
            if _is_temporal_token(child):
                continue
            if cpos in {"VERB", "PUNCT", "ADP", "SCONJ"} and cdep != "mark:clf":
                continue
            if cdep in _VALID_NOMINAL_DEPS:
                walk(child)

    walk(token)

    # Fallback to subtree if mock token or explicitly defined subtree without children generator
    if len(collected_tokens) <= 1 and hasattr(token, "subtree") and token.subtree:
        subtree_list = [
            t for t in token.subtree
            if getattr(t, "pos_", "") not in {"PUNCT", "VERB", "ADP"}
            and not _is_temporal_token(t)
        ]
        if len(subtree_list) > 1:
            collected_tokens = subtree_list

    sorted_toks = sorted(set(collected_tokens), key=lambda t: getattr(t, "i", 0))
    span_text = "".join(getattr(t, "text_with_ws", getattr(t, "text", "") + " ") for t in sorted_toks).strip()

    # Clean trailing/leading particles
    import re
    span_text = re.sub(r"[的之]+$", "", span_text).strip()
    return span_text or None


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
    """Chinese passive detection: character markers + dep tags on the verb / clause."""
    markers: list[str] = []

    # 1. The verb itself is a passive/receptive predicate (e.g. 被, 遭, 遭到, 受到)
    if verb_token.text in {"被", "遭", "遭到", "受到"} or any(
        verb_token.text.startswith(c) for c in {"被", "遭", "遭到", "受到"}
    ):
        markers.append(verb_token.text)

    # 2. Check direct children (excluding subordinate clauses / conjunctions)
    for child in verb_token.children:
        if child.dep_ in _CLAUSE_DEPS:
            continue
        if child.dep_ in _ZH_PASSIVE_DEP_MARKERS or "pass" in child.dep_:
            markers.append(child.text)
        elif child.text in {"被", "遭", "由", "经", "為"}:
            if child.text not in markers:
                markers.append(child.text)

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
            is_passive, voice_markers = _detect_passive(verb, sent, resolved_lang)

            pass_subjs: list[str] = []
            act_subjs: list[str] = []
            objects: list[str] = []

            for child in verb.children:
                if child.dep_ in _PASSIVE_SUBJ_DEPS:
                    span = _collect_participant_span(child)
                    if span:
                        pass_subjs.append(span)
                elif child.dep_ in _ACTIVE_SUBJ_DEPS:
                    span = _collect_participant_span(child)
                    if span:
                        act_subjs.append(span)
                elif child.dep_ in _OBJECT_DEPS:
                    span = _collect_participant_span(child)
                    if span:
                        objects.append(span)
                elif child.dep_ in {"agent"} or (child.dep_ == "prep" and child.lemma_ == "by"):
                    # Prepositional agent in English passive: "arrested by the police"
                    for grandchild in child.children:
                        if grandchild.dep_ in _OBJECT_DEPS:
                            span = _collect_participant_span(grandchild)
                            if span:
                                objects.append(span)

            subjects: list[str] = []
            if is_passive:
                # If there are explicit passive subjects (nsubjpass), they are subjects (logical patients).
                # Any active subject (nsubj) in this passive clause is the agent -> put in objects.
                if pass_subjs and act_subjs:
                    subjects.extend(pass_subjs)
                    objects.extend(act_subjs)
                elif pass_subjs:
                    subjects.extend(pass_subjs)
                else:
                    subjects.extend(act_subjs)
            else:
                subjects.extend(act_subjs or pass_subjs)

            # Inherit subjects from head verb in conjunction chains or complement of 遭
            if not subjects and verb.head != verb:
                if verb.head.text in {"遭", "遭到"} and verb.dep_ in {"ccomp", "xcomp"}:
                    for sib in verb.head.children:
                        if sib.dep_ in _ACTIVE_SUBJ_DEPS or sib.dep_ in _PASSIVE_SUBJ_DEPS:
                            span = _collect_participant_span(sib)
                            if span and span not in objects:
                                objects.append(span)
                else:
                    for sibling in verb.head.children:
                        if sibling.dep_ in _SUBJECT_DEPS:
                            span = _collect_participant_span(sibling)
                            if span:
                                subjects.append(span)

            # If verb is ccomp of 遭 (e.g. 遭警方逮捕):
            # verb.head (遭) has subject '參與者' (patient). If verb has subject '警方' (agent),
            # then '參與者' is the logical patient (object) of verb!
            if verb.head != verb and verb.head.text in {"遭", "遭到"} and verb.dep_ in {"ccomp", "xcomp"}:
                for sib in verb.head.children:
                    if sib.dep_ in _ACTIVE_SUBJ_DEPS or sib.dep_ in _PASSIVE_SUBJ_DEPS:
                        span = _collect_participant_span(sib)
                        if span and span not in objects:
                            objects.append(span)

            from news_deframe.parser.predicate_normalization import normalize_predicate_text

            norm_verb = normalize_predicate_text(
                verb.lemma_ or verb.text,
                sentence=sent.text,
                head_token=verb,
                lang=resolved_lang,
            )

            records.append(
                SVORecord(
                    sentence=sent.text.strip(),
                    verb=norm_verb or verb.lemma_ or verb.text,
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
