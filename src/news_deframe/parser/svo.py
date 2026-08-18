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
        "name",
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
        "當時", "日前", "過去", "近期", "動期", "初期", "昨天", "今天", "今天（14日）", "14日",
        "yesterday", "today", "tomorrow", "morning", "evening", "period",
    }
)

_ZH_TITLE_ROLE_MORPHEMES: tuple[str, ...] = (
    "院長", "部長", "主席", "總召", "委員", "立委", "議員", "參選人", "發言人", "召集人",
    "署長", "局長", "處長", "司長", "主任", "校長", "代表", "總經理", "執行長", "秘書長",
    "幹事長", "書記長", "總統", "市長", "縣長", "首長", "閣揆", "黨主席", "黨團總召", "長",
)


def _is_temporal_token(tok: Token) -> bool:
    """Return True if *tok* is a temporal noun or modifier."""
    text = getattr(tok, "text", "")
    dep = getattr(tok, "dep_", "")
    if text in _TEMPORAL_MODIFIERS or dep in {"nmod:tmod", "obl:tmod"}:
        return True
    if any(text.endswith(s) for s in ("期間", "傍晚", "晚間", "上午", "下午", "昨日", "日前", "昨天", "今天")):
        return True
    return False


def _is_participant_head(token: Token) -> bool:
    """Return True when *token* is a POS-valid head for a participant span."""
    return getattr(token, "pos_", "") in _PARTICIPANT_HEAD_POS


def _collect_participant_span(token: Token) -> str | None:
    """Return the participant span text for a dependency head token, or None.

    Preserves meaningful noun phrases while structurally excluding unrelated
    temporal material, locative material, clausal material, and predicates.
    Reconstructs complete title + name and institutional participant compounds.
    """
    if not _is_participant_head(token):
        return None

    collected_tokens: list[Token] = []

    def walk(tok: Token) -> None:
        collected_tokens.append(tok)
        for child in getattr(tok, "children", []):
            cdep = getattr(child, "dep_", "")
            cpos = getattr(child, "pos_", "")
            ctext = getattr(child, "text", "")
            if cdep in _PRUNE_DEPS:
                continue
            if _is_temporal_token(child):
                continue
            if cpos in {"VERB", "PUNCT", "ADP", "SCONJ"} and cdep != "mark:clf":
                # Special allowance: in Chinese parser, title+name or person name mis-tagged as VERB
                # (e.g. '長韓', '長卓', '榮泰') adjacent to nominal head
                if len(ctext) >= 2 and any(ctext.startswith(t) for t in _ZH_TITLE_ROLE_MORPHEMES):
                    collected_tokens.append(child)
                continue
            if cdep in _VALID_NOMINAL_DEPS:
                walk(child)

    walk(token)

    # Linear span expansion for Chinese titles, proper names, and institutional compounds
    doc = getattr(token, "doc", None)
    if doc is not None and len(collected_tokens) > 0 and getattr(token, "i", None) is not None:
        min_i = min(getattr(t, "i", 0) for t in collected_tokens)
        max_i = max(getattr(t, "i", 0) for t in collected_tokens)

        # Expand left if immediately preceded by institution / party nominal
        while min_i > 0:
            prev_t = doc[min_i - 1]
            ptext = getattr(prev_t, "text", "")
            ppos = getattr(prev_t, "pos_", "")
            pdep = getattr(prev_t, "dep_", "")
            if _is_temporal_token(prev_t) or ppos in {"PUNCT", "ADP", "CCONJ", "VERB"}:
                break
            if pdep in {"compound", "compound:nn", "name", "flat", "nmod", "appos", "nsubj"}:
                collected_tokens.append(prev_t)
                min_i -= 1
            else:
                break

        # Expand right if immediately followed by title / proper name morpheme
        while max_i < len(doc) - 1:
            next_t = doc[max_i + 1]
            curr_t = doc[max_i]
            ntext = getattr(next_t, "text", "")
            npos = getattr(next_t, "pos_", "")
            ndep = getattr(next_t, "dep_", "")
            ctext = getattr(curr_t, "text", "")
            if _is_temporal_token(next_t) or npos in {"PUNCT", "ADP", "CCONJ"}:
                break
            if ntext in {"表示", "指出", "說", "認為", "強調", "呼籲", "敲槌", "受訪", "提案", "簽名", "協商", "執行", "推動"}:
                break
            if ndep in {"compound", "compound:nn", "name", "flat", "appos", "dobj", "ccomp"} or npos in {"NOUN", "PROPN"}:
                collected_tokens.append(next_t)
                max_i += 1
            elif len(ntext) >= 2 and any(ntext.startswith(t) for t in _ZH_TITLE_ROLE_MORPHEMES):
                collected_tokens.append(next_t)
                max_i += 1
            elif ctext.startswith("長") or any(ctext.endswith(m) for m in ("長", "主席", "總召", "立委", "議員", "參選人", "部長", "院長")):
                # Given name following title morpheme (e.g. 長卓 + 榮泰 -> 行政院長卓榮泰)
                collected_tokens.append(next_t)
                max_i += 1
            else:
                break

    # Fallback to subtree if single token without children
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

    # If token is PROPN/NOUN and its governing verb starts with a name character (e.g. 卓榮 + 泰強 -> 卓榮泰)
    if getattr(token, "pos_", "") in {"PROPN", "NOUN"} and hasattr(token, "head") and token.head != token:
        head_text = getattr(token.head, "text", "")
        if len(head_text) == 2 and getattr(token.head, "pos_", "") == "VERB":
            from news_deframe.parser.predicate_normalization import _ZH_VALID_LEXICAL_COMPOUND_VERBS
            for child in getattr(token.head, "children", []):
                ctext = getattr(child, "text", "")
                if (head_text[1:] + ctext) in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
                    if not span_text.endswith(head_text[0]):
                        span_text = span_text + head_text[0]

    # Clean leading/trailing particles, temporals, and actions
    import re
    span_text = re.sub(r"^(昨天|今天|日前|當時|過去|近期)\s*", "", span_text)
    span_text = re.sub(r"(敲槌後|受訪說|提案指出|受訪時|受訪|敲槌|簽名|協商|提案)$", "", span_text)
    span_text = re.sub(r"[的之時後前等在地向從於]+$", "", span_text).strip()
    span_text = re.sub(r"^[在向從於對到]\s*", "", span_text).strip()
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

    # 2. Check head token if it is a passive auxiliary (e.g. 遭, 遭暴, 被, 受到)
    if verb_token.head != verb_token:
        head_text = getattr(verb_token.head, "text", "")
        if head_text in {"被", "遭", "遭到", "受到"} or any(
            head_text.startswith(c) for c in {"被", "遭", "遭到", "受到"}
        ):
            markers.append(head_text)

    # 3. Check direct children (excluding subordinate clauses / conjunctions)
    for child in verb_token.children:
        if child.dep_ in _CLAUSE_DEPS:
            continue
        if child.dep_ in _ZH_PASSIVE_DEP_MARKERS or "pass" in child.dep_:
            markers.append(child.text)
        elif child.text in {"被", "遭", "由", "经", "為"}:
            if child.text not in markers:
                markers.append(child.text)

    # 4. Check preceding passive markers in the clause
    if not markers:
        for t in sent:
            if t.text in {"被", "遭", "遭到", "受到"} or t.text.startswith("遭") or t.text.startswith("被"):
                if t.i < verb_token.i and (t == verb_token.head or verb_token in getattr(t, "children", [])):
                    markers.append(t.text)

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
    from news_deframe.parser.predicate_normalization import (
        is_valid_predicate_token,
        normalize_predicate_text,
    )

    resolved_lang: Lang = lang or _detect_lang_from_text(doc.text)
    records: list[SVORecord] = []

    for sent in doc.sents:
        # Collect all verbal roots in this sentence that pass linguistic validation
        verb_tokens: list[Token] = [
            t
            for t in sent
            if t.pos_ in {"VERB", "AUX"}
            and t.dep_ in {"ROOT", "conj", "relcl", "advcl", "ccomp", "xcomp"}
            and is_valid_predicate_token(t, lang=resolved_lang)
        ]

        if not verb_tokens:
            # Fallback: any valid verb
            verb_tokens = [
                t for t in sent
                if t.pos_ in {"VERB", "AUX"} and is_valid_predicate_token(t, lang=resolved_lang)
            ]

        if not verb_tokens:
            continue

        # If a passive auxiliary has a lexical verb complement/conj in the same sentence, skip the auxiliary
        has_lexical_verbs = any(
            t.text not in _ZH_PASSIVE_CHARS and not any(t.text.startswith(c) for c in _ZH_PASSIVE_CHARS)
            for t in verb_tokens
        )
        if has_lexical_verbs and len(verb_tokens) > 1:
            filtered_verbs = []
            for t in verb_tokens:
                is_aux_only = t.text in _ZH_PASSIVE_CHARS or any(t.text.startswith(c) for c in _ZH_PASSIVE_CHARS)
                if is_aux_only:
                    continue
                filtered_verbs.append(t)
            if filtered_verbs:
                verb_tokens = filtered_verbs

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
                        if grandchild.dep_ in _OBJECT_DEPS or grandchild.pos_ in _PARTICIPANT_HEAD_POS:
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

            # Inherit subjects from head verb only in non-reporting conjunction chains or complement of 遭
            is_reporting_head = verb.head.text in {"表示", "指出", "說", "認為", "強調", "呼籲", "批評", "譴責", "say", "state", "report"}
            if not subjects and verb.head != verb and not is_reporting_head:
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
            if verb.head != verb and verb.head.text in {"遭", "遭到"} and verb.dep_ in {"ccomp", "xcomp"}:
                for sib in verb.head.children:
                    if sib.dep_ in _ACTIVE_SUBJ_DEPS or sib.dep_ in _PASSIVE_SUBJ_DEPS:
                        span = _collect_participant_span(sib)
                        if span and span not in objects:
                            objects.append(span)

            norm_verb = normalize_predicate_text(
                verb.lemma_ or verb.text,
                sentence=sent.text,
                head_token=verb,
                lang=resolved_lang,
            )

            final_verb = norm_verb or verb.lemma_ or verb.text
            if not is_valid_predicate_token(verb, final_verb, lang=resolved_lang):
                continue

            records.append(
                SVORecord(
                    sentence=sent.text.strip(),
                    verb=final_verb,
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

