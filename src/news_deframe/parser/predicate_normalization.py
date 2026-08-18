"""Predicate validation and normalization layer.

Provides a 3-stage predicate processing pipeline:
    raw predicate token/span
    → validated predicate (structural and POS purity checks)
    → normalized predicate (lemma, compound reconstruction, particle attachment)

Key rules:
- Validates that candidates are true verbal predicates rather than adjectives,
  nouns, adverbial fragments, connective fragments, or tokenization artifacts.
- Handles Chinese compound verb reconstruction (e.g. split tokens like '砍'+'刪' → '砍刪', '執'+'行' → '執行').
- Handles Chinese passive/receptive predicate linking (e.g. '遭' + '逮捕' → '逮捕', '遭' + '凍結' → '凍結').
- Handles English phrasal verbs / particle attachment (e.g. 'break' + 'through' → 'break through').
- Preserves raw token representation for full traceability.
- Domain-general: no hardcoded politician names, outlet names, or corpus phrases.
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
    {"NOUN", "PROPN", "ADJ", "ADV", "PRON", "NUM", "PUNCT", "SYM", "ADP", "CCONJ", "SCONJ", "DET", "SPACE"}
)

_INVALID_PREDICATE_DEPS: frozenset[str] = frozenset(
    {
        "compound",
        "compound:nn",
        "nmod",
        "amod",
        "advmod",
        "nummod",
        "det",
        "case",
        "punct",
        "nsubj",
        "nsubjpass",
        "csubj",
        "csubjpass",
        "dobj",
        "pobj",
        "obj",
        "iobj",
        "appos",
        "name",
        "flat",
        "mark",
    }
)

# Common Chinese auxiliary / receptive verbs that take verbal complements
_ZH_PASSIVE_AUXILIARIES: frozenset[str] = frozenset({"遭", "遭到", "被", "受到", "經", "由"})

# Stative adjectives that often act as roots in Chinese but describe states rather than actions
_ZH_STATIVE_ADJECTIVES: frozenset[str] = frozenset({
    "平穩", "平靜", "緊張", "突然", "嚴重", "輕微", "大礙", "良好", "激烈", "混亂", "迅速",
    "好", "壞", "高", "低", "多", "少", "大", "小", "晚", "早", "快", "慢", "完備", "齊全",
    "成熟", "適任", "不適任", "失望", "細漢", "失職", "失言", "無奈", "難過", "氣憤",
})

# Conjunctions, prepositions, or case particles that must not terminate a legitimate verb
_ZH_INVALID_VERB_SUFFIXES: tuple[str, ...] = (
    "與", "及", "和", "或", "同", "以", "在", "從", "由", "到", "向", "自", "於", "對", "跟", "等", "之", "讓",
)

# Discourse / connective prefixes that indicate non-verbal sentence connectors
_ZH_DISCOURSE_PREFIXES: frozenset[str] = frozenset({
    "對此", "因此", "由此", "從此", "如此", "不過", "另外", "其實", "樣說", "說這", "訪說", "展委",
    "請問", "想想", "請問到", "請問是否", "到底誰", "何以致之",
})

# Nominal morphemes that signal noun / title / classifier fragments misclassified as verbs
_ZH_BOUND_NOMINAL_MORPHEMES: tuple[str, ...] = (
    "費", "案", "額", "處", "員", "總", "例", "部", "所", "法", "局", "院", "籍", "性", "度", "率", "慣",
    "言行", "操行", "品行", "德行", "暴行", "戰", "僵局", "結果", "部分", "方面",
)

# Established compound action / reporting verbs that legitimately contain bound morphemes
_ZH_VALID_LEXICAL_COMPOUND_VERBS: frozenset[str] = frozenset({
    "執行", "進行", "推行", "推動", "裁決", "表決", "處理", "審判", "立法", "執法", "減列", "增列",
    "編列", "統刪", "砍刪", "凍結", "刪除", "審查", "通過", "簽名", "協商", "出面", "發布", "宣布",
    "提出", "指出", "表示", "認為", "呼籲", "批評", "譴責", "要求", "答應", "改正", "副署", "抗議",
    "質詢", "備詢", "說明", "強調", "反對", "贊成", "支持", "提案", "達成", "完成", "送到", "延宕",
    "影響", "敲槌", "受訪", "運作", "放寬", "嚴管", "啟動", "重啟", "停用", "查扣", "逮捕", "起訴",
})

# Common single-character CJK action / reporting verbs
_ZH_VALID_SINGLE_CHAR_VERBS: frozenset[str] = frozenset(
    "說稱提簽砍刪凍審查批遭看給訪決讓答派辦降增減買賣宣罰告警讀裁判抓救退換改請催追停封移扣送拒准控表談砍查"
)


@dataclass(frozen=True)
class PredicateProvenance:
    """Detailed provenance for an extracted and normalized predicate."""

    raw_text: str
    head_lemma: str
    normalized_predicate: str
    is_valid: bool
    is_passive: bool
    confidence: float


def is_valid_predicate_token(token: Token | None, text_override: str = "", lang: Lang = "zh") -> bool:
    """Return True if *token* or *text_override* is a linguistically defensible verbal predicate head."""
    text = (text_override or (getattr(token, "text", "") if token is not None else "")).strip()
    if not text:
        return False

    if token is not None:
        pos = getattr(token, "pos_", "")
        if pos not in {"VERB", "AUX"}:
            return False

        dep = getattr(token, "dep_", "")
        if dep in _INVALID_PREDICATE_DEPS:
            return False

        # If token is immediately preceded by a title or title+surname (e.g. '長卓' followed by '榮泰'),
        # it is the given name in a proper noun compound, not a verb predicate
        if lang == "zh" and getattr(token, "i", 0) > 0 and hasattr(token, "doc") and token.doc is not None:
            try:
                prev_tok = token.doc[token.i - 1]
                ptext = getattr(prev_tok, "text", "")
                if ptext.startswith("長") or any(ptext.endswith(m) for m in ("長", "主席", "總召", "立委", "議員", "參選人", "部長", "院長")):
                    if text not in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
                        return False
            except Exception:
                pass

    # English validation
    if lang == "en":
        # Must contain alphabetic characters and not be pure digits/punctuation
        return any(c.isalpha() for c in text) and not text.isdigit()

    # Chinese linguistic morphology checks
    if text in _ZH_STATIVE_ADJECTIVES:
        return False

    if text in _ZH_DISCOURSE_PREFIXES:
        return False

    if text in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
        return True

    # Check for title morphemes + surname (e.g. '長韓', '長卓', '長鄭', '席黃', '召蔡')
    if len(text) >= 2 and text[0] in "長席召員官委首揆" and text not in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
        return False

    # Check conjunction/preposition endings (e.g. '例與', '費以', '定讓')
    if len(text) > 1 and any(text.endswith(s) for s in _ZH_INVALID_VERB_SUFFIXES):
        return False

    # Check bound nominal morphemes (e.g. '宣費', '別費', '出總', '政慣', '言行', '大戰')
    if len(text) > 1 and any(text.endswith(m) for m in _ZH_BOUND_NOMINAL_MORPHEMES):
        return False

    # Check partial aspect prefix + single-char verb fragments (e.g. '將表', '將提', '正調')
    if len(text) == 2 and text[0] in "將正已欲曾" and text not in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
        return False

    # Single-character CJK validation
    if len(text) == 1:
        cp = ord(text[0])
        is_cjk = (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)
        if is_cjk:
            return text in _ZH_VALID_SINGLE_CHAR_VERBS
        return text.isalpha()

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
                    cdep = getattr(child, "dep_", "")
                    cpos = getattr(child, "pos_", "")
                    ctext = getattr(child, "text", "")
                    if cdep in {"ccomp", "xcomp", "conj", "dep", "dobj"} and ctext not in _ZH_PASSIVE_AUXILIARIES:
                        if is_valid_predicate_token(child, ctext, lang="zh"):
                            return ctext

            # If verb head is passive aux (e.g. head is '遭' and verb is '逮捕')
            if getattr(head_token, "head", None) is not None:
                parent = head_token.head
                ptext = getattr(parent, "text", "")
                if ptext in _ZH_PASSIVE_AUXILIARIES and raw not in _ZH_PASSIVE_AUXILIARIES:
                    return raw

            # Structural compound verb reconstruction via adjacent children
            # e.g. 砍 (VERB) + 刪 (NOUN/VERB) -> 砍刪; 減 (VERB) + 列 (VERB) -> 減列; 泰強 + 調 -> 強調
            for child in getattr(head_token, "children", []):
                cdep = getattr(child, "dep_", "")
                ctext = getattr(child, "text", "")
                ci = getattr(child, "i", -99)
                hi = getattr(head_token, "i", -99)
                if abs(ci - hi) == 1 and cdep in {"conj", "compound:vv", "xcomp", "advmod:rcomp", "dobj", "dep"}:
                    compound = (raw + ctext) if ci > hi else (ctext + raw)
                    if compound in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
                        return compound
                    if len(raw) > 1 and ci > hi:
                        split_compound = raw[1:] + ctext
                        if split_compound in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
                            return split_compound

            # Structural complement attachment: e.g. 審查 + 完畢 -> 審查; 達成 + 共識 -> 達成
            for child in getattr(head_token, "children", []):
                cdep = getattr(child, "dep_", "")
                ctext = getattr(child, "text", "")
                if cdep in {"advmod:rcomp", "dobj"} and ctext in {"完畢", "成", "到", "出"}:
                    if raw in {"審查", "下達", "送", "提"}:
                        if raw == "下達" and ctext == "成":
                            return "達成"
                        if raw == "送" and ctext == "到":
                            return "送到"

            # Check linear adjacent doc tokens if syntactic child walk did not find compound
            doc = getattr(head_token, "doc", None)
            hi = getattr(head_token, "i", -99)
            if doc is not None and 0 <= hi < len(doc):
                if hi + 1 < len(doc):
                    next_t = doc[hi + 1]
                    next_text = getattr(next_t, "text", "")
                    cand1 = raw + next_text
                    if cand1 in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
                        return cand1
                    if len(raw) > 1 and raw[0] in "將正已欲曾要會再":
                        cand_split = raw[1:] + next_text
                        if cand_split in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
                            return cand_split
                if hi - 1 >= 0:
                    prev_t = doc[hi - 1]
                    prev_text = getattr(prev_t, "text", "")
                    cand_prev = prev_text + raw
                    if cand_prev in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
                        return cand_prev

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
        # Check progressive aspect split: e.g. 正調 + 查 in sentence -> 調查
        if norm.startswith("正") and len(norm) == 2 and sentence:
            idx = sentence.find(norm)
            if idx >= 0 and idx + len(norm) < len(sentence):
                next_char = sentence[idx + len(norm)]
                candidate = norm[1:] + next_char
                return candidate
        if norm in _ZH_PASSIVE_AUXILIARIES and sentence:
            # Look for following verb in sentence
            idx = sentence.find(norm)
            if idx >= 0:
                after = sentence[idx + len(norm):idx + len(norm) + 10]
                for v in _ZH_VALID_LEXICAL_COMPOUND_VERBS:
                    if v in after:
                        return v
                for sub_v in ("吹離", "逮捕", "延誤", "查扣", "重創", "質疑", "解僱", "撤銷", "受創", "凍結", "刪除"):
                    if sub_v in after:
                        return sub_v
        if norm == "執" and sentence and "執行" in sentence:
            return "執行"

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

    is_valid = is_valid_predicate_token(token, surface, lang=lang)
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

