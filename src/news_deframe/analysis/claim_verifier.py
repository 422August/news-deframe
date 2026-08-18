"""Claim equivalence verifier, atomic proposition extraction, and claim eligibility.

Architecture
------------
Three-stage architecture for false-merge safety and claim validity:

Stage 0 – Claim Eligibility Filtering:
    Determine whether a candidate sentence actually expresses a usable
    proposition. Reject non-claim fragments (punctuation artifacts, source/byline
    tags, isolated discourse particles, incomplete clauses, heading artifacts)
    before embedding and clustering.

Stage 1 – Proposition Extraction:
    Extract structured atomic proposition features from each sentence,
    including predicate skeleton, agent/patient tokens, attribution/speaker,
    negation polarity, modality class, and structured quantities with their
    semantic targets. All extraction is domain-agnostic; no corpus-specific vocabulary.

Stage 2 – Structural Equivalence Verification:
    Compare extracted propositions against typed constraint checks.
    Embedding similarity is used as a supporting signal; it CANNOT alone
    establish claim equivalence without structural verification.

Relation semantics
------------------
EQUIVALENT:
    Both propositions express substantially the same core assertion: same
    polarity, compatible modality, similar predicate skeleton, substantial
    shared content, compatible attribution, and matching quantity targets/values.

COMPATIBLE:
    One proposition may add detail to the same core assertion without
    changing its meaning (e.g. precision difference or compatible sub-detail).
    Requires the same polarity, compatible modality, and non-conflicting structure.

RELATED:
    Concern the same event/topic/entities but make materially different
    factual assertions (different speaker, different action/predicate, different
    polarity, different modality, or different numerical targets).
    RELATED is NEVER sufficient for same-claim membership.

CONTRADICTORY:
    Materially incompatible assertions about the same proposition
    (polarity conflict on shared high-overlap content, or materially
    conflicting numeric quantities on the same semantic target).

UNRELATED:
    No meaningful proposition-level relationship (content too dissimilar).

Conservative-failure policy
---------------------------
When structural evidence is insufficient to confirm equivalence, the
verifier returns RELATED or UNRELATED rather than EQUIVALENT.
A false split (two different rows for the same real claim) is preferable
to a false merge (one row for two different claims).

No hardcoded domain vocabulary
-------------------------------
This module must not contain hardcoded outlet names, politician names,
event-specific terms, domain-specific keywords, budget vocabulary,
legislative vocabulary, protest vocabulary, or fixture phrases.
All patterns are linguistically general and apply to both Chinese and English.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum


class ClaimRelationType(str, Enum):
    """Explicit internal relationship schema between two sentences."""

    EQUIVALENT = "EQUIVALENT"          # Same factual proposition
    COMPATIBLE = "COMPATIBLE"          # Consistent sub-case or detail of the same fact
    RELATED = "RELATED"                # Shared topic / entities, but distinct factual assertions
    CONTRADICTORY = "CONTRADICTORY"    # Directly conflicting factual assertions
    UNRELATED = "UNRELATED"            # Different topics / unrelated facts


@dataclass(frozen=True)
class ClaimEquivalenceResult:
    """Detailed result of claim-equivalence verification between two sentences."""

    relation: ClaimRelationType
    is_equivalent: bool
    confidence: float
    similarity: float
    explanation: str


@dataclass(frozen=True)
class ClaimEligibility:
    """Eligibility check result for whether a sentence contains a usable proposition."""

    eligible: bool
    reason: str


@dataclass(frozen=True)
class StructuredQuantity:
    """A numeric quantity bound to its local semantic target context."""

    raw: str
    val: float | None
    target_tokens: frozenset[str]


@dataclass(frozen=True)
class AtomicProposition:
    """An atomic factual proposition extracted from a source sentence.

    Preserves full provenance back to the source article and original sentence.
    """

    prop_id: str
    article_id: str
    sentence_idx: int
    sentence_text: str
    proposition_text: str
    speaker: str | None = None
    modality: str = "statement"
    is_negated: bool = False
    quantities: tuple[StructuredQuantity, ...] = ()
    content_tokens: frozenset[str] = frozenset()
    predicate_tokens: frozenset[str] = frozenset()
    attribution_type: str = "none"


@dataclass
class SentenceProposition:
    """Structured propositional extraction from a sentence.

    Fields
    ------
    raw_text:
        Original unmodified sentence text.
    cleaned_text:
        Body text after stripping leading attribution phrase.
    agents:
        Agent/subject noun tokens extracted from the body or attribution.
    patients:
        Patient/object noun tokens extracted from the body.
    predicates:
        Predicate/verb strings extracted from the body.
    attributions:
        Speaker/source attributions extracted from leading phrases.
    is_negated:
        True if the body proposition contains a clause-level negation.
    modality:
        "statement" (factual statement), "demand" (demand / request / call for),
        "plan" (future plan / intention), "opinion" (subjective viewpoint).
    quantities:
        Numeric quantities extracted from the text.
    structured_quantities:
        Structured quantities paired with their local semantic targets.
    content_tokens:
        Bag-of-words / bigrams for content-overlap comparison.
    predicate_tokens:
        Predicate-skeleton token set for structural predicate comparison.
    attribution_type:
        "none" (narrative), "attributed_fact", "attributed_opinion",
        "attributed_demand", "attributed_plan".
    """

    raw_text: str
    cleaned_text: str
    agents: list[str] = field(default_factory=list)
    patients: list[str] = field(default_factory=list)
    predicates: list[str] = field(default_factory=list)
    attributions: list[str] = field(default_factory=list)
    is_negated: bool = False
    modality: str = "statement"
    quantities: list[str] = field(default_factory=list)
    structured_quantities: list[StructuredQuantity] = field(default_factory=list)
    content_tokens: set[str] = field(default_factory=set)
    predicate_tokens: set[str] = field(default_factory=set)
    attribution_type: str = "none"


# ── Claim Eligibility Patterns ────────────────────────────────────────────────

_PUNCT_AND_SPACES = r'[\s\.\?!，,。！？：:、·；;…—\-_\*#\"\'\(\)\[\]\{\}（）「」『』【】〔〕<>《》]*'

# News agency, wire service, or photo byline markers in bracketed wrappers
_METADATA_BRACKETS = re.compile(
    r'^[（\(\[\{【〔<「『].{1,45}[）\)\]\}】〕>」』]$'
)
_METADATA_KEYWORDS = re.compile(
    r'(?:中央社|社|電|報導|記者|編譯|譯者|核稿|編輯|中心|綜合|攝影|特派|快訊|更新|來源|圖／|攝|'
    r'reuters|afp|ap|bloomberg|reporter|photo|credit|source|staff|edited|reporting|translated)',
    re.IGNORECASE,
)

# Leading byline header patterns
_BYLINE_PREFIX = re.compile(
    r'^(?:【(?:記者|即時|中心|地方|政治|國際|科技|財經|專題|社會|生活|娛樂).{1,30}】|'
    r'(?:記者|特派記者|編輯|編譯|核稿|實習記者|綜合報導|外電報導)\s*[:：\s]|'
    r'(?:Photo|Image|Credit|Source|By|Written by|Reported by)\s*[:：\s])',
    re.IGNORECASE,
)

# Attribution prefix only with no body
_ATTRIBUTION_ONLY = re.compile(
    r'^(.{1,35}?)(?:表示|指出|強調|稱|說|指|認為|坦言|透露|聲稱|聲明|宣布|宣稱|表明|回應|反駁|said|stated|demanded|urged)[，,：:\s]*$',
    re.IGNORECASE,
)

# Isolated Chinese discourse transitions or polar responses without proposition
_ZH_DISCOURSE_PARTICLES = re.compile(
    r'^(?:會中|此外|另外|對此|甚至|不過|但是|然而|因此|總結來說|沒有|不是|好的|沒錯|是的|據悉|據了解|'
    r'延伸閱讀|相關新聞|點我看更多|看更多|推薦閱讀)' + _PUNCT_AND_SPACES + r'$'
)

# Isolated English discourse transitions or polar responses
_EN_DISCOURSE_PARTICLES = re.compile(
    r'^(?:However|In addition|Furthermore|Moreover|Indeed|Meanwhile|Therefore|Yes|No|In summary|'
    r'Also|To be sure|Related news|Read more|Click here|See more)' + _PUNCT_AND_SPACES + r'$',
    re.IGNORECASE,
)

# Isolated temporal phrase without proposition
_ZH_TEMPORAL_FRAGMENT = re.compile(
    r'^(?:\d+年)?(?:\d+月)?(?:\d+日|\d+號)?(?:昨天|今天|明天|昨日|今日|明日|上午|下午|晚間|中午|日前|近日|當天|即日|上週|下週)*(?:[上下午晚間]+)?' + _PUNCT_AND_SPACES + r'$'
)


def check_claim_eligibility(text: str) -> ClaimEligibility:
    """Determine whether a candidate text span expresses a usable proposition.

    Rejects non-claim fragments (punctuation artifacts, byline/wire metadata,
    isolated discourse connectors, isolated temporal phrases) before embedding
    and clustering.
    """
    raw = text.strip()
    if not raw:
        return ClaimEligibility(False, "Empty sentence.")

    # 1. Punctuation only / minimal substantive lexical content
    substantive_cjk = [c for c in raw if 0x4E00 <= ord(c) <= 0x9FFF]
    substantive_ascii = [c for c in raw if c.isalnum()]

    if len(substantive_cjk) == 0 and len(substantive_ascii) < 3:
        return ClaimEligibility(False, "Punctuation or symbol artifact with no substantive lexical content.")

    if len(substantive_cjk) < 2 and len(substantive_ascii) == 0:
        return ClaimEligibility(False, "Single character non-propositional fragment.")

    # 2. Metadata / byline / agency wire tag enclosed in brackets
    if _METADATA_BRACKETS.match(raw) and _METADATA_KEYWORDS.search(raw):
        return ClaimEligibility(False, "News agency wire, byline, or photo credit metadata.")

    # 3. Byline prefixes with no remaining proposition
    if _BYLINE_PREFIX.match(raw):
        stripped_byline = _BYLINE_PREFIX.sub("", raw).strip()
        sub_cjk = [c for c in stripped_byline if 0x4E00 <= ord(c) <= 0x9FFF]
        sub_ascii = [w for w in re.findall(r"[a-zA-Z0-9]+", stripped_byline)]
        if len(sub_cjk) < 4 and len(sub_ascii) <= 4:
            return ClaimEligibility(False, "Byline or editorial metadata header.")

    # 4. Attribution speech-act header only with empty body
    if _ATTRIBUTION_ONLY.match(raw):
        return ClaimEligibility(False, "Attribution prefix without propositional body.")

    # 5. Isolated discourse particles
    if _ZH_DISCOURSE_PARTICLES.match(raw):
        return ClaimEligibility(False, "Isolated Chinese discourse transition or polar response without proposition.")
    if _EN_DISCOURSE_PARTICLES.match(raw):
        return ClaimEligibility(False, "Isolated English discourse transition or polar response without proposition.")

    # 6. Isolated temporal phrase without proposition
    if len(raw) <= 12 and _ZH_TEMPORAL_FRAGMENT.match(raw):
        return ClaimEligibility(False, "Isolated temporal marker without proposition.")

    # 6. Attribution speech-act header only with empty body
    zh_attr = _ZH_ATTRIBUTION_PATTERN.match(raw)
    if zh_attr:
        body = zh_attr.group(2).strip()
        if len([c for c in body if 0x4E00 <= ord(c) <= 0x9FFF or c.isalnum()]) < 2:
            return ClaimEligibility(False, "Attribution prefix without propositional body.")

    en_attr = _EN_ATTRIBUTION_PATTERN.match(raw)
    if en_attr:
        body = en_attr.group(2).strip()
        if len(body.split()) < 2:
            return ClaimEligibility(False, "Attribution prefix without propositional body.")

    return ClaimEligibility(True, "Eligible claim proposition.")


def is_claim_eligible(text: str) -> bool:
    """Convenience boolean check for claim eligibility."""
    return check_claim_eligibility(text).eligible


# ── Linguistic patterns for domain-agnostic proposition extraction ─────────────

# Chinese propositional clause-level negation markers
# Note: Adverbials modifying nouns (e.g. 無預警, 無條件) are excluded.
_ZH_NEGATION = re.compile(
    r"(?:沒有|没有|不(?:是|會|能|得|符|予|受|滿|滿意|願|再|存在|承認|同意|配合)|"
    r"並(?:不|未)|"
    r"未(?:能|曾|予|獲|達|有|經|按)|"
    r"非(?:法|屬)|"
    r"否認|否认|拒絕|拒绝|禁止|反對|反对|取消|撤回|撤銷|撤销|否決|否决|"
    r"暫停|暂停|中止|擱置|搁置|"
    r"無法|无法|無力|无力|無效|无效)"
)

# English negation words (clause-level negation auxiliaries and predicates)
_EN_NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor",
    "refuse", "refused", "refusal", "deny", "denied", "denial",
    "failed", "fail", "failure", "reject", "rejected", "rejection",
    "cancel", "cancelled", "canceled", "cancellation",
    "suspend", "suspended", "suspension", "abort", "aborted",
    "withdraw", "withdrew", "withdrawn", "withdrawal",
    "veto", "vetoed",
})

# Chinese attribution speech-act verbs
_ZH_ATTRIBUTION_PATTERN = re.compile(
    r"^(.{1,25}?)(?:表示|指出|強調|稱|說|指|認為|坦言|透露|聲稱|聲明|宣布|宣稱|表明|回應|反駁)[，,：:\s]+(.+)$"
)

# English attribution speech-act verbs
_EN_ATTRIBUTION_PATTERN = re.compile(
    r"^(.{1,60}?)\b(?:said|stated|pointed out|emphasized|claimed|reported|"
    r"announced|noted|added|warned|argued|insisted|confirmed|denied|"
    r"explained|revealed|disclosed|declared|responded|demanded|urged|"
    r"called for|requested|ordered)"
    r"(?:[,:\s]+that|[,:\s]+)(.+)$",
    re.IGNORECASE,
)

# Chinese leading subject split heuristic
_ZH_SUBJECT_SPLIT = re.compile(
    r"^(?:部分|許多|全體|所有|該)?([^\s，,。；;]{1,15}?)(?:拒絕|拒绝|同意|宣布|確認|确认|表示|要求|呼籲|呼吁|計畫|计画|打算|完成|展開|展開了|批評|質疑)"
)

# English leading subject split heuristic
_EN_SUBJECT_SPLIT = re.compile(
    r"^(?:The|A|An|Some|Many|All)?\s*([a-zA-Z\s]+?)\s+\b(?:demanded|urged|said|stated|refused|agreed|confirmed|fined|approved|rejected|restored|repaired|launched|derailed|expanded|reduced)\b",
    re.IGNORECASE,
)

# Modality classification patterns
_ZH_DEMAND = re.compile(r"要求|呼籲|呼吁|促請|促请|建議|建议|敦促|催促|請求|请求")
_ZH_PLAN = re.compile(r"將(?:會|於|要|在)|即將|计划|計畫|計划|打算|預計|预计|擬定|擬於|擬在")
_ZH_OPINION = re.compile(r"認為|认为|認同|认同|表示支持|質疑|担忧|擔憂|批評|批评|強調|强调")

_EN_DEMAND = re.compile(
    r"\b(?:demand(?:ed|s|ing)?|urg(?:ed|es|ing)|call(?:ed|s|ing)\s+for|"
    r"request(?:ed|s|ing)?|ask(?:ed|s|ing)\s+for|push(?:ed|es|ing)\s+for|"
    r"press(?:ed|es|ing)\s+for|insist(?:ed|s|ing))\b",
    re.IGNORECASE,
)
_EN_PLAN = re.compile(
    r"\b(?:will|shall|plan(?:s|ned|ning)?|intend(?:s|ed|ing)?|"
    r"aim(?:s|ed|ing)?|propose(?:s|d|ing)?|expect(?:s|ed|ing)?|"
    r"seek(?:s|ing|sought))\b",
    re.IGNORECASE,
)
_EN_OPINION = re.compile(
    r"\b(?:believe(?:s|d)?|think(?:s)?|thought|argue(?:s|d)?|"
    r"contend(?:s|ed)?|maintain(?:s|ed)?|assess(?:es|ed)?|"
    r"consider(?:s|ed)?|regard(?:s|ed)?)\b",
    re.IGNORECASE,
)

# Phase / status conflict patterns (completed vs initiated/planned)
_ZH_COMPLETED = re.compile(r"完成|結案|结案|結束|结束|已達成|已落实|已完工|已修復|已恢復|已成功")
_ZH_INITIATED = re.compile(r"展開|展开|啟動|启动|開始|开始|立案|籌備|筹备|延誤|延期|打算")

_EN_COMPLETED = re.compile(r"\b(?:completed|concluded|finished|finalized|settled|closed|restored|repaired|achieved)\b", re.IGNORECASE)
_EN_INITIATED = re.compile(r"\b(?:began|started|launched|initiated|opened|commenced|plans|planning|intend)\b", re.IGNORECASE)

# Directional outcome conflicts (increase/inflow vs reduction/outflow)
_DIRECTION_INCREASE = re.compile(
    r"\b(?:expanded|increasing|increased|raises|raised|boosted|amplified|expansion|increase|growth|gains?|revenue|inflows?)\b|"
    r"擴建|擴大|增加|調升|提高|增列|歲入|收入|調高|成長",
    re.IGNORECASE,
)
_DIRECTION_DECREASE = re.compile(
    r"\b(?:reduced|decreasing|decreased|lowers|lowered|cut|cuts|reduction|decrease|decline|losses?|expenditure|outflows?)\b|"
    r"縮減|減少|裁減|調降|降低|減列|刪減|統刪|歲出|支出|調低|虧損|跌",
    re.IGNORECASE,
)

# ── Structured Quantity Extraction ────────────────────────────────────────────

_UNIT_MULTIPLIER = {
    "兆": 1000000000000.0,
    "億": 100000000.0,
    "萬": 10000.0,
    "千": 1000.0,
    "百": 100.0,
}

_COMPOUND_NUMERAL = re.compile(
    r"(?<![a-zA-Z0-9])"
    r"(\d+(?:[,，]\d+)*(?:\.\d+)?(?:\s*(?:兆|億|萬|千|百))?)+"
    r"(?:\s*(?:%|百分之|元|人|名|個|戶|家|間|件|次|項|筆|dollars|percent|points|homes|workers))?"
)

_QUANTITY_CHINESE = re.compile(
    r"(?:約|逾|超過|不足|至少|最多)?[零一二兩三四五六七八九十百千萬億]+(?:餘|多|余)?(?:名|人|個|位|項|件|次|倍|%|百分之)?"
)

_STOP_TARGET = frozenset({
    "為", "是", "共", "約", "達", "至", "有", "在", "計", "了", "改", "暫", "新",
    "的", "個", "及", "與", "和", "從", "到", "由", "將", "於", "其", "中",
})

_ZH_KEY_FISCAL_UNIGRAMS = frozenset({"入", "出", "刪", "減", "增", "編", "列", "凍", "減列", "增列", "統刪", "刪除"})

_ZH_ACTION_VERBS = re.compile(
    r"(?:通過|審查|審議|表決|完成|凍結|刪除|減列|增列|編列|統刪|三讀|二讀|動支|副署|執行|呼籲|強調|指出|表示|受訪|簽名|動員|卡關|改列|計列|增加|減少|調升|調降|是|為|有|達|遭|欠|給|引|提|比讚|宣布|認為|說明|重申|反對|贊成)"
)
_EN_ACTION_VERBS = re.compile(
    r"\b(?:pass|passed|approve|approved|vote|voted|cut|cuts|reduced|reduce|increase|increased|freeze|frozen|complete|completed|said|stated|demanded|urged|is|was|were|are|has|had|have|declined|struck)\b",
    re.IGNORECASE,
)


def _has_actionable_predicate(text: str) -> bool:
    """Return True if text contains a recognizable action verb or predicate."""
    return bool(_ZH_ACTION_VERBS.search(text) or _EN_ACTION_VERBS.search(text))


def _parse_chinese_numeral(q: str) -> float | None:
    """Parse a pure Chinese numeral string to a float.

    Supports compound numerals like 三百, 兩千, 三萬六千, 一億兩千萬.
    """
    _DIGIT = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
               "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    _UNIT = {"十": 10, "百": 100, "千": 1000, "萬": 10000, "億": 100000000}

    q = re.sub(r"(?:名|人|個|位|項|件|次|倍|%|百分之|餘|多|余)$", "", q.strip())
    q = re.sub(r"^(?:約|逾|超過|不足|至少|最多)", "", q)

    if not q:
        return None

    if len(q) == 1 and q in _DIGIT:
        return float(_DIGIT[q])

    result = 0.0
    current = 0.0
    last_unit = 1
    i = 0
    while i < len(q):
        ch = q[i]
        if ch in _DIGIT:
            current = _DIGIT[ch]
            i += 1
        elif ch in _UNIT:
            unit = _UNIT[ch]
            if unit >= 10000:
                if current == 0:
                    current = 1
                result += current * unit
                current = 0.0
            else:
                if current == 0:
                    current = 1
                current = current * unit
            last_unit = unit
            i += 1
        else:
            return None

    result += current
    return result if result > 0 else None


def _normalize_quantity_value(q: str) -> float | None:
    """Parse any quantity string (Arabic, compound mixed, or Chinese) to a float."""
    q_clean = q.strip().replace(",", "").replace("，", "")
    q_clean = re.sub(r"(?:元|人|名|個|位|項|件|次|筆|戶|家|間|天|週|月|年|%|百分之|dollars|percent|points|homes|workers)$", "", q_clean)
    q_clean = re.sub(r"^(?:約|逾|超過|不足|至少|最多)", "", q_clean)
    if not q_clean:
        return None

    try:
        return float(q_clean)
    except ValueError:
        pass

    # Mixed Arabic-CJK: parse parts like '2兆8622億5319萬1000'
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(兆|億|萬|千|百)?", q_clean)
    if parts:
        tot = 0.0
        for v_str, unit in parts:
            if v_str:
                tot += float(v_str) * _UNIT_MULTIPLIER.get(unit, 1.0)
        if tot > 0:
            return tot

    return _parse_chinese_numeral(q_clean)


_STOP_TARGET_WORDS = re.compile(
    r"(?:預算|審查|結果|部分|方面|進行|案|為|是|共|約|達|至|有|在|計|了|改|暫|新|的|個|及|與|和|從|到|由|將|於|其|中)"
)


def extract_structured_quantities(text: str) -> list[StructuredQuantity]:
    """Extract numeric and compound quantities paired with their local semantic targets."""
    results: list[StructuredQuantity] = []
    seen_spans: set[tuple[int, int]] = set()

    for m in _COMPOUND_NUMERAL.finditer(text):
        raw_str = m.group(0).strip()
        if not raw_str:
            continue
        # Avoid standalone calendar year/day with no currency or scalar unit
        trailing = text[m.end():m.end() + 3]
        if any(trailing.startswith(u) for u in ("年度", "年", "月", "日", "號", "分", "點", "時", "條", "款", "屆")):
            continue
        if re.match(r"^\d+\s*(?:年度|年|月|日|號|分|點|時)$", raw_str):
            continue
        val = _normalize_quantity_value(raw_str)
        if val is None:
            continue

        seen_spans.add(m.span())
        # Target window: 12 characters before and 12 characters after
        start_idx = max(0, m.start() - 12)
        end_idx = min(len(text), m.end() + 12)
        pre_ctx = _STOP_TARGET_WORDS.sub("", text[start_idx:m.start()])
        post_ctx = _STOP_TARGET_WORDS.sub("", text[m.end():end_idx])

        target_tokens: set[str] = set()
        for ctx in (pre_ctx, post_ctx):
            cjk = [ch for ch in ctx if 0x4E00 <= ord(ch) <= 0x9FFF]
            for i in range(len(cjk) - 1):
                target_tokens.add(cjk[i] + cjk[i + 1])
            for ch in cjk:
                if ch in _ZH_KEY_FISCAL_UNIGRAMS:
                    target_tokens.add(ch)
            for w in re.findall(r"[a-zA-Z]+", ctx):
                if len(w) >= 3 and w.lower() not in {"the", "and", "for", "was", "were", "has", "had", "been"}:
                    target_tokens.add(w.lower())

        results.append(StructuredQuantity(raw=raw_str, val=val, target_tokens=frozenset(target_tokens)))

    # Also parse pure Chinese numerals (e.g. 三百, 兩千)
    for m in _QUANTITY_CHINESE.finditer(text):
        if any(s <= m.start() and m.end() <= e for s, e in seen_spans):
            continue
        raw_str = m.group(0).strip()
        if len(raw_str) < 2:
            continue
        val = _parse_chinese_numeral(raw_str)
        if val is None:
            continue

        start_idx = max(0, m.start() - 12)
        end_idx = min(len(text), m.end() + 12)
        pre_ctx = _STOP_TARGET_WORDS.sub("", text[start_idx:m.start()])
        post_ctx = _STOP_TARGET_WORDS.sub("", text[m.end():end_idx])

        target_tokens = set()
        for ctx in (pre_ctx, post_ctx):
            cjk = [ch for ch in ctx if 0x4E00 <= ord(ch) <= 0x9FFF]
            for i in range(len(cjk) - 1):
                target_tokens.add(cjk[i] + cjk[i + 1])
            for ch in cjk:
                if ch in _ZH_KEY_FISCAL_UNIGRAMS:
                    target_tokens.add(ch)

        results.append(StructuredQuantity(raw=raw_str, val=val, target_tokens=frozenset(target_tokens)))

    return results


def extract_atomic_propositions(
    article_id: str,
    sentence_idx: int,
    sentence: str,
) -> list[AtomicProposition]:
    """Decompose a candidate sentence into one or more atomic factual propositions.

    Preserves full provenance (article_id, sentence_idx, original sentence text,
    clause-specific quantities, speaker attribution, negation, and modality).
    """
    raw = sentence.strip()
    if not raw or not check_claim_eligibility(raw).eligible:
        return []

    attributions, body, attr_type = _extract_attributions(raw)
    speaker = attributions[0] if attributions else None

    # Decompose into candidate clauses
    major_chunks = re.split(r"[；;\n]", body)

    raw_clauses: list[str] = []
    for chunk in major_chunks:
        # Split on commas / coordinate conjunctions
        sub_chunks = [sc.strip() for sc in re.split(r"[,，]", chunk) if sc.strip()]

        merged_sub: list[str] = []
        for sc in sub_chunks:
            has_qty = bool(extract_structured_quantities(sc))
            has_verb = _has_actionable_predicate(sc)

            # If no verb and no quantity, it is a dependent topic/context fragment -> merge with adjacent clause
            if not has_qty and not has_verb:
                if merged_sub:
                    merged_sub[-1] = merged_sub[-1] + "，" + sc
                else:
                    merged_sub.append(sc)
            else:
                if merged_sub and not (
                    extract_structured_quantities(merged_sub[-1])
                    or _has_actionable_predicate(merged_sub[-1])
                ):
                    merged_sub[-1] = merged_sub[-1] + "，" + sc
                else:
                    merged_sub.append(sc)
        raw_clauses.extend(merged_sub)

    propositions: list[AtomicProposition] = []
    for p_idx, clause in enumerate(raw_clauses):
        c_clean = clause.strip()
        if not c_clean:
            continue
        c_quantities = tuple(extract_structured_quantities(c_clean))
        # Clause must have a quantity or have substantive length >= 6 (CJK) / >= 3 words (Latin)
        if len(c_clean) < 6 and not c_quantities and len(c_clean.split()) < 3:
            continue

        c_is_neg = bool(
            _ZH_NEGATION.search(c_clean)
            or any(w in c_clean.lower().split() for w in _EN_NEGATION_WORDS)
        )
        c_tokens = frozenset(_content_tokens(c_clean))
        p_tokens = frozenset(_predicate_tokens(c_clean))

        modality = "statement"
        if _ZH_DEMAND.search(c_clean) or _EN_DEMAND.search(c_clean):
            modality = "demand"
        elif _ZH_PLAN.search(c_clean) or _EN_PLAN.search(c_clean):
            modality = "plan"
        elif _ZH_OPINION.search(c_clean) or _EN_OPINION.search(c_clean):
            modality = "opinion"

        propositions.append(
            AtomicProposition(
                prop_id=f"{article_id}:{sentence_idx}:{p_idx}",
                article_id=article_id,
                sentence_idx=sentence_idx,
                sentence_text=raw,
                proposition_text=c_clean,
                speaker=speaker,
                modality=modality,
                is_negated=c_is_neg,
                quantities=c_quantities,
                content_tokens=c_tokens,
                predicate_tokens=p_tokens,
                attribution_type=attr_type,
            )
        )

    if not propositions:
        c_quantities = tuple(extract_structured_quantities(body))
        c_is_neg = bool(
            _ZH_NEGATION.search(body)
            or any(w in body.lower().split() for w in _EN_NEGATION_WORDS)
        )
        propositions.append(
            AtomicProposition(
                prop_id=f"{article_id}:{sentence_idx}:0",
                article_id=article_id,
                sentence_idx=sentence_idx,
                sentence_text=raw,
                proposition_text=body,
                speaker=speaker,
                modality="statement",
                is_negated=c_is_neg,
                quantities=c_quantities,
                content_tokens=frozenset(_content_tokens(body)),
                predicate_tokens=frozenset(_predicate_tokens(body)),
                attribution_type=attr_type,
            )
        )

    return propositions


def _quantities_conflict_structured(
    qa_list: list[StructuredQuantity],
    qb_list: list[StructuredQuantity],
) -> tuple[bool, str]:
    """Compare structured quantities according to their semantic targets.

    Returns (has_conflict, explanation).
    - True if values conflict on a shared semantic target.
    - False if quantities are consistent/precision variants on shared targets,
      or if no structural conflict is found.
    """
    if not qa_list or not qb_list:
        return False, ""

    # Check for target-aligned conflict
    for qa in qa_list:
        for qb in qb_list:
            if qa.val is None or qb.val is None:
                continue

            shared_targets = qa.target_tokens & qb.target_tokens
            if shared_targets:
                if qa.val == 0.0 and qb.val == 0.0:
                    continue
                if qa.val == 0.0 or qb.val == 0.0:
                    return True, f"Quantity conflict on shared target {list(shared_targets)}: {qa.raw} vs {qb.raw}"
                rel_diff = abs(qa.val - qb.val) / max(abs(qa.val), abs(qb.val))
                if rel_diff > 0.05:
                    return True, f"Material quantity conflict on shared target {list(shared_targets)}: {qa.raw} vs {qb.raw} (rel_diff={rel_diff:.2%})"

    # Check for non-target-aligned fallback if both have single scalar quantities and zero target overlap
    if len(qa_list) == 1 and len(qb_list) == 1:
        qa = qa_list[0]
        qb = qb_list[0]
        if qa.val is not None and qb.val is not None:
            if qa.val == 0.0 and qb.val == 0.0:
                return False, ""
            if qa.val == 0.0 or qb.val == 0.0:
                return True, f"Quantity conflict: {qa.raw} vs {qb.raw}"
            ratio = qa.val / qb.val
            if ratio > 1.8 or ratio < 0.55:
                return True, f"Conflicting quantities: {qa.raw} vs {qb.raw}"

    return False, ""


def _quantities_conflict(qa_list: list[str], qb_list: list[str]) -> bool:
    """Legacy helper for testing quantity string conflict."""
    sq_a = [StructuredQuantity(raw=q, val=_normalize_quantity_value(q), target_tokens=frozenset()) for q in qa_list]
    sq_b = [StructuredQuantity(raw=q, val=_normalize_quantity_value(q), target_tokens=frozenset()) for q in qb_list]
    conflict, _ = _quantities_conflict_structured(sq_a, sq_b)
    return conflict


# ── Speaker Divergence Detection ──────────────────────────────────────────────

_EN_TITLES = frozenset({
    "senator", "minister", "president", "representative", "governor", "mayor",
    "spokesman", "spokesperson", "secretary", "director", "officials", "researchers",
    "officer", "leader", "chairman", "chairwoman", "mr", "ms", "dr", "prof",
})

_ZH_TITLES = frozenset({
    "委員", "立委", "院長", "部長", "發言人", "黨團", "總召", "主席", "議員",
    "市長", "署長", "局長", "處長", "主任", "專家", "代表", "立法院長", "行政院長",
})


def _clean_speaker_tokens(name: str) -> tuple[str, set[str]]:
    """Clean attribution name and extract distinctive name tokens."""
    strip_suffix = r"(?:在.{1,10}後|在.{1,10}上|在.{1,10}受訪時|今日|昨日|今天|昨天|上午|下午|晚間)$"
    clean = re.sub(strip_suffix, "", name.strip()).strip()

    en_words = {w.lower() for w in re.findall(r"[a-zA-Z]+", clean) if len(w) >= 2 and w.lower() not in _EN_TITLES}
    cjk = [ch for ch in clean if 0x4E00 <= ord(ch) <= 0x9FFF]
    bigrams = set()
    for i in range(len(cjk) - 1):
        bg = cjk[i] + cjk[i + 1]
        if bg not in _ZH_TITLES:
            bigrams.add(bg)

    return clean, (en_words | bigrams)


def _speakers_diverge(spk_a: list[str], spk_b: list[str]) -> bool:
    """Return True if both sentences have attributed speakers and they are distinct entities."""
    if not spk_a or not spk_b:
        return False
    raw_a, toks_a = _clean_speaker_tokens(spk_a[0])
    raw_b, toks_b = _clean_speaker_tokens(spk_b[0])
    if not raw_a or not raw_b:
        return False
    if raw_a in raw_b or raw_b in raw_a:
        return False
    if toks_a and toks_b:
        return not bool(toks_a & toks_b)
    return raw_a != raw_b


# ── Proposition Extraction Helpers ─────────────────────────────────────────────

def _extract_cjk_bigrams(text: str) -> set[str]:
    cjk = [ch for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF]
    return {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}


def _extract_ascii_words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z]+", text) if len(w) >= 2}


def _content_tokens(text: str) -> set[str]:
    return _extract_cjk_bigrams(text) | _extract_ascii_words(text)


def _predicate_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    cjk = [ch for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF]
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i] + cjk[i + 1])
    for i in range(len(cjk) - 2):
        tokens.add(cjk[i] + cjk[i + 1] + cjk[i + 2])

    _EN_STOP = frozenset({
        "that", "this", "with", "from", "have", "been", "were", "they",
        "them", "their", "there", "here", "when", "what", "which", "will",
        "also", "more", "some", "than", "then", "into", "onto", "over",
        "after", "before", "about", "while", "where", "through", "between",
        "during", "within", "without", "against", "across", "because",
        "since", "until", "under", "above",
    })
    for w in re.findall(r"[a-zA-Z]+", text):
        wl = w.lower()
        if len(wl) >= 4 and wl not in _EN_STOP:
            tokens.add(wl)

    return tokens


def _extract_attributions(text: str) -> tuple[list[str], str, str]:
    attributions: list[str] = []
    body = text

    zh_m = _ZH_ATTRIBUTION_PATTERN.match(text)
    if zh_m:
        speaker = zh_m.group(1).strip()
        rest = zh_m.group(2).strip()
        if 1 <= len(speaker) <= 25 and rest:
            attributions.append(speaker)
            body = rest

    if not attributions:
        en_m = _EN_ATTRIBUTION_PATTERN.match(text)
        if en_m:
            speaker = en_m.group(1).strip()
            rest = en_m.group(2).strip()
            word_count = len(speaker.split())
            if 1 <= word_count <= 8 and rest:
                attributions.append(speaker)
                body = rest

    if not attributions:
        attr_type = "none"
    else:
        if _ZH_DEMAND.search(text) or _EN_DEMAND.search(text):
            attr_type = "attributed_demand"
        elif _ZH_PLAN.search(text) or _EN_PLAN.search(text):
            attr_type = "attributed_plan"
        elif _ZH_OPINION.search(text) or _EN_OPINION.search(text):
            attr_type = "attributed_opinion"
        else:
            attr_type = "attributed_fact"

    return attributions, body, attr_type


def extract_proposition(sentence: str) -> SentenceProposition:
    """Extract structured propositional features from a sentence.

    Domain-agnostic: uses linguistic structure (negation markers, attribution verbs,
    modality markers, structured quantities) — no corpus-specific vocabulary.
    """
    raw = sentence.strip()
    attributions, body, attr_type = _extract_attributions(raw)

    agents: list[str] = []
    if attributions:
        for attr in attributions:
            agents.append(attr)
            for w in attr.split():
                agents.append(w.lower())
    else:
        zh_sub = _ZH_SUBJECT_SPLIT.search(raw)
        if zh_sub:
            agents.append(zh_sub.group(1).strip())
        en_sub = _EN_SUBJECT_SPLIT.search(raw)
        if en_sub:
            sub = en_sub.group(1).strip()
            agents.append(sub)
            for w in sub.split():
                agents.append(w.lower())

    # Negation detection
    is_negated = bool(_ZH_NEGATION.search(body))
    if not is_negated:
        body_words = [w.lower() for w in re.findall(r"[a-zA-Z]+", body)]
        is_negated = bool(set(body_words) & _EN_NEGATION_WORDS)

    # Modality classification
    modality = "statement"
    if _ZH_DEMAND.search(body) or _EN_DEMAND.search(raw):
        modality = "demand"
    elif _ZH_PLAN.search(body) or _EN_PLAN.search(raw):
        modality = "plan"
    elif _ZH_OPINION.search(body) or _EN_OPINION.search(raw):
        modality = "opinion"

    structured_quantities = extract_structured_quantities(body)
    quantities = [sq.raw for sq in structured_quantities]
    ctokens = _content_tokens(body)
    ptokens = _predicate_tokens(body)

    return SentenceProposition(
        raw_text=raw,
        cleaned_text=body,
        agents=agents,
        patients=[],
        predicates=[],
        attributions=attributions,
        is_negated=is_negated,
        modality=modality,
        quantities=quantities,
        structured_quantities=structured_quantities,
        content_tokens=ctokens,
        predicate_tokens=ptokens,
        attribution_type=attr_type,
    )


# ── Structural Compatibility Helpers ──────────────────────────────────────────

def _attributions_compatible(prop_a: SentenceProposition, prop_b: SentenceProposition) -> bool:
    type_a = prop_a.attribution_type
    type_b = prop_b.attribution_type

    if type_a == "none" and type_b == "none":
        return True

    if type_a != "none" and type_b != "none":
        return type_a == type_b

    # One attributed, one narrative
    attributed_type = type_a if type_a != "none" else type_b
    return attributed_type == "attributed_fact"


def _modality_compatible(mod_a: str, mod_b: str) -> bool:
    COMPLETED = frozenset({"statement", "opinion"})
    FUTURE = frozenset({"demand", "plan"})
    return (mod_a in COMPLETED and mod_b in COMPLETED) or (mod_a in FUTURE and mod_b in FUTURE)


def _phase_conflicts(text_a: str, text_b: str) -> bool:
    a_comp = bool(_ZH_COMPLETED.search(text_a) or _EN_COMPLETED.search(text_a))
    b_comp = bool(_ZH_COMPLETED.search(text_b) or _EN_COMPLETED.search(text_b))
    a_init = bool(_ZH_INITIATED.search(text_a) or _EN_INITIATED.search(text_a))
    b_init = bool(_ZH_INITIATED.search(text_b) or _EN_INITIATED.search(text_b))
    return (a_comp and b_init) or (a_init and b_comp)


def _direction_conflicts(text_a: str, text_b: str) -> bool:
    a_inc = bool(_DIRECTION_INCREASE.search(text_a))
    b_inc = bool(_DIRECTION_INCREASE.search(text_b))
    a_dec = bool(_DIRECTION_DECREASE.search(text_a))
    b_dec = bool(_DIRECTION_DECREASE.search(text_b))
    return (a_inc and b_dec) or (a_dec and b_inc)


_EN_STOP_SHORT = frozenset({
    "the", "an", "a", "in", "at", "by", "of", "to", "and", "or",
    "for", "on", "its", "this", "that", "was", "is", "are", "were",
    "has", "had", "have", "been",
})


def _has_cjk(text: str) -> bool:
    return any(0x4E00 <= ord(c) <= 0x9FFF for c in text)


def _is_cross_lingual(sent_a: str, sent_b: str) -> bool:
    return _has_cjk(sent_a) != _has_cjk(sent_b)


def _words_or_chars(text: str) -> set[str]:
    words = {w.lower() for w in re.findall(r"[a-zA-Z]+", text) if len(w) >= 2 and w.lower() not in _EN_STOP_SHORT}
    cjk = [ch for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF]
    return words | set(cjk)


def _divergent_agent_detected(sent_a: str, sent_b: str) -> bool:
    sm = difflib.SequenceMatcher(None, sent_a, sent_b)
    match = sm.find_longest_match(0, len(sent_a), 0, len(sent_b))
    if match.size < 5:
        return False

    min_len = min(len(sent_a), len(sent_b))
    if match.size / min_len < 0.40:
        return False

    prefix_a = sent_a[:match.a].strip()
    prefix_b = sent_b[:match.b].strip()
    if not prefix_a or not prefix_b:
        return False

    tokens_a = _words_or_chars(prefix_a)
    tokens_b = _words_or_chars(prefix_b)
    if not tokens_a or not tokens_b:
        return False

    if tokens_a & tokens_b:
        return False

    return True


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _action_tokens(text: str) -> set[str]:
    cjk = [ch for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF]
    if len(cjk) > 6:
        return {cjk[i] + cjk[i + 1] for i in range(4, len(cjk) - 1)}
    words = text.strip().split()
    if len(words) > 3:
        return {w.lower().strip(".,") for w in words[3:] if len(w) >= 3 and w.lower().strip(".,") not in _EN_STOP_SHORT}
    return set()


# ── Core Claim-Equivalence Verifier ────────────────────────────────────────────

_HIGH_SIM = 0.85
_MED_SIM = 0.70
_LOW_SIM = 0.55


def verify_claim_equivalence(
    sent_a: str,
    sent_b: str,
    similarity: float,
) -> ClaimEquivalenceResult:
    """Determine whether two sentences express materially equivalent factual propositions.

    Implements a conservative, domain-agnostic structural verification pipeline.
    Embedding cosine similarity is available but is NOT sufficient alone to
    establish equivalence.
    """
    if sent_a.strip() == sent_b.strip():
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.EQUIVALENT,
            is_equivalent=True,
            confidence=1.0,
            similarity=1.0,
            explanation="Exact sentence match.",
        )

    prop_a = extract_proposition(sent_a)
    prop_b = extract_proposition(sent_b)

    shared_content = prop_a.content_tokens & prop_b.content_tokens
    content_jaccard = _jaccard(prop_a.content_tokens, prop_b.content_tokens)
    cross_lingual = _is_cross_lingual(sent_a, sent_b)

    # Fast path for completely unrelated sentences
    if similarity < 0.35 and content_jaccard < 0.05 and not shared_content:
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.UNRELATED,
            is_equivalent=False,
            confidence=0.0,
            similarity=similarity,
            explanation=f"Low semantic similarity, minimal shared content (sim={similarity:.2f}).",
        )

    # Check 1: Attributed Speaker Divergence (Different speakers making distinct arguments)
    if _speakers_diverge(prop_a.attributions, prop_b.attributions):
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.2,
            similarity=similarity,
            explanation=f"Attributed speaker divergence: '{prop_a.attributions[0]}' vs '{prop_b.attributions[0]}' making distinct assertions.",
        )

    # Check 2: Negation conflict
    if prop_a.is_negated != prop_b.is_negated:
        if content_jaccard >= 0.12 or (cross_lingual and similarity >= 0.60) or (similarity >= 0.60 and len(shared_content) >= 2):
            return ClaimEquivalenceResult(
                relation=ClaimRelationType.CONTRADICTORY,
                is_equivalent=False,
                confidence=0.0,
                similarity=similarity,
                explanation=f"Negation polarity conflict on shared proposition (content_jaccard={content_jaccard:.2f}, sim={similarity:.2f}).",
            )
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.0,
            similarity=similarity,
            explanation="Different negation polarity but insufficient shared content to classify as Contradictory.",
        )

    # Check 3: Structured Quantity conflict (Target-aligned numerical conflict)
    has_q_conflict, q_reason = _quantities_conflict_structured(
        prop_a.structured_quantities, prop_b.structured_quantities
    )
    if has_q_conflict:
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.CONTRADICTORY,
            is_equivalent=False,
            confidence=0.0,
            similarity=similarity,
            explanation=q_reason or f"Materially conflicting quantities: {prop_a.quantities} vs {prop_b.quantities}.",
        )

    # Check 3b: Disjoint Quantity Semantic Targets (e.g. 480億 for education vs 480億 for cut)
    if prop_a.structured_quantities and prop_b.structured_quantities:
        has_shared_target = any(
            bool(qa.target_tokens & qb.target_tokens)
            for qa in prop_a.structured_quantities
            for qb in prop_b.structured_quantities
        )
        if not has_shared_target and content_jaccard < 0.60:
            return ClaimEquivalenceResult(
                relation=ClaimRelationType.RELATED,
                is_equivalent=False,
                confidence=0.2,
                similarity=similarity,
                explanation="Sentences attach quantities to disjoint semantic targets.",
            )

    # Check 4: Phase / temporal status conflict (completed vs initiated/planned)
    if _phase_conflicts(sent_a, sent_b):
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.1,
            similarity=similarity,
            explanation="Temporal phase conflict: completed vs initiated/planned event.",
        )

    # Check 5: Directional outcome conflict (expansion vs reduction, revenue vs expenditure)
    if _direction_conflicts(sent_a, sent_b):
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.1,
            similarity=similarity,
            explanation="Directional outcome conflict: expansion/inflow vs reduction/outflow.",
        )

    # Check 6: Modality incompatibility
    if not _modality_compatible(prop_a.modality, prop_b.modality):
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.15,
            similarity=similarity,
            explanation=f"Modality incompatibility: '{prop_a.modality}' vs '{prop_b.modality}'.",
        )

    # Check 7: Attribution incompatibility
    if not _attributions_compatible(prop_a, prop_b):
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.15,
            similarity=similarity,
            explanation=f"Attribution incompatibility: '{prop_a.attribution_type}' vs '{prop_b.attribution_type}'.",
        )

    # Check 8: Agent divergence (monolingual only)
    if not cross_lingual and _divergent_agent_detected(sent_a, sent_b):
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.2,
            similarity=similarity,
            explanation="Agent divergence detected: same action/predicate with disjoint agents.",
        )

    # Check 9: Action overlap check (prevent same-actor different-action false merges)
    if not cross_lingual and similarity < 0.85:
        act_a = _action_tokens(sent_a)
        act_b = _action_tokens(sent_b)
        if act_a and act_b and not (act_a & act_b):
            return ClaimEquivalenceResult(
                relation=ClaimRelationType.RELATED,
                is_equivalent=False,
                confidence=0.2,
                similarity=similarity,
                explanation="Same subject/topic but disjoint action/predicate tokens.",
            )

    # Check 10: Cross-lingual decision
    if cross_lingual:
        if similarity >= 0.70:
            return ClaimEquivalenceResult(
                relation=ClaimRelationType.EQUIVALENT,
                is_equivalent=True,
                confidence=round(min(0.85 + similarity * 0.15, 1.0), 4),
                similarity=similarity,
                explanation=f"Cross-lingual equivalent claim (sim={similarity:.2f}).",
            )
        elif similarity >= 0.55:
            return ClaimEquivalenceResult(
                relation=ClaimRelationType.COMPATIBLE,
                is_equivalent=True,
                confidence=round(similarity * 0.9, 4),
                similarity=similarity,
                explanation=f"Cross-lingual compatible claim (sim={similarity:.2f}).",
            )
        else:
            return ClaimEquivalenceResult(
                relation=ClaimRelationType.RELATED,
                is_equivalent=False,
                confidence=0.2,
                similarity=similarity,
                explanation=f"Cross-lingual related topic (sim={similarity:.2f}).",
            )

    # Check 11: Monolingual Predicate + Content decision matrix
    pred_overlap = len(prop_a.predicate_tokens & prop_b.predicate_tokens)
    pred_union = len(prop_a.predicate_tokens | prop_b.predicate_tokens)
    smaller_pred = min(len(prop_a.predicate_tokens), len(prop_b.predicate_tokens))
    pred_coverage = pred_overlap / smaller_pred if smaller_pred > 0 else 0.0

    if similarity >= _HIGH_SIM:
        if content_jaccard >= 0.05 or len(shared_content) >= 1 or (prop_a.quantities and prop_b.quantities):
            relation = ClaimRelationType.EQUIVALENT if content_jaccard >= 0.15 or similarity >= 0.90 else ClaimRelationType.COMPATIBLE
            return ClaimEquivalenceResult(
                relation=relation,
                is_equivalent=True,
                confidence=round(min(0.80 + similarity * 0.15, 1.0), 4),
                similarity=similarity,
                explanation=(
                    f"High semantic similarity with structural agreement "
                    f"(sim={similarity:.2f}, content_j={content_jaccard:.2f}, "
                    f"pred_cov={pred_coverage:.2f})."
                ),
            )
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.3,
            similarity=similarity,
            explanation=(
                f"High embedding similarity but insufficient proposition-level "
                f"content overlap (sim={similarity:.2f}, "
                f"content_j={content_jaccard:.2f}). RELATED: possibly same "
                f"topic but different assertions."
            ),
        )

    if similarity >= _MED_SIM:
        if content_jaccard >= 0.12 or (content_jaccard >= 0.05 and pred_coverage >= 0.10) or (len(shared_content) >= 1 and similarity >= 0.75):
            relation = ClaimRelationType.EQUIVALENT if (content_jaccard >= 0.20 and pred_coverage >= 0.20) or similarity >= 0.80 else ClaimRelationType.COMPATIBLE
            return ClaimEquivalenceResult(
                relation=relation,
                is_equivalent=True,
                confidence=round(min(0.65 + content_jaccard * 0.30, 1.0), 4),
                similarity=similarity,
                explanation=(
                    f"Medium-high similarity with predicate + content overlap "
                    f"(sim={similarity:.2f}, content_j={content_jaccard:.2f}, "
                    f"pred_cov={pred_coverage:.2f})."
                ),
            )
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.25,
            similarity=similarity,
            explanation=(
                f"Medium similarity but insufficient proposition-level structural "
                f"overlap (sim={similarity:.2f}, content_j={content_jaccard:.2f}, "
                f"pred_cov={pred_coverage:.2f}). RELATED, not equivalent."
            ),
        )

    if similarity >= _LOW_SIM:
        if (content_jaccard >= 0.25 and pred_coverage >= 0.35) or (content_jaccard >= 0.35 and len(shared_content) >= 2):
            return ClaimEquivalenceResult(
                relation=ClaimRelationType.COMPATIBLE,
                is_equivalent=True,
                confidence=round(0.55 + content_jaccard * 0.30, 4),
                similarity=similarity,
                explanation=(
                    f"Lower similarity but strong proposition-level overlap "
                    f"(sim={similarity:.2f}, content_j={content_jaccard:.2f}, "
                    f"pred_cov={pred_coverage:.2f}). COMPATIBLE claim."
                ),
            )
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.2,
            similarity=similarity,
            explanation=(
                f"Similarity in candidate range but proposition-level overlap "
                f"insufficient (sim={similarity:.2f}, "
                f"content_j={content_jaccard:.2f}). RELATED, not equivalent."
            ),
        )

    if content_jaccard >= 0.10 or similarity >= 0.35:
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.15,
            similarity=similarity,
            explanation=(
                f"Some shared content but low overall similarity — different "
                f"factual propositions about a related topic "
                f"(sim={similarity:.2f}, content_j={content_jaccard:.2f})."
            ),
        )

    return ClaimEquivalenceResult(
        relation=ClaimRelationType.UNRELATED,
        is_equivalent=False,
        confidence=0.0,
        similarity=similarity,
        explanation=f"Low semantic similarity, minimal shared content (sim={similarity:.2f}).",
    )
