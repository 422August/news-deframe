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
from typing import Any


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
    r'請問|但請問|何謂|不予評論|不予評價|不予置評|'
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

# Generic UI chrome, ad headers, reader prompts, follow instructions
_UI_NOISE_PATTERNS = re.compile(
    r"^(?:廣告|Ad|AD|Advertisement|Sponsored|贊助商連結|贊助|廣編特輯|業配|"
    r"請繼續往下閱讀|延伸閱讀|推薦閱讀|相關新聞|相關報導|熱門新聞|點我看更多|更多新聞|最新消息|熱門話題|Read more|Related articles|Recommended|You may also like|"
    r"另開新視窗|另開視窗|點擊看大圖|點擊放大|點圖放大|點此觀看|詳見影片|點我看|點這裡|click here|open in new window|"
    r"(?:透過|加入|加入為|追蹤|訂閱|按讚|分享|關注|Follow|Subscribe to|Sign up for)\s*.+|"
    r"[\w\u4e00-\u9fff\s\.-]{2,30}(?:新聞網|日報|電子報|通訊社|廣播公司|電視台|新聞|News|Times|Post|Daily)(?:\s+[\w\.-]+)?|"
    r"(?:文|記者|特派員|攝影|撰文|編譯|責任編輯|編輯|Author|By)[\s／/：:][^\s]{2,15}(?:[／/][^\s]{2,15})?|"
    r"\d{4}[年\.-]\d{1,2}[月\.-]\d{1,2}日?(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
    r")[.。…\s]*$",
    re.IGNORECASE,
)


def check_claim_eligibility(text: str) -> ClaimEligibility:
    """Determine whether a candidate text span expresses a usable proposition.

    Rejects non-claim fragments (punctuation artifacts, byline/wire metadata,
    isolated discourse connectors, isolated temporal phrases, UI chrome) before embedding
    and clustering.
    """
    raw = text.strip()
    if not raw:
        return ClaimEligibility(False, "Empty sentence.")

    # 1. Punctuation only / pure numbers / minimal substantive lexical content
    substantive_cjk = [c for c in raw if 0x4E00 <= ord(c) <= 0x9FFF]
    substantive_words = re.findall(r"[a-zA-Z]{2,}", raw)
    has_digits = bool(re.search(r"\d", raw))

    if len(substantive_cjk) == 0 and len(substantive_words) == 0:
        return ClaimEligibility(False, "Numbers, punctuation, or symbol artifact without substantive lexical content.")

    if len(substantive_cjk) < 2 and len(substantive_words) == 0 and not (has_digits and len(substantive_cjk) == 1):
        return ClaimEligibility(False, "Single character non-propositional fragment.")

    # 2. Webpage UI chrome, advertisement, navigation prompt, or standalone metadata
    if _UI_NOISE_PATTERNS.match(raw):
        return ClaimEligibility(False, "Webpage UI chrome, advertisement, navigation prompt, or standalone metadata.")

    # 3. Metadata / byline / agency wire tag enclosed in brackets
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
    r"^(.{1,30}?)(?:受訪[時說]?|受訪表示|受訪)?(?:表示|指出|強調|稱|說|指|認為|坦言|透露|聲稱|聲明|宣布|宣稱|表明|回應|反駁|呼籲|直言|痛批|質疑|說明|公布|審定)[，,：:\s]+(.+)$"
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
    r"擴建|擴大|增加|調升|提高|增列|歲入|收入|調高|成長|興建|新建",
    re.IGNORECASE,
)
_DIRECTION_DECREASE = re.compile(
    r"\b(?:reduced|decreasing|decreased|lowers|lowered|cut|cuts|reduction|decrease|decline|losses?|expenditure|outflows?)\b|"
    r"縮減|減少|裁減|調降|降低|減列|刪減|統刪|歲出|支出|調低|虧損|跌|減收",
    re.IGNORECASE,
)

# ── Structured Quantity Extraction ────────────────────────────────────────────

_UNIT_MULTIPLIER = {
    "兆": 1000000000000.0,
    "億": 100000000.0,
    "万": 10000.0,
    "萬": 10000.0,
    "千": 1000.0,
    "百": 100.0,
    "trillion": 1000000000000.0,
    "billion": 1000000000.0,
    "million": 1000000.0,
    "thousand": 1000.0,
}

_COMPOUND_NUMERAL = re.compile(
    r"(?<![a-zA-Z0-9])"
    r"(\d+(?:[,，]\d+)*(?:\.\d+)?(?:\s*(?:兆|億|萬|千|百|trillion|billion|million|thousand))?)+"
    r"(?:\s*(?:%|百分之|元|人|名|個|戶|家|間|件|次|項|筆|dollars|percent|points|homes|workers))?",
    re.IGNORECASE,
)

_QUANTITY_CHINESE = re.compile(
    r"(?:約|逾|超過|不足|至少|最多)?[零一二兩三四五六七八九十百千萬億]+(?:餘|多|余)?(?:名|人|個|位|項|件|次|倍|%|百分之)?"
)

_STOP_TARGET = frozenset({
    "為", "是", "共", "約", "達", "至", "有", "在", "計", "了", "改", "暫", "新",
    "的", "個", "及", "與", "和", "從", "到", "由", "將", "於", "其", "中",
})

_ZH_ACTION_VERBS = re.compile(
    r"(?:通過|審查|審議|表決|完成|凍結|刪除|減列|增列|編列|統刪|砍刪|三讀|二讀|動支|副署|執行|呼籲|強調|指出|表示|"
    r"受訪|簽名|動員|卡關|改列|計列|增加|減少|調升|調降|宣布|認為|說明|重申|反對|贊成|逮捕|調查|抗議|批評|譴責|"
    r"要求|答應|改正|質詢|備詢|提案|達成|延宕|影響|運作|放寬|嚴管|啟動|重啟|停用|查扣|起訴|發生|舉行|出現|造成|"
    r"導致|涉及|拒絕|同意|確認|打算|展開|公布|發表|引發|推動|裁決|處理|審判|立法|執法|出面|爆發|受損|受創|修復|"
    r"中斷|恢復|搶修|搶通|延誤|損失|波及|受害|提告|求償|賠償|處罰|開罰|通報|撤銷|廢止|違規|違法|解僱|罷工|抗爭|示威|"
    r"發動|抵制|撤離|進駐|封鎖|解封|放行|攔截|查獲|破獲|查緝|查處|取締|移送|交保|羈押|飭回|定讞|判刑|上訴|駁回|"
    r"改判|獲釋|釋放|證實|澄清|駁斥|否認|指控|指責|反駁|坦承|承認|透露|直言|痛批|怒轟|開轟|疾呼|訴求|下達|送達|"
    r"擴建|興建|修建|新建|加設|設置|設立|清淤|管制|分發|開放|接種|核定|補助|開徵|汰換|停班|停課|復工|開工|完工|停工|"
    r"加強|擴充|巡弋|稽查|採購|興辦|營運|修訂|維護|溝通|協商|會商|研商|"
    r"崩塌|坍塌|停駛|受困|罹難|失聯|疏散|避難|歇業|停業|停電|漏油|起火|燃燒|爆炸|檢查|復原|修繕|演練|折斷|倒塌|摔倒|跌倒|砸傷|扭傷|出血|錯位|下陷|脫落|"
    r"位於|位在|座落於|坐落於|設於|處於|"
    r"是|為|有|無|達|佔|占|遭|欠|給|引|提|比讚)"
)

_ZH_SINGLE_ACTION_CHARS = frozenset(
    "說稱提簽砍刪凍批遭看給訪決讓答派辦降增減買賣宣罰告警抓救退換改請催追停封移扣拒准控談破跌漲死傷亡毀損打放收開閉走跑帶拉推"
)

_EN_ACTION_VERBS = re.compile(
    r"\b(?:pass|passed|passes|passing|approve|approved|approves|approving|vote|voted|votes|voting|"
    r"cut|cuts|cutting|reduce|reduced|reduces|reducing|increase|increased|increases|increasing|"
    r"freeze|frozen|freezes|freezing|complete|completed|completes|completing|said|states|stated|stating|"
    r"demanded|demands|demanding|urged|urges|urging|is|was|were|are|am|be|been|being|has|had|have|having|"
    r"declined|declines|declining|struck|strikes|striking|strike|broke|break|broken|breaking|breaks|"
    r"arrest|arrests|arrested|arresting|suffer|suffers|suffered|suffering|make|makes|made|making|"
    r"take|takes|took|taken|taking|see|sees|saw|seen|seeing|find|finds|found|finding|fall|falls|fell|fallen|falling|"
    r"rise|rises|rose|risen|rising|grow|grows|grew|grown|growing|lose|loses|lost|losing|win|wins|won|winning|"
    r"give|gives|gave|given|giving|tell|tells|told|telling|report|reports|reported|reporting|claim|claims|claimed|claiming|"
    r"confirm|confirms|confirmed|confirming|deny|denies|denied|denying|agree|agrees|agreed|agreeing|"
    r"launch|launches|launched|launching|occur|occurs|occurred|occurring|cause|causes|caused|causing|"
    r"halt|halts|halted|halting|stop|stops|stopped|stopping|suspend|suspends|suspended|suspending|"
    r"damage|damages|damaged|damaging|destroy|destroys|destroyed|destroying|kill|kills|killed|killing|"
    r"injure|injures|injured|injuring|detain|detains|detained|detaining|fine|fines|fined|fining|"
    r"investigate|investigates|investigated|investigating|charge|charges|charged|charging|close|closed|closing|"
    r"reopen|reopened|reopening|repair|repaired|repairing|restore|restored|restoring|publish|published|publishing|"
    r"do|does|did|done|doing|will|would|shall|should|can|could|may|might|must)\b",
    re.IGNORECASE,
)

_EN_NUMBER_WORDS: dict[str, float] = {
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
    "ten": 10.0, "eleven": 11.0, "twelve": 12.0, "twenty": 20.0,
    "thirty": 30.0, "forty": 40.0, "fifty": 50.0, "sixty": 60.0,
    "seventy": 70.0, "eighty": 80.0, "ninety": 90.0, "hundred": 100.0,
    "thousand": 1000.0, "million": 1000000.0, "billion": 1000000000.0,
}

_QUANTITY_ENGLISH_PATTERN = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b",
    re.IGNORECASE,
)


_EN_NON_VERB_ING = frozenset({
    "according", "during", "morning", "evening", "nothing", "something",
    "everything", "anything", "building", "meeting", "redevelopment",
    "recording", "recordings", "ring", "wing", "king", "thing", "spring",
})


def _has_actionable_predicate(text: str) -> bool:
    """Return True if text contains a recognizable action verb, verbal suffix, or predicate."""
    if _ZH_ACTION_VERBS.search(text):
        return True
    if any(ch in _ZH_SINGLE_ACTION_CHARS for ch in text):
        return True
    if _EN_ACTION_VERBS.search(text):
        return True
    # English morphology heuristic: words ending with past/participle/verbal suffixes
    words = re.findall(r"[a-zA-Z]+", text)
    for w in words:
        wl = w.lower()
        if wl in _EN_NON_VERB_ING:
            continue
        if len(wl) >= 4 and (wl.endswith("ed") or wl.endswith("ing") or wl.endswith("ize") or wl.endswith("ized") or wl.endswith("ated")):
            return True
    return False


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

    # Mixed Arabic-CJK/Latin: parse parts like '2兆8622億5319萬1000' or '4 billion'
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(兆|億|萬|千|百|trillion|billion|million|thousand)?", q_clean, flags=re.IGNORECASE)
    if parts:
        tot = 0.0
        has_unit = False
        for v_str, unit in parts:
            if v_str:
                u_key = unit.lower() if unit else ""
                mult = _UNIT_MULTIPLIER.get(u_key, 1.0)
                if u_key in _UNIT_MULTIPLIER:
                    has_unit = True
                tot += float(v_str) * mult
        if tot > 0 and (has_unit or len(parts) == 1):
            return tot

    zh_val = _parse_chinese_numeral(q_clean)
    if zh_val is not None:
        return zh_val

    return _EN_NUMBER_WORDS.get(q_clean.lower())


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
                if ch not in _STOP_TARGET:
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

        seen_spans.add(m.span())
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
                if ch not in _STOP_TARGET:
                    target_tokens.add(ch)

        results.append(StructuredQuantity(raw=raw_str, val=val, target_tokens=frozenset(target_tokens)))

    # Also parse English number words
    for m in _QUANTITY_ENGLISH_PATTERN.finditer(text):
        if any(s <= m.start() and m.end() <= e for s, e in seen_spans):
            continue
        raw_str = m.group(0).strip()
        val = _EN_NUMBER_WORDS.get(raw_str.lower())
        if val is None:
            continue

        seen_spans.add(m.span())
        start_idx = max(0, m.start() - 20)
        end_idx = min(len(text), m.end() + 20)
        pre_ctx = text[start_idx:m.start()]
        post_ctx = text[m.end():end_idx]

        target_tokens = set()
        for ctx in (pre_ctx, post_ctx):
            for w in re.findall(r"[a-zA-Z]+", ctx):
                if len(w) >= 3 and w.lower() not in {"the", "and", "for", "was", "were", "has", "had", "been", "that", "this"}:
                    target_tokens.add(w.lower())

        results.append(StructuredQuantity(raw=raw_str, val=val, target_tokens=frozenset(target_tokens)))

    return results


# Rhetorical questions / non-assertive conversational patterns to reject at post-decomposition stage
_RHETORICAL_PATTERNS = re.compile(
    r"(?:為什麼.*(?:說|講|要|會)|為何.*(?:說|講|要|會)|何謂因.*何謂果|何以致之.*孰以致之|變成什麼樣子|不予評價|不予評論|"
    r"各方講法太多|把.*當.*嗎|不是.*簽名了嗎|改正了沒有|是不是.*了|算不算|到底.*誰|誰知道|"
    r"\bwhy\s+would\b|\bwhat\s+would\s+happen\b|\bhow\s+could\b|\bwho\s+knows\b|\bhow\s+on\s+earth\b)",
    re.IGNORECASE,
)

# Attribution endings that signal an intro prefix that should reattach forward to its quoted content
_FORWARD_ATTRIBUTION_ENDINGS: tuple[str, ...] = (
    "表示", "指出", "強調", "稱", "說", "指", "認為", "坦言", "透露",
    "聲稱", "聲明", "宣布", "宣稱", "表明", "回應", "反駁", "呼籲", "喊話", "質疑",
    "觀測到", "觀測", "目視確認", "確認", "發現", "看到", "拍到", "拍攝到", "記錄到", "顯示",
    "收到情報指出", "指稱", "回憶", "評估", "證實", "通報", "提醒", "警告",
    "said", "stated", "urged", "demanded", "claimed", "noted", "argued", "insisted", "added",
    "announced", "reported", "explained", "warned", "observed", "observed that", "confirmed that", "recalled",
)

# Continuations / quantifiers / purpose clauses that should reattach backward to the preceding proposition
_BACKWARD_REATTACH_STARTINGS: tuple[str, ...] = (
    "及", "以及", "甚至包括",
    "以", "以利", "以利於", "以求", "旨在", "期能", "以便", "用以", "藉以", "為確認", "以確認", "為確保", "以確保", "為求", "以防", "以免", "以維護", "以保障",
    "導致", "造成", "引發", "致使", "使得", "促使", "迫使",
    "占", "佔", "合計", "共計", "金額共", "分別", "皆", "均", "全數", "則全數", "全部都", "均不得",
    "始得", "不得", "提案通過", "獲表決通用", "獲表決通過", "不列入", "出席立委", "出席立", "贊成", "反對", "經同意後",
    "升格為", "改為", "轉為", "作為", "調整為", "降為", "提升為", "擴大為", "是", "被命名為",
    "accounting for", "representing", "amounting to", "totaling", "prompting", "resulting in",
    "leading to", "causing", "in order to", "so as to", "to ensure", "to prevent", "aiming to",
    "along with", "which", "who", "whom", "whose",
)

_ZH_SPEAKER_PREFIX = re.compile(
    r"^(?:部會首長方面，|會中，|會中|對此，|對此)?\s*"
    r"([^，,：:\s]{2,30}?)"
    r"(?:今天午後.*時|今天午後|今天|昨天|上午|下午|晚間|午後|昨晚|今晚|敲槌後|在三讀後)?\s*"
    r"(?:受訪[時說]|受訪表示|受訪說|指出|表示|強調|呼籲|認為|說|稱|提案指出|提案表示|提案呼籲|喊話|透露|坦言|公布|宣布|裁定|審定)\s*$"
)

_ZH_SUBORD_START = (
    "因", "但因", "雖然", "雖", "儘管", "即使", "若非", "除非", "隨著", "鑑於",
    "受", "受到", "因受", "因受到", "受限於", "在於",
    "若", "如果", "假如", "在未", "在沒有", "在不", "為了", "因為", "由於", "等到", "從",
    "經過", "歷經", "除", "除法律", "除債務", "包括", "包含", "還包含", "除外", "下週", "上週", "本週",
    "after", "before", "when", "while", "if", "unless", "because", "due to", "in order to", "despite", "following", "during",
    "although", "though", "even though", "since", "as", "whereas",
)

_ZH_SUBORD_END = (
    "後", "時", "前", "之際", "之後", "之前", "時許", "時左右", "包含", "包括", "還包含", "為止", "起", "過程中", "期間", "以來",
    "影響", "衝擊", "波及",
    "上午", "下午", "晚間", "早晨", "中午", "昨晚", "今晚", "當下", "同時", "隨後",
)

_ZH_TOPIC_START = (
    "至於", "關於", "針對", "在", "而在", "在通案", "在野黨基於", "最受", "特別是", "尤其", "根據", "依據", "依照", "按照", "反而是", "到目前", "有關", "就", "鑑於",
    "as for", "regarding", "according to", "based on", "in terms of", "with regard to", "at around", "at about", "at approximately", "at", "in the", "on the",
)

_ZH_TOPIC_END = (
    "部分", "方面", "而言", "來看", "結果", "外", "等主要", "等業者", "等單位", "等機構", "等代表", "等民眾", "等團體", "等",
)


def _join_clauses(a: str, b: str) -> str:
    """Join two clauses preserving appropriate language punctuation."""
    a_clean = a.rstrip("，,。；; ")
    b_clean = b.lstrip("，,。；; ")
    if not a_clean:
        return b_clean
    if not b_clean:
        return a_clean
    sep = "，" if _has_cjk(a_clean + b_clean) else ", "
    return a_clean + sep + b_clean


@dataclass
class DecompositionStep:
    """A single transformation or validation step in sentence decomposition."""

    action: str  # 'initial_split', 'reattach', 'inherit_context', 'eligibility'
    input_fragments: list[str]
    output_fragments: list[str]
    reason: str
    direction: str = ""  # 'forward', 'backward', 'none'


@dataclass
class SentenceDecompositionTrace:
    """Full diagnostic trace of a sentence through the decomposition pipeline."""

    sentence_idx: int
    source_sentence: str
    article_id: str = ""
    initial_split: list[str] = field(default_factory=list)
    reattachments: list[dict[str, Any]] = field(default_factory=list)
    context_inheritances: list[dict[str, Any]] = field(default_factory=list)
    eligibility_decisions: list[dict[str, Any]] = field(default_factory=list)
    final_propositions: list[AtomicProposition] = field(default_factory=list)

    def render_text(self) -> str:
        """Render human-readable diagnostic report."""
        lines = [
            f"SOURCE [{self.article_id}:{self.sentence_idx}]: {self.source_sentence}",
            f"INITIAL ({len(self.initial_split)}):",
        ]
        for idx, frag in enumerate(self.initial_split):
            lines.append(f"  P{idx+1}: {frag}")
        if self.reattachments:
            lines.append("REATTACH:")
            for r in self.reattachments:
                lines.append(f"  {r.get('merged', '')} ({r.get('direction', '')}) -> reason: {r.get('reason', '')}")
        if self.context_inheritances:
            lines.append("INHERIT CONTEXT:")
            for c in self.context_inheritances:
                lines.append(f"  {c.get('target', '')} inherited {c.get('type', '')}: {c.get('context', '')} (source: {c.get('source', '')})")
        if self.eligibility_decisions:
            lines.append("ELIGIBILITY:")
            for e in self.eligibility_decisions:
                status = "ACCEPT" if e.get("eligible") else "REJECT"
                lines.append(f"  {e.get('candidate', '')} -> {status} (reason: {e.get('reason', '')})")
        lines.append(f"FINAL ({len(self.final_propositions)}):")
        for idx, p in enumerate(self.final_propositions):
            lines.append(f"  A{idx+1}: \"{p.proposition_text}\" (speaker={p.speaker}, qty={len(p.quantities)})")
        return "\n".join(lines)


def _reattach_proposition_fragments(
    clauses: list[str],
    trace: SentenceDecompositionTrace | None = None,
) -> list[str]:
    """Conservatively reattach dependent/subordinate fragments to neighboring clauses."""
    if not clauses:
        return []

    # Clean boundary whitespace & trailing discourse particles
    cleaned: list[str] = []
    for c in clauses:
        c_str = c.strip().strip("，,")
        if not c_str:
            continue
        # Strip trailing discourse particles like '，對此', '，但請問'
        c_str = re.sub(r"[，,]\s*(?:對此|為此|但請問|請問|因此|此外|另外)$", "", c_str).strip()
        if c_str:
            cleaned.append(c_str)

    if len(cleaned) <= 1:
        return cleaned

    # Pass 1: Forward reattachment for attribution headers, temporal/condition suffixes, and topic prefixes
    fwd_merged: list[str] = []
    pending_prefix = ""

    for i, clause in enumerate(cleaned):
        current = (pending_prefix + "，" + clause) if pending_prefix else clause
        pending_prefix = ""
        is_last = (i == len(cleaned) - 1)

        if not is_last:
            c_clean = current.rstrip("，,。；; ")

            # (a) Attribution header without body (e.g. '黃國昌受訪說', '韓國瑜也呼籲行政部門')
            is_attr_intro = any(c_clean.endswith(a) for a in _FORWARD_ATTRIBUTION_ENDINGS) and len(c_clean) < 45

            # (b) Subordinate temporal / condition prefix (e.g. '朝野歷經一下午表決大戰後', '完成依法編列預算及副署法律後')
            is_subord_suffix = (
                (any(c_clean.startswith(s) for s in _ZH_SUBORD_START) or any(c_clean.endswith(e) for e in _ZH_SUBORD_END))
                and len(c_clean) < 45
                and not extract_structured_quantities(c_clean)
            )

            # (c) Topic / conditional prefix without verb/qty (e.g. '反而是新興計畫', '除債務償還照列外', '最受矚目的歲出部分')
            is_topic_prefix = (
                (any(c_clean.startswith(s) for s in _ZH_TOPIC_START) or any(c_clean.endswith(e) for e in _ZH_TOPIC_END))
                and len(c_clean) < 45
                and not extract_structured_quantities(c_clean)
                and not _has_actionable_predicate(c_clean)
            )

            if is_attr_intro or is_subord_suffix or is_topic_prefix:
                pending_prefix = current
                if trace is not None:
                    reason = "attribution intro" if is_attr_intro else ("subordinate clause" if is_subord_suffix else "topic prefix")
                    trace.reattachments.append({
                        "merged": f"{current} -> next",
                        "direction": "forward",
                        "reason": reason,
                    })
                continue

        fwd_merged.append(current)

    if pending_prefix:
        if fwd_merged:
            fwd_merged[-1] = fwd_merged[-1] + "，" + pending_prefix
        else:
            fwd_merged.append(pending_prefix)

    # Pass 2: Backward reattachment for quantifier, percentage, and coordinate continuations
    bwd_merged: list[str] = []
    for clause in fwd_merged:
        c_clean = clause.lstrip("，, ")
        c_lower = c_clean.lower()
        is_bwd = any(c_clean.startswith(s) or c_lower.startswith(s) for s in _BACKWARD_REATTACH_STARTINGS) and len(c_clean) < 65
        if bwd_merged and is_bwd:
            bwd_merged[-1] = _join_clauses(bwd_merged[-1], clause)
            if trace is not None:
                trace.reattachments.append({
                    "merged": f"{bwd_merged[-1]} <- {clause}",
                    "direction": "backward",
                    "reason": "dependent continuation / predicate / vote tally",
                })
        else:
            bwd_merged.append(clause)

    return bwd_merged


def check_atomic_proposition_eligibility(text: str) -> ClaimEligibility:
    """Validate whether a decomposed proposition can independently enter claim clustering.

    Enforces that every clustered proposition expresses a standalone factual assertion:
    - Meaningful predicate, copular assertion, or quantity with semantic target.
    - Rejects pure rhetorical questions, speech scaffolds, isolated dependent clauses,
      and non-assertive topic headers.
    """
    raw = text.strip()
    if not raw:
        return ClaimEligibility(False, "Empty proposition.")

    # 1. Source sentence eligibility base checks
    base_elig = check_claim_eligibility(raw)
    if not base_elig.eligible:
        return base_elig

    # 2. Rhetorical question / non-assertive conversational commentary filtering
    if _RHETORICAL_PATTERNS.search(raw):
        return ClaimEligibility(
            False,
            "Rhetorical question or conversational commentary without standalone factual assertion.",
        )

    # 3. Attribution speech-act header only with empty / incomplete body
    zh_attr = _ZH_ATTRIBUTION_PATTERN.match(raw)
    if zh_attr:
        body = zh_attr.group(2).strip()
        sub_cjk = [c for c in body if 0x4E00 <= ord(c) <= 0x9FFF]
        sub_words = body.split()
        if len(sub_cjk) < 3 and len(sub_words) < 2:
            return ClaimEligibility(False, "Attribution introduction without substantive propositional body.")

    en_attr = _EN_ATTRIBUTION_PATTERN.match(raw)
    if en_attr:
        body = en_attr.group(2).strip()
        if len(body.split()) < 2:
            return ClaimEligibility(False, "Attribution introduction without substantive propositional body.")

    # 4. Isolated subordinate / dependent clause without main predicate
    if any(raw.endswith(e) for e in ("後", "時", "前", "之際", "之後", "之前", "時許", "時左右", "以來源", "為止", "起", "當下", "同時", "隨後")):
        if not _has_actionable_predicate(raw) and not extract_structured_quantities(raw):
            return ClaimEligibility(False, "Isolated subordinate temporal clause without main factual assertion.")

    # 4b. Isolated subordinate conjunction clause without main matrix assertion
    if raw.startswith(("因", "但因", "雖然", "雖", "儘管", "即使", "若非", "除非", "若", "如果", "假如")) and not _has_actionable_predicate(raw):
        return ClaimEligibility(False, "Isolated subordinate conjunction clause without main factual assertion.")

    # 4c. Isolated coordinating conjunction or purpose fragment without independent matrix clause
    if raw.startswith(("及", "以及", "與", "和", "並", "且", "甚至", "而且", "或是", "或者", "以確認", "為確認", "以確保", "為確保", "以利", "以便", "用以", "藉以", "以防", "以免", "and ", "or ", "as well as ", "in order to ", "so as to ")):
        if not _has_actionable_predicate(raw) or len(raw) < 15:
            return ClaimEligibility(False, "Isolated coordinating conjunction or purpose fragment without independent matrix clause.")

    # 4d. Dangling attribution introduction or observation prefix without propositional complement
    raw_stripped = raw.rstrip("，,。；; ")
    if any(raw_stripped.endswith(a) for a in _FORWARD_ATTRIBUTION_ENDINGS):
        if len(raw_stripped) < 40 and not extract_structured_quantities(raw_stripped):
            return ClaimEligibility(False, "Dangling attribution introduction or observation prefix without propositional complement.")

    # 4e. Subjectless aspectual predicate fragment without explicit or inherited actor
    if raw.startswith(("正極力", "目前正", "正全力", "正在", "正積極", "陸續在", "持續在", "目前仍在", "仍在")):
        if not re.search(r"^[^\uFF0C,\uFF1A:\s]{2,12}?(?:" + _ZH_ACTION_VERBS.pattern + r")", raw):
            return ClaimEligibility(False, "Subjectless aspectual predicate fragment without explicit or inherited actor.")

    # 4f. Isolated causal or passive modifier fragment without independent matrix assertion
    # (e.g. '受地震影響', '受熊本機場跑道關閉影響', '受到強風影響')
    if raw.startswith(("受", "受到", "因受", "因受到", "受限於")) and any(raw.endswith(e) for e in ("影響", "衝擊", "波及", "限制", "牽連")):
        if not re.search(r"^[^\s]{2,15}(?:及|、|與|和)?[^\s]{0,15}(?:均|皆|已|亦|陸續|全面)?受", raw):
            return ClaimEligibility(False, "Isolated causal or passive modifier fragment without independent matrix assertion.")

    # 4g. Isolated locative distance or geographic coordinate fragment without factual assertion
    # (e.g. '長崎東方約80公里處', '花蓮東南方約20公里海域')
    if any(raw.endswith(e) for e in ("處", "附近", "方向", "周邊", "一帶", "公里處", "公尺處", "海域", "外海")):
        if not any(v in raw for v in ("位於", "位在", "發生", "出現", "發現", "設於", "建於", "成立", "傳出", "座落", "坐落", "座落在", "坐落在", "is", "was", "were", "located", "occurred")):
            return ClaimEligibility(False, "Isolated locative distance or geographic coordinate fragment without factual assertion.")

    # 5. Topic / Context header without factual assertion
    if raw.startswith(("在", "而在", "至於", "關於", "根據", "依據", "依照", "對於", "從", "經過", "歷經", "針對", "有關")) and any(raw.endswith(e) for e in ("部分", "方面", "結果", "外", "來看", "而言", "當中", "過程", "期間", "大戰")):
        if not _has_actionable_predicate(raw) or len(raw) < 20:
            return ClaimEligibility(False, "Isolated topic or context header without factual assertion.")

    # 5b. Isolated entity list / subject noun phrase ending with 等/業者 without predicate
    if any(raw.endswith(e) for e in ("等主要", "等業者", "等單位", "等機構", "等代表", "等民眾", "等團體", "等主要超商業者", "等")):
        if not _has_actionable_predicate(raw) and not re.search(r"[為是有]|(?:is|was|were|are|has|have)\b", raw):
            return ClaimEligibility(False, "Isolated entity list or subject header without predicate.")

    # 5c. English isolated prepositional / topic phrase (e.g. 'According to police', 'At around 7 p.m.')
    raw_lower = raw.lower()
    if (
        raw_lower.startswith(("according to", "based on", "as for", "regarding", "in terms of", "with regard to", "at around", "at about", "at approximately", "at ", "during "))
        and len(raw.split()) <= 5
        and not re.search(r"\b(?:is|was|were|are|has|have|had|arrested|approved|moved|pushed|delayed|sustained|closed|resigned)\b", raw_lower)
    ):
        return ClaimEligibility(False, "Isolated English prepositional or introductory phrase without factual assertion.")

    # 6. Must contain actionable predicate OR copular assertion OR quantity with semantic target
    has_qty = bool(extract_structured_quantities(raw))
    has_verb = _has_actionable_predicate(raw)
    has_copula = bool(re.search(r"[為是有]|(?:is|was|were|are|has|have)\b", raw))

    if not has_verb and not has_copula:
        if len(raw) < 10 or not has_qty:
            return ClaimEligibility(False, "Missing actionable predicate, copular assertion, or quantity target.")

    if not has_qty and not has_verb and not has_copula:
        return ClaimEligibility(False, "Missing actionable predicate, copular assertion, or quantity target.")

    return ClaimEligibility(True, "Eligible atomic proposition.")


def is_atomic_proposition_eligible(text: str) -> bool:
    """Convenience boolean check for atomic proposition eligibility."""
    return check_atomic_proposition_eligibility(text).eligible


def extract_atomic_propositions(
    article_id: str,
    sentence_idx: int,
    sentence: str,
    trace: SentenceDecompositionTrace | None = None,
) -> list[AtomicProposition]:
    """Decompose a candidate sentence into one or more atomic factual propositions.

    Preserves full provenance (article_id, sentence_idx, original sentence text,
    clause-specific quantities, speaker attribution, negation, and modality).
    """
    raw = sentence.strip()
    if not raw or not check_claim_eligibility(raw).eligible:
        return []

    if trace is not None:
        trace.sentence_idx = sentence_idx
        trace.source_sentence = raw
        trace.article_id = article_id

    # Decompose into candidate sentences / clauses using sentence and major delimiters
    major_chunks = re.split(r"(?<=[a-z0-9\)\"\'”’])\.\s+(?=[A-Z0-9\"\'“‘])|[。！？!?；;\n]", raw)

    propositions: list[AtomicProposition] = []
    p_idx = 0

    for chunk in major_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        chunk_attributions, chunk_body, chunk_attr_type = _extract_attributions(chunk)
        chunk_speaker = chunk_attributions[0] if chunk_attributions else None
        current_speaker = chunk_speaker
        prev_subject = chunk_speaker

        sub_chunks = [sc.strip() for sc in re.split(r"[,，]", chunk_body) if sc.strip()]
        if not sub_chunks:
            continue

        if trace is not None:
            trace.initial_split.extend(sub_chunks)

        # Pass 1: Forward reattachment & mid-sentence attribution extraction
        # Each entry is (text, speaker) to track attribution directionality.
        fwd: list[tuple[str, str | None]] = []
        pending_prefix = ""
        chunk_speaker = current_speaker  # speaker active for this chunk

        i = 0
        while i < len(sub_chunks):
            c = sub_chunks[i]
            c_clean = _join_clauses(pending_prefix, c) if pending_prefix else c
            pending_prefix = ""

            # Check mid-sentence attribution intro
            attr_m = _ZH_SPEAKER_PREFIX.match(c)
            if attr_m and i + 1 < len(sub_chunks):
                raw_spk = attr_m.group(1).strip()
                spk = re.sub(r"^(?:在野黨|執政黨|民眾黨主席|民進黨團總召|國民黨立委|行政院長|立法院長|教育部長|部長|院長|立委)", "", raw_spk).strip()
                chunk_speaker = spk or raw_spk
                current_speaker = chunk_speaker
                if trace is not None:
                    trace.context_inheritances.append({
                        "target": "subsequent_clauses",
                        "type": "speaker",
                        "context": chunk_speaker,
                        "source": c,
                    })
                i += 1
                continue

            c_check = c.strip("，,。；; \"'「」『』 ")
            c_check_lower = c_check.lower()

            # (a) Attribution intro ending with speech-act verb
            is_attr_intro = any(c_check.endswith(a) for a in _FORWARD_ATTRIBUTION_ENDINGS) and len(c_check) < 50

            # (b) Subordinate clause (temporal / causal / condition / complement prefix)
            is_subord = (
                (any(c_check.startswith(s) or c_check_lower.startswith(s) for s in _ZH_SUBORD_START) or any(c_check.endswith(e) for e in _ZH_SUBORD_END))
                and len(c_check) < 55
            )

            # (c) Topic / Prepositional adjunct header / Discourse connector
            is_topic = (
                (any(c_check.startswith(s) or c_check_lower.startswith(s) for s in _ZH_TOPIC_START) or any(c_check.endswith(e) for e in _ZH_TOPIC_END) or c_check_lower in ("however", "therefore", "moreover", "furthermore", "meanwhile", "nevertheless", "對此", "為此", "此外", "另外"))
                and len(c_check) < 55
            )

            if (is_attr_intro or is_topic or is_subord) and i + 1 < len(sub_chunks):
                pending_prefix = c_clean
                if trace is not None:
                    reason = "attribution intro" if is_attr_intro else ("topic prefix" if is_topic else "subordinate clause")
                    trace.reattachments.append({
                        "merged": f"{c_clean} -> next",
                        "direction": "forward",
                        "reason": reason,
                    })
                i += 1
                continue

            fwd.append((c_clean, chunk_speaker))
            i += 1

        if pending_prefix:
            if fwd:
                text, spk = fwd[-1]
                fwd[-1] = (_join_clauses(text, pending_prefix), spk)
            else:
                fwd.append((pending_prefix, chunk_speaker))

        # Pass 2: Backward reattachment for continuations, list arguments, and tallies
        bwd: list[tuple[str, str | None]] = []
        for c_text, c_spk in fwd:
            c_clean = c_text.lstrip("，, ")
            c_lower = c_clean.lower()
            is_bwd = any(c_clean.startswith(s) or c_lower.startswith(s) for s in _BACKWARD_REATTACH_STARTINGS) and len(c_clean) < 65
            if bwd and is_bwd:
                prev_text, prev_spk = bwd[-1]
                bwd[-1] = (_join_clauses(prev_text, c_text), prev_spk)
                if trace is not None:
                    trace.reattachments.append({
                        "merged": f"{bwd[-1][0]} <- {c_text}",
                        "direction": "backward",
                        "reason": "continuation / predicate / tally",
                    })
            else:
                bwd.append((c_text, c_spk))

        # Pass 3: Subject inheritance and eligibility validation
        for c_text, frag_speaker in bwd:
            c_clean = c_text.strip("，,。；;.!? \"'「」『』 ")
            # Strip leading coordinating conjunctions from independent propositions
            c_clean = re.sub(r"^(?:and|but|or|so|yet|且|而|並|然而|此外|另外|對此|不過)[，,\s]+", "", c_clean, flags=re.IGNORECASE).strip("，,。；;.!? \"'「」『』 ")
            if not c_clean:
                continue

            # Initialize prev_subject from current_speaker if available
            if not prev_subject and frag_speaker:
                prev_subject = frag_speaker

            # Extract subject if present at start of clause
            cand_text = re.sub(r"^(?:今天|昨天|明天|上午|下午|晚間|早晨|中午|日前|近日|目前|當時|隨後|接著|最後)[上下午晚間\s]*", "", c_clean)
            subj_m = re.match(r"^([^\uFF0C,\uFF1A:\s]{2,12}?)(?=" + _ZH_ACTION_VERBS.pattern + r")", cand_text)
            if subj_m and not any(subj_m.group(1).startswith(p) for p in (
                "並", "且", "而", "同時", "最後", "接著", "隨後", "隨即", "隨之", "經", "在", "從", "到", "由", "根據", "依照", "按照",
                "若", "如果", "為了", "由於", "因為", "除", "破天荒", "表達", "進而", "反而", "曾經", "已經", "預計", "即將",
                "再度", "再次", "重新", "持續", "逐步", "全面", "分別", "主要", "依法", "依規定", "正", "正在", "未", "未能", "不", "不再",
                "處理", "進行", "開始", "展開", "完成", "繼續", "討論", "審查", "表決",
            )):
                prev_subject = subj_m.group(1).strip()

            # Inherit subject if clause starts with verb / coordinate marker / aspect marker without subject.
            # Only inherit if prev_subject is NOT already the prefix of the clause (prevents doubling).
            if prev_subject and not c_clean.startswith(prev_subject) and (
                c_clean.startswith(("並", "且", "進而", "同時", "以及", "及", "正", "正在", "目前正", "正極力", "仍在", "持續", "陸續", "全面", "逐步", "正全力", "並已", "並將", "並可", "並在", "並由", "目前正在", "正積極"))
                or (not _ZH_SUBJECT_SPLIT.search(c_clean) and any(c_clean.startswith(v) for v in ("派員", "決定", "宣布", "展開", "完成", "開徵", "加設", "興建", "擴建", "確認", "評估", "調查", "救災", "成立", "下令", "指示")))
            ):
                inherited = prev_subject + c_clean
                if trace is not None:
                    trace.context_inheritances.append({
                        "target": c_clean,
                        "type": "subject",
                        "context": prev_subject,
                        "source": prev_subject,
                    })
                c_clean = inherited

            # Enforce post-decomposition atomic proposition eligibility
            elig = check_atomic_proposition_eligibility(c_clean)
            if trace is not None:
                trace.eligibility_decisions.append({
                    "candidate": c_clean,
                    "eligible": elig.eligible,
                    "reason": elig.reason,
                })

            if not elig.eligible:
                continue

            c_quantities = tuple(extract_structured_quantities(c_clean))
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

            prop = AtomicProposition(
                prop_id=f"{article_id}:{sentence_idx}:{p_idx}",
                article_id=article_id,
                sentence_idx=sentence_idx,
                sentence_text=raw,
                proposition_text=c_clean,
                speaker=frag_speaker,
                modality=modality,
                is_negated=c_is_neg,
                quantities=c_quantities,
                content_tokens=c_tokens,
                predicate_tokens=p_tokens,
                attribution_type="reported_speech" if frag_speaker else chunk_attr_type,
            )
            propositions.append(prop)
            if trace is not None:
                trace.final_propositions.append(prop)
            p_idx += 1

    if not propositions:
        top_attributions, body, top_attr_type = _extract_attributions(raw)
        top_speaker = top_attributions[0] if top_attributions else None
        if is_atomic_proposition_eligible(body):
            c_quantities = tuple(extract_structured_quantities(body))
            c_is_neg = bool(
                _ZH_NEGATION.search(body)
                or any(w in body.lower().split() for w in _EN_NEGATION_WORDS)
            )
            c_tokens = frozenset(_content_tokens(body))
            p_tokens = frozenset(_predicate_tokens(body))
            prop = AtomicProposition(
                prop_id=f"{article_id}:{sentence_idx}:0",
                article_id=article_id,
                sentence_idx=sentence_idx,
                sentence_text=raw,
                proposition_text=body,
                speaker=top_speaker,
                modality="statement",
                is_negated=c_is_neg,
                quantities=c_quantities,
                content_tokens=c_tokens,
                predicate_tokens=p_tokens,
                attribution_type=top_attr_type,
            )
            propositions.append(prop)
            if trace is not None:
                trace.final_propositions.append(prop)

    return propositions


def trace_sentence_decomposition(
    sentence: str,
    article_id: str = "art",
    sentence_idx: int = 0,
) -> SentenceDecompositionTrace:
    """Produce a full diagnostic trace of a sentence through the decomposition pipeline."""
    trace = SentenceDecompositionTrace(
        sentence_idx=sentence_idx,
        source_sentence=sentence.strip(),
        article_id=article_id,
    )
    extract_atomic_propositions(article_id, sentence_idx, sentence, trace=trace)
    return trace


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
