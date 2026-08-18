"""Actor resolution pipeline for the Entity x Outlet Framing Matrix.

This module implements the layered pipeline:

    raw NER candidates  ──┐
                           ├─→ actor validation
    SVO-grounded spans  ──┘
    → actor mentions (SVO-grounded role assignment)
    → canonical actors (event-level deduplication)
    → framing statistics (typed, ratio-denominator-explicit)

Design principles
-----------------
* Two complementary candidate extraction paths:

  Path A (NER-based): Named entities whose NER label is in _ACTOR_NER_LABELS.
    Validated with structural checks to reject spans that the NER model may
    have misclassified (adjectives, verb fragments, short noise tokens).

  Path B (SVO-based): Subject and object spans from SVO records, validated
    by structural properties (surface length, character composition).
    This path recovers participants that NER fails to tag (e.g. generic
    institutional references in Chinese text such as 警方, 主辦團體).

* Actor validation uses multiple converging signals (NER type, SVO
  participation, frequency, recurrence) rather than any single heuristic.
* NER type supports actorhood but is not mandatory; structural SVO evidence
  can compensate for missing or incorrect NER labels.
* Canonicalization is conservative -- prefer false negatives over false merges.
* All statistics have explicit denominators and zero-denominator guards.
* No hard-coded vocabulary, blacklists, or fixture-specific aliases.
* Deterministic -- same corpus in any order produces the same canonical actors.
* Provenance is preserved throughout; every statistic is traceable to its
  source (article, sentence, mention, role, verb, passive flag, modifiers).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import NamedTuple

from pydantic import BaseModel, Field

from news_deframe.schemas import ParsedArticle
from news_deframe.parser.predicate_normalization import is_valid_predicate_token


# Labels that represent human-scale actors or structured social entities.
_ACTOR_NER_LABELS: frozenset[str] = frozenset(
    {"PERSON", "PER", "ORG", "GPE", "NORP", "FAC", "LOC"}
)

# Labels that are never actors by type (temporal, quantitative, abstract, or verb).
_NON_ACTOR_NER_LABELS: frozenset[str] = frozenset(
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
        "EVENT",
        "VERB_ACTION",
        "EVENT_NOUN",
    }
)

# spaCy non-actor NER labels (excludes synthetic modifier tags like EVENT_NOUN)
_SPACY_NON_ACTOR_NER_LABELS: frozenset[str] = frozenset(
    {
        "FAC",
        "LOC",
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
        "EVENT",
    }
)

_STRUCTURAL_NON_ACTOR_ENDINGS = (
    # Locations / Spatial relators
    "現場", "廣場", "區域", "道路", "外圍", "範圍", "附近", "方向", "路", "街", "館", "樓", "處", "市中心",
    "廠", "中心", "園區", "保護區", "基地", "變電所", "所", "站",
    "street", "square", "road", "avenue", "area", "direction", "zone", "district", "place", "site", "venue",
    "station", "facility", "center", "plant", "base", "park",
    # Events / Actions / Processes / Deadlocks / Natural disasters
    "活動", "示威", "集會", "推擠", "衝突", "逮捕", "行動", "勤務", "調查", "過程", "經過", "情形", "事件",
    "結果", "僵局", "大戰", "協商", "審查", "共識", "爭議", "推動", "政務", "義務", "制度", "原則", "政策", "決議", "延宕", "審查延宕",
    "施政", "表決", "爭辯", "比讚", "讚",
    "地震", "強震", "餘震", "主震", "海嘯", "爆炸", "火災", "大火", "火警", "火勢", "煙霧", "濃煙", "氣爆", "土石流", "風災", "水災",
    "暴雨", "颱風", "坍方", "災情", "災害", "災難", "事故", "車禍", "出軌", "翻覆", "停電", "停水", "斷電", "斷水", "破裂", "落差", "落石",
    "protest", "rally", "demonstration", "clash", "arrest", "process", "investigation", "procedure", "incident", "operation",
    "earthquake", "tsunami", "explosion", "fire", "wildfire", "storm", "flood", "disaster", "accident", "blackout", "outage",
    # Fiscal / Budgetary / Statutory concepts
    "預算", "預算案", "總預算", "總預", "歲出", "歲入", "特別費", "媒宣費", "人事費", "事務費", "經費", "金額", "總額", "歲出總額", "額",
    "薪水", "薪資", "工資", "退休金", "加薪", "資遣費", "罰鍰", "規模", "條例", "法規", "法律", "修正案", "憲政慣例", "慣例", "法", "案",
    # Inanimate physical infrastructure / materials / vehicles / equipment
    "建築", "建築物", "建物", "外牆", "石牆", "牆面", "屋頂", "天花板", "地板", "地面", "鋼筋", "水泥", "磚瓦", "土石", "瓦礫", "碎片",
    "設備", "設施", "貨架", "碎紙機", "文件櫃", "窗框", "門窗", "孔洞", "破損", "裂痕", "煙囪", "橋樑", "高架", "路段", "鐵軌", "軌道",
    "跑道", "車輛", "列車", "班機", "飛機", "直升機", "線路", "網路", "訊號", "通訊", "基地台", "變電箱", "電廠", "核電廠", "水庫", "水壩",
    "building", "structure", "wall", "roof", "debris", "equipment", "facility", "infrastructure", "runway", "train", "plane", "aircraft",
    # Measurements / counts / reporting artifacts / statuses
    "震度", "深度", "級", "公里", "公尺", "人次", "筆", "件", "起", "通報", "報案", "電話", "人數", "傷亡", "人員傷亡",
    "受傷", "受害", "罹難", "失聯", "死亡", "受困",
    # Locative settings / Activity concepts
    "縣內", "市內", "屋內", "境內", "國內", "區內", "廠內", "店內", "校內", "院內",
    "旅遊", "旅行", "觀光", "交通",
    # Abstract concepts / states / injuries / conduct / outcomes / media
    "計畫", "規定", "秩序", "意見", "權利", "說法", "方式", "資料", "影像", "畫面", "支援", "溝通", "擦傷", "大礙", "傷勢", "傷害", "死傷",
    "舉止", "言行", "舉動", "行為", "言行舉止", "完整", "片段",
    "plan", "rule", "order", "opinion", "right", "statement", "method", "data", "footage", "image", "injury", "scratch", "support",
    # Pronoun / discourse fragments
    "他代表", "大家", "彼此", "雙方", "各方", "本身", "部分", "方面", "前", "後", "時", "上面", "下面", "拖", "共計", "該", "減列", "增列",
)

_STRUCTURAL_ACTOR_ENDINGS = (
    "者", "人", "員", "團體", "警方", "局", "署", "黨", "黨團", "隊", "眾", "群", "部", "府", "院", "師", "官",
    "單位", "當局", "警消", "公務人員", "警察", "教師", "學生", "市民", "民眾", "人民", "朝野", "朝野黨團", "三黨團",
    "立委", "議員", "發言人", "召集人", "部長", "院長", "主席", "總召", "參選人", "委員", "首長", "校長", "代表",
    "police", "officers", "protesters", "demonstrators", "organizers", "witnesses", "citizens", "crowd",
    "authorities", "government", "council", "department", "court", "union", "spokesperson", "guard", "guards",
    "body", "team", "corps", "agency", "group", "minister", "president", "chairman", "director", "senator", "representative",
)

_UNACCUSATIVE_LOCATIVE_VERBS = frozenset({
    "發生", "舉行", "觀測", "測得", "感受", "搖晃", "震度", "停電", "受損", "倒塌", "起火", "起",
    "出現", "位於", "傳出", "有", "遭", "突發", "happen", "occur", "take place", "appear", "stand", "located", "experience",
})



# Sentinel NER type for SVO-derived candidates that lack a NER label.
_SVO_DERIVED_TYPE: str = "SVO_PARTICIPANT"


# -- Span structural validation -----------------------------------------------


def _is_pure_quantity_or_date(text: str) -> bool:
    """Return True if text is a quantity, date, time, or numerical count rather than a participant."""
    stripped = text.strip()
    if not stripped:
        return True
    # Digits + unit/classifier
    if re.match(r"^(\d+|[一二三四五六七八九十百千萬億兆]+)\s*(人|元|天|年|月|日|號|分|點|時|%|成|度|案)?$", stripped):
        return True
    if re.match(r"^\d+年\d+月\d+日$", stripped):
        return True
    if re.match(r"^\d+(\.\d+)?%$", stripped):
        return True
    if any(stripped.endswith(s) for s in ("億元", "萬元", "億", "萬", "元", "天", "日", "年度", "年", "月")):
        if not any(stripped.endswith(a) for a in _STRUCTURAL_ACTOR_ENDINGS):
            return True
    return False


def _is_valid_surface(text: str) -> bool:
    """Return True when *text* is structurally plausible as an actor name.

    Checks are based on linguistic/structural properties only -- no vocabulary.
    """
    stripped = text.strip()
    if len(stripped) < 2:
        return False

    if _is_pure_quantity_or_date(stripped):
        return False

    # Must contain at least one letter or CJK character
    has_word_char = any(
        ch.isalpha() or (0x4E00 <= ord(ch) <= 0x9FFF) or (0x3400 <= ord(ch) <= 0x4DBF)
        for ch in stripped
    )
    if not has_word_char:
        return False

    # First and last char must not be pure punctuation/symbol
    first_cat = unicodedata.category(stripped[0])
    last_cat = unicodedata.category(stripped[-1])
    if first_cat.startswith("P") or first_cat.startswith("S"):
        return False
    if last_cat.startswith("P") or last_cat.startswith("S"):
        return False

    return True


def _is_valid_candidate_length(text: str, *, max_tokens: int = 8) -> bool:
    """Reject spans that are suspiciously long (likely parser fragments)."""
    token_count = len(text.split())
    if token_count == 1 and len(text) > 20:
        return False
    return token_count <= max_tokens


def _is_structurally_valid_ner_span(entity_name: str, entity_type: str) -> bool:
    """Structural validation for NER-derived candidates.

    Beyond type-level checks, validates that the surface form is plausible as
    an actor name using character-level structural properties only.
    Rejects very short spans, all-digit spans, and spans that contain no
    word characters.
    """
    text = entity_name.strip()
    if not _is_valid_surface(text):
        return False
    if not _is_valid_candidate_length(text):
        return False
    return True


def _is_valid_svo_span(span_text: str) -> bool:
    """Structural validation for SVO-derived participant spans.

    Returns True when *span_text* is plausible as a participant:
    - Contains at least one CJK character or alphabetic word char
    - Is not excessively long (protects against parser subtree artifacts)
    - Does not consist solely of digits or punctuation
    """
    text = span_text.strip()
    if len(text) < 2:
        return False

    if _is_pure_quantity_or_date(text):
        return False

    # Character length limit for a participant span
    if len(text) > 30:
        return False

    # Token count limit (space-separated tokens for English; CJK is dense)
    token_count = len(text.split())
    if token_count > 6:
        return False

    has_word_char = any(
        ch.isalpha() or (0x4E00 <= ord(ch) <= 0x9FFF) or (0x3400 <= ord(ch) <= 0x4DBF)
        for ch in text
    )
    if not has_word_char:
        return False

    # First and last char must not be punctuation
    first_cat = unicodedata.category(text[0])
    last_cat = unicodedata.category(text[-1])
    if first_cat.startswith("P") or first_cat.startswith("S"):
        return False
    if last_cat.startswith("P") or last_cat.startswith("S"):
        return False

    return True


# -- Actor mention: a single SVO-grounded occurrence --------------------------


class ActorMention(NamedTuple):
    """A single occurrence of an actor in a specific grammatical role.

    Provenance fields allow every aggregated statistic to be traced to source.
    """

    article_id: str
    sentence: str
    surface: str       # original surface form from NER or SVO span
    role: str          # "agent", "patient", or "modifier_only"
    verb: str          # lemma of the associated verb (empty if modifier_only)
    is_passive: bool   # whether the containing SVO record is passive
    modifiers: list    # evaluative modifiers attached to this mention


# -- Canonical actor: event-level resolved identity ---------------------------


class CanonicalActor(BaseModel):
    """Event-level canonical actor with aggregated framing evidence."""

    canonical_name: str = Field(..., description="Chosen canonical surface form")
    entity_type: str = Field(..., description="NER label of the canonical form")
    surface_mentions: list[str] = Field(
        default_factory=list, description="All distinct surface forms mapped here"
    )
    mentions: list = Field(
        default_factory=list, description="All SVO-grounded occurrences (ActorMention)"
    )
    article_ids: list[str] = Field(
        default_factory=list, description="Distinct articles containing this actor"
    )


# -- Actor role statistics (per canonical actor x article) --------------------


class ActorRoleStats(BaseModel):
    """Strongly typed framing statistics for one canonical actor in one article.

    Denominator contract
    --------------------
    role_occurrence_count = agent_count + patient_count

    agent_ratio = agent_count / role_occurrence_count  (0.0 when denom = 0)
    patient_ratio = patient_count / role_occurrence_count  (0.0 when denom = 0)
    passive_patient_ratio = passive_patient_count / patient_count  (0.0 when denom = 0)
    """

    canonical_name: str
    article_id: str

    # Raw counts
    mention_count: int = Field(default=0, ge=0)
    role_occurrence_count: int = Field(
        default=0, ge=0, description="agent_count + patient_count"
    )
    agent_count: int = Field(default=0, ge=0)
    patient_count: int = Field(default=0, ge=0)
    passive_patient_count: int = Field(
        default=0, ge=0, description="patient occurrences in passive constructions"
    )

    # Normalized ratios (denominator = role_occurrence_count)
    agent_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    patient_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    passive_patient_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    # Verb evidence (lemmas, deduplicated, order-stable)
    associated_agent_verbs: list[str] = Field(default_factory=list)
    associated_patient_verbs: list[str] = Field(default_factory=list)

    # Modifier evidence
    associated_modifiers: list[str] = Field(default_factory=list)

    # Provenance
    provenance: list = Field(
        default_factory=list,
        description="Source ActorMention tuples this profile was aggregated from",
    )


# -- Stage 1a: NER candidate extraction with structural validation -------------


def _extract_ner_candidates(article: ParsedArticle) -> list:
    """Return (surface, ner_type, modifiers) triples for NER actor candidates.

    Applies structural validation in addition to type filtering to reject
    spans that the NER model has misclassified (e.g. adjectives tagged as
    PERSON or FAC by a low-accuracy Chinese model).
    """
    results = []
    for em in article.entity_modifiers:
        if em.entity_type in _NON_ACTOR_NER_LABELS:
            continue
        if em.entity_type not in _ACTOR_NER_LABELS:
            continue
        surface = em.entity_name.strip()
        if not _is_structurally_valid_ner_span(surface, em.entity_type):
            continue
        results.append((surface, em.entity_type, list(em.modifiers)))
    return results


def _extract_svo_candidates(article: ParsedArticle) -> list:
    """Return (surface, ner_type, modifiers) triples derived from SVO subjects/objects.

    This path recovers participants that NER fails to tag. The ner_type is
    set to _SVO_DERIVED_TYPE (a sentinel). These candidates are validated
    primarily by structural evidence (surface form, SVO recurrence across
    articles) rather than NER label.

    Spans that were explicitly identified by the entity extractor as non-actor
    entity types (DATE, TIME, CARDINAL, EVENT, LAW, PRODUCT, EVENT_NOUN, etc.)
    are excluded so that non-actor entities are not resurrected via SVO.

    Each unique surface form is emitted at most once per article.
    """
    non_actor_keys = {
        _normalize_key(em.entity_name)
        for em in article.entity_modifiers
        if em.entity_type in _SPACY_NON_ACTOR_NER_LABELS
    }

    seen: set[str] = set()
    results = []

    for record in article.svo_records:
        for span in record.subjects + record.objects:
            text = span.strip()
            # Strip leading numerical quantifiers and counters (e.g. '4起民眾' -> '民眾', '30名員工' -> '員工')
            text = re.sub(r"^\d+\s*(?:起|處|間|棟|名|人|個|戶|項|家|列|架|班|艘|位)\s*", "", text).strip()
            if text in seen:
                continue
            if _normalize_key(text) in non_actor_keys:
                continue
            if not _is_valid_svo_span(text):
                continue
            seen.add(text)
            results.append((text, _SVO_DERIVED_TYPE, []))

    return results


# -- Stage 2: SVO matching ----------------------------------------------------


def _normalize_for_matching(text: str) -> str:
    """Lowercase and strip for matching."""
    return text.strip().lower()


def _candidate_in_span(candidate: str, span: str) -> bool:
    """Return True when candidate is present in span (case-insensitive).

    Uses word-boundary matching for ASCII to reduce false positives.
    For CJK, allows substring matching while preventing pure locative/brand modifiers
    (e.g. '日本' in '日本首相高市早苗', '日本氣象廳', '日本郵便', '熊本' in '永旺夢樂城熊本', '熊本城外牆')
    from falsely capturing predicates belonging to specific corporate entities, institutions, or officials.
    """
    norm_cand = _normalize_for_matching(candidate)
    norm_span = _normalize_for_matching(span)

    if norm_cand not in norm_span:
        return False

    if norm_cand == norm_span:
        return True

    has_cjk = any(
        (0x4E00 <= ord(ch) <= 0x9FFF) or (0x3400 <= ord(ch) <= 0x4DBF)
        for ch in norm_cand
    )
    if has_cjk:
        distinct_compound_suffixes = (
            "警局", "消防局", "警察", "警消", "警方", "氣象廳", "事務所", "公所", "工廠", "商場", "機場", "分局", "總局",
            "車站", "站", "港", "門市", "外牆", "石牆", "城", "廠", "橋", "交流道", "夢樂城",
            "放送", "電視台", "新聞", "報社", "日報", "通訊社", "廣播", "航空", "郵便", "電力",
            "公司", "會社", "集團", "中心", "園", "所", "署", "廳", "院", "部", "局",
            "首相", "總統", "市長", "縣長", "部長", "院長", "署長", "局長", "處長", "主席", "總召", "立委", "議員",
            "知事", "相關人士", "人士", "店員", "職員", "民眾", "市民", "長官", "閣員", "大臣",
        )
        valid_continuation_affixes = (
            "縣", "市", "省", "州", "町", "村", "區", "政府", "當局", "官方", "隊", "代表團", "團", "黨團", "本部",
        )

        # If span ends with candidate (e.g. 行政院長卓榮泰 ends with 卓榮泰, 日本氣象廳 ends with 氣象廳)
        if norm_span.endswith(norm_cand):
            prefix = norm_span[:-len(norm_cand)].strip()
            if any(prefix.endswith(t) or prefix.startswith(t) for t in (
                "長", "主席", "總召", "立委", "議員", "部長", "院長", "署長", "局長", "處長",
                "市長", "縣長", "首相", "總統", "黨", "黨團", "朝野", "在野", "執政",
            )):
                return True
            if len(prefix) <= 2:
                return True
            return False

        # If span starts with candidate (e.g. candidate='熊本', span='熊本縣今日發生強震', candidate='索比安', span='索比安當局逮捕...')
        if norm_span.startswith(norm_cand):
            rem = norm_span[len(norm_cand):].strip()
            if any(rem.startswith(d) for d in distinct_compound_suffixes):
                return False
            matched_affix = next((a for a in valid_continuation_affixes if rem.startswith(a)), None)
            if matched_affix:
                rem_after = rem[len(matched_affix):].strip()
                if any(rem_after.startswith(d) for d in distinct_compound_suffixes):
                    return False
                return True
            if len(norm_cand) <= 4:
                return False
            return True

        # If candidate is inside span (e.g. '高市' in '日本首相高市早苗')
        if norm_cand in norm_span:
            idx = norm_span.find(norm_cand)
            prefix = norm_span[:idx].strip()
            rem = norm_span[idx + len(norm_cand):].strip()
            if any(prefix.endswith(t) for t in ("長", "主席", "總召", "立委", "議員", "部長", "院長", "署長", "局長", "處長", "市長", "縣長", "首相", "總統")):
                if not rem or rem in ("早苗", "院長", "部長", "先生", "女士"):
                    return True
            return False

        return True

    # English / Latin: word-boundary and compound entity protection
    pattern = r"(?<![a-z0-9_])" + re.escape(norm_cand) + r"(?![a-z0-9_])"
    if not re.search(pattern, norm_span):
        return False

    # Prevent country/city matching within compound company/institution names (e.g. 'Japan' in 'Japan Airlines')
    words_cand = norm_cand.split()
    words_span = norm_span.split()
    if len(words_span) > len(words_cand):
        org_indicators = {"agency", "airlines", "airport", "association", "authority", "bank", "broadcasting", "bureau", "center", "centre", "company", "corp", "corporation", "council", "court", "department", "force", "group", "hospital", "institute", "ministry", "museum", "news", "office", "party", "police", "post", "press", "service", "station", "times", "union", "university"}
        span_indicators = set(words_span) & org_indicators
        cand_indicators = set(words_cand) & org_indicators
        if span_indicators and not cand_indicators:
            return False

    return True

def _match_candidate_to_svo(
    candidate: str,
    modifiers: list,
    article: ParsedArticle,
) -> list:
    """Produce SVO-grounded ActorMention records for candidate in article.

    Passive role inversion:
    - Passive grammatical subject -> logical patient
    - Passive grammatical object  -> logical agent
    """
    mentions = []

    for record in article.svo_records:
        in_subjects = any(_candidate_in_span(candidate, s) for s in record.subjects)
        in_objects = any(_candidate_in_span(candidate, o) for o in record.objects)

        if not (in_subjects or in_objects):
            continue

        verb = record.verb or ""
        if verb and not is_valid_predicate_token(verb):
            continue

        # Unaccusative occurrence event check: geographic settings of natural occurrences are not active agents
        is_unaccusative = verb in _UNACCUSATIVE_LOCATIVE_VERBS or any(
            verb.startswith(v) for v in ("發生", "出現", "傳出", "爆發", "震度", "搖晃", "happen", "occur", "take place")
        )

        if record.is_passive:
            if in_subjects:
                mentions.append(ActorMention(
                    article_id=article.article_id,
                    sentence=record.sentence,
                    surface=candidate,
                    role="patient",
                    verb=verb,
                    is_passive=True,
                    modifiers=list(modifiers),
                ))
            if in_objects:
                mentions.append(ActorMention(
                    article_id=article.article_id,
                    sentence=record.sentence,
                    surface=candidate,
                    role="agent",
                    verb=verb,
                    is_passive=True,
                    modifiers=list(modifiers),
                ))
        elif is_unaccusative:
            if in_subjects:
                mentions.append(ActorMention(
                    article_id=article.article_id,
                    sentence=record.sentence,
                    surface=candidate,
                    role="patient",
                    verb=verb,
                    is_passive=False,
                    modifiers=list(modifiers),
                ))
            if in_objects:
                mentions.append(ActorMention(
                    article_id=article.article_id,
                    sentence=record.sentence,
                    surface=candidate,
                    role="patient",
                    verb=verb,
                    is_passive=False,
                    modifiers=list(modifiers),
                ))
        else:
            if in_subjects:
                mentions.append(ActorMention(
                    article_id=article.article_id,
                    sentence=record.sentence,
                    surface=candidate,
                    role="agent",
                    verb=verb,
                    is_passive=False,
                    modifiers=list(modifiers),
                ))
            if in_objects:
                mentions.append(ActorMention(
                    article_id=article.article_id,
                    sentence=record.sentence,
                    surface=candidate,
                    role="patient",
                    verb=verb,
                    is_passive=False,
                    modifiers=list(modifiers),
                ))

    return mentions


# -- Stage 3: Actor validation ------------------------------------------------


def _validate_actor(
    surface: str,
    ner_type: str,
    mentions: list,
    total_article_count: int,
    cross_article_frequency: int,
) -> bool:
    """Apply multi-signal actor validation.

    Combines:
    - Structural morphology & non-actor semantic category discrimination
    - Role-grounded verb transitivity
    - NER label signals
    - Cross-outlet recurrence
    """
    if not _is_valid_surface(surface):
        return False
    if ner_type in _NON_ACTOR_NER_LABELS:
        return False

    # Pronoun / discourse prefix check
    if surface.startswith(("他", "她", "你", "我", "其", "這", "那")):
        return False

    s_lower = surface.strip().lower()

    # Structural non-actor check (locations, events, actions, abstract concepts, injuries, fiscal items)
    if any(s_lower.endswith(na) for na in _STRUCTURAL_NON_ACTOR_ENDINGS):
        if not any(s_lower.endswith(a) for a in (
            "警局", "分局", "總局", "管理局", "調查局", "立法院", "行政院", "監察院", "教育部", "法院",
            "代表", "委員", "部長", "院長", "主席", "總召", "立委", "議員", "官員", "警察", "警消",
            "市民", "民眾", "人民", "團隊", "黨團", "署", "隊", "師", "團體",
            "police", "officers", "protesters", "demonstrators", "organizers", "witnesses", "citizens", "crowd",
            "authorities", "government", "council", "department", "court", "union", "spokesperson", "guard", "guards",
        )):
            return False

    # Broken clausal / parser fragments check
    if any(c in s_lower for c in ("對", "憑", "難", "均", "間", "離開", "突破", "依照", "服從", "並")):
        if not any(s_lower.endswith(a) for a in _STRUCTURAL_ACTOR_ENDINGS):
            return False

    # Check unaccusative locative verbs:
    # If candidate only ever appears as subject of 發生 / 舉行 without other actions, reject as setting
    verbs = [m.verb for m in mentions if m.verb]
    if verbs and all(v in _UNACCUSATIVE_LOCATIVE_VERBS or any(v.startswith(u) for u in ("發生", "出現", "傳出", "爆發", "震度", "搖晃")) for v in verbs) and ner_type not in {"PERSON", "PER", "ORG", "NORP"}:
        return False

    s1 = ner_type in _ACTOR_NER_LABELS
    s2 = len(mentions) > 0
    s3 = cross_article_frequency > 1
    s4 = any(m.verb for m in mentions)
    s5 = len(mentions) >= 2

    # Check actor morphology
    has_actor_morphology = any(s_lower.endswith(a) for a in _STRUCTURAL_ACTOR_ENDINGS) or any(
        w.endswith(("er", "ers", "or", "ors", "ist", "ists", "ant", "ants", "ian", "ians", "men", "women"))
        for w in s_lower.split()
    )

    score = sum([s1, s2, s3, s4, s5, has_actor_morphology])

    if s1 and score >= 2:
        return True
    if not s1 and (has_actor_morphology or score >= 3):
        return True

    return False


# -- Stage 4: Event-level canonicalization ------------------------------------


def _normalize_key(text: str) -> str:
    """Deterministic normalization key for canonicalization.

    Language-general transformations only:
    1. Strip whitespace
    2. Lowercase
    3. Remove leading English determiners (a/an/the)
    4. Collapse internal whitespace
    """
    text = text.strip().lower()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard token overlap between two surface forms."""
    set_a = set(_normalize_key(a).split())
    set_b = set(_normalize_key(b).split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _is_abbreviation_of(short: str, long_form: str) -> bool:
    """Return True when short is a plausible acronym of long_form (>= 2 chars)."""
    short_up = short.upper().strip()
    if len(short_up) < 2:
        return False
    words = [w for w in long_form.split() if w and w[0].isalpha()]
    initials = "".join(w[0].upper() for w in words)
    return short_up in initials and len(short_up) >= 2


def _type_group(t: str) -> str:
    """Map NER label to a broad compatibility group."""
    if t in {"PERSON", "PER"}:
        return "person"
    if t == "ORG":
        return "org"
    if t in {"GPE", "LOC"}:
        return "geo"
    if t == "NORP":
        return "group"
    if t == "FAC":
        return "fac"
    if t == _SVO_DERIVED_TYPE:
        return "svo"
    return t


def _should_merge(surface_a: str, type_a: str, surface_b: str, type_b: str) -> bool:
    """Return True when two candidates should share a canonical actor.

    Conservative: requires compatible NER types AND at least one strong
    lexical/abbreviation signal or title+name subsumption.
    """
    group_a = _type_group(type_a)
    group_b = _type_group(type_b)

    # Never merge distinct entity classes (e.g. PERSON vs ORG, GEO vs ORG, GEO vs PERSON)
    if group_a != "svo" and group_b != "svo" and group_a != group_b:
        return False

    key_a = _normalize_key(surface_a)
    key_b = _normalize_key(surface_b)

    if key_a == key_b:
        return True

    # High token overlap (Jaccard >= 0.75) for multi-word phrases of same group
    if _token_overlap_ratio(surface_a, surface_b) >= 0.75:
        return True

    # 1. PERSON: Title + Name subsumption (e.g. 行政院長卓榮泰 <-> 卓榮泰, President Biden <-> Biden)
    if (group_a == "person" and group_b == "person") or ("svo" in {group_a, group_b} and {group_a, group_b} <= {"person", "svo"}):
        for long_k, short_k, long_surf, short_surf in ((key_a, key_b, surface_a, surface_b), (key_b, key_a, surface_b, surface_a)):
            if len(short_k) >= 2 and len(long_k) > len(short_k):
                # Suffix / title match (e.g. 行政院長卓榮泰 <-> 卓榮泰)
                if long_k.endswith(short_k):
                    prefix = long_k[:-len(short_k)].strip()
                    if any(prefix.endswith(t) or prefix.startswith(t) for t in (
                        "長", "主席", "總召", "立委", "議員", "參選人", "部長", "院長", "代表", "署長", "局長", "處長",
                        "president", "minister", "senator", "representative", "director", "chair", "chairman", "spokesperson",
                        "首相", "總統", "市長", "縣長",
                    )):
                        return True
                    if len(prefix) <= 2:
                        return True
                # Truncated proper name prefix within same entity stem (e.g. 周曉 <-> 周曉芸, 韓國 <-> 韓國瑜)
                if long_k.startswith(short_k) and len(long_k) - len(short_k) <= 2:
                    return True
                # Embedded title + name match (e.g. 立法院長韓國瑜 <-> 韓國瑜 or 韓國)
                if short_k in long_k:
                    for t in ("長", "主席", "總召", "立委", "議員", "參選人", "部長", "院長", "代表", "署長", "局長", "處長", "首相", "總統"):
                        if t in long_k:
                            idx = long_k.rfind(t)
                            name_part = long_k[idx + len(t):]
                            if len(name_part) >= 2 and (short_k == name_part or (len(short_k) == 2 and short_k in name_part) or (len(name_part) >= 2 and name_part.startswith(short_k))):
                                return True
        return False

    # 2. ORG: Legislative caucus of SAME party (e.g. 民進黨團 <-> 民進黨, 朝野黨團 <-> 朝野)
    if (group_a == "org" and group_b == "org") or ("svo" in {group_a, group_b} and {group_a, group_b} <= {"org", "svo"}):
        for long_k, short_k in ((key_a, key_b), (key_b, key_a)):
            if long_k.startswith(short_k) and any(long_k[len(short_k):] == s for s in ("黨團", "各黨團", "黨", "立院黨團")):
                return True
            if ("黨團" in long_k or "黨" in long_k) and ("黨團" in short_k or "黨" in short_k):
                base_long = long_k.replace("立院", "").replace("各", "")
                base_short = short_k.replace("立院", "").replace("各", "")
                if base_long.endswith("黨團") and base_short.endswith("黨"):
                    if base_long[:-2] == base_short[:-1]:
                        return True
                if base_long.endswith("黨團") and base_short.endswith("黨團"):
                    if base_long == base_short:
                        return True
                if base_long == "朝野黨團" and base_short in ("朝野", "朝野各黨團"):
                    return True
                if base_short == "民進" and "民進黨" in base_long:
                    return True
            # Abbreviation relationship for organizations (e.g. 立院 <-> 立法院)
            if _is_abbreviation_of(short_k, long_k):
                return True
        return False

    # 3. GEO: Strict administrative suffix of SAME base place (e.g. 熊本縣 <-> 熊本, 福岡縣 <-> 福岡)
    if (group_a == "geo" and group_b == "geo") or ("svo" in {group_a, group_b} and {group_a, group_b} <= {"geo", "svo"}):
        for long_k, short_k in ((key_a, key_b), (key_b, key_a)):
            if len(short_k) >= 2 and len(long_k) == len(short_k) + 1:
                if long_k.startswith(short_k) and long_k[-1] in ("縣", "市", "省", "州", "町", "村", "區"):
                    return True
        return False

    return False


def _canonicalize_actors(validated: list) -> list:
    """Group validated (surface, ner_type, mentions) triples into CanonicalActors.

    Uses union-find over a deterministically sorted list of candidates.
    Canonical name chosen by mention frequency (tie-break: lexicographic).

    When a group contains both NER-backed and SVO-derived candidates, the
    canonical entity_type is chosen from NER-backed entries (majority vote
    among NER labels only; SVO_PARTICIPANT is used only as a fallback).
    """
    if not validated:
        return []

    # Sort for determinism
    sorted_cands = sorted(validated, key=lambda x: _normalize_key(x[0]))
    n = len(sorted_cands)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            surf_i, type_i, _ = sorted_cands[i]
            surf_j, type_j, _ = sorted_cands[j]
            if _should_merge(surf_i, type_i, surf_j, type_j):
                union(i, j)

    groups: dict = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    canonical_actors = []
    for root, indices in sorted(groups.items()):
        members = [sorted_cands[i] for i in indices]
        surfaces = [m[0] for m in members]

        # Choose canonical name: prefer specific proper name / title over bare fragments
        surface_counter: Counter = Counter()
        for _, _, mention_list in members:
            for mention in mention_list:
                surface_counter[mention.surface] += 1
        for surf in surfaces:
            if surf not in surface_counter:
                surface_counter[surf] = 0

        canonical_name = max(
            surface_counter.keys(),
            key=lambda s: (surface_counter[s], len(_normalize_key(s))),
        )

        all_mentions = []
        for _, _, mention_list in members:
            all_mentions.extend(mention_list)

        # Prefer NER-backed type over SVO_PARTICIPANT
        ner_backed_types = [m[1] for m in members if m[1] != _SVO_DERIVED_TYPE]
        if ner_backed_types:
            type_counts: Counter = Counter(ner_backed_types)
        else:
            type_counts = Counter(m[1] for m in members)
        entity_type = type_counts.most_common(1)[0][0]

        article_ids = sorted({m.article_id for m in all_mentions})

        canonical_actors.append(
            CanonicalActor(
                canonical_name=canonical_name,
                entity_type=entity_type,
                surface_mentions=sorted(set(surfaces)),
                mentions=all_mentions,
                article_ids=article_ids,
            )
        )

    return canonical_actors


# -- Stage 5: Role aggregation ------------------------------------------------


def _deduplicate_ordered(items: list) -> list:
    """Remove duplicates while preserving insertion order."""
    seen: set = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _aggregate_role_stats(
    actor: CanonicalActor,
    article_id: str,
    article_modifiers: list,
) -> ActorRoleStats:
    """Compute ActorRoleStats for actor in a specific article.

    Denominator: role_occurrence_count = agent_count + patient_count.
    All ratios are 0.0 when the denominator is zero.
    """
    from news_deframe.parser.predicate_normalization import is_valid_predicate_token

    article_mentions = [m for m in actor.mentions if m.article_id == article_id]

    mention_count = len(article_mentions)
    agent_count = sum(1 for m in article_mentions if m.role == "agent")
    patient_count = sum(1 for m in article_mentions if m.role == "patient")
    passive_patient_count = sum(
        1 for m in article_mentions if m.role == "patient" and m.is_passive
    )

    role_occurrence_count = agent_count + patient_count

    agent_ratio = round(agent_count / role_occurrence_count, 4) if role_occurrence_count > 0 else 0.0
    patient_ratio = round(patient_count / role_occurrence_count, 4) if role_occurrence_count > 0 else 0.0
    passive_patient_ratio = round(passive_patient_count / patient_count, 4) if patient_count > 0 else 0.0

    agent_verbs = _deduplicate_ordered(
        [m.verb for m in article_mentions if m.role == "agent" and m.verb and is_valid_predicate_token(None, m.verb)]
    )
    patient_verbs = _deduplicate_ordered(
        [m.verb for m in article_mentions if m.role == "patient" and m.verb and is_valid_predicate_token(None, m.verb)]
    )

    modifier_sources = list(article_modifiers)
    for m in article_mentions:
        modifier_sources.extend(m.modifiers)
    all_modifiers = _deduplicate_ordered(modifier_sources)

    return ActorRoleStats(
        canonical_name=actor.canonical_name,
        article_id=article_id,
        mention_count=mention_count,
        role_occurrence_count=role_occurrence_count,
        agent_count=agent_count,
        patient_count=patient_count,
        passive_patient_count=passive_patient_count,
        agent_ratio=agent_ratio,
        patient_ratio=patient_ratio,
        passive_patient_ratio=passive_patient_ratio,
        associated_agent_verbs=agent_verbs,
        associated_patient_verbs=patient_verbs,
        associated_modifiers=all_modifiers,
        provenance=article_mentions,
    )


# -- Stage 6: Importance ranking ----------------------------------------------


def _compute_importance(actor: CanonicalActor, total_articles: int) -> float:
    """Deterministic importance score based on structural signals only.

    Signals: cross-outlet presence, role-grounded mention count, verb diversity.
    No sentiment, political, or domain-specific weights.
    """
    outlet_fraction = len(actor.article_ids) / max(total_articles, 1)
    total_role_mentions = sum(1 for m in actor.mentions if m.role in {"agent", "patient"})
    distinct_verbs = len({m.verb for m in actor.mentions if m.verb})

    return (
        outlet_fraction * 4.0
        + min(total_role_mentions, 20) * 0.5
        + min(distinct_verbs, 10) * 0.3
    )


# -- Public API ---------------------------------------------------------------


def resolve_actors(
    articles: list,
    *,
    min_outlets: int = 1,
    max_actors_console: int = 20,
) -> tuple:
    """Run the full actor resolution pipeline over an event corpus.

    Parameters
    ----------
    articles : list[ParsedArticle]
        All parsed articles in the event corpus.
    min_outlets : int
        Minimum distinct articles an actor must appear in (default 1).
    max_actors_console : int
        Soft cap for console display; JSON export retains all.

    Returns
    -------
    (canonical_actors, role_stats)
        canonical_actors -- list[CanonicalActor] sorted by importance desc.
        role_stats       -- list[ActorRoleStats], one per (actor x article).
    """
    total_articles = len(articles)
    article_ids = sorted(a.article_id for a in articles)

    # Collect all surface keys across the corpus that were explicitly identified
    # as non-actor entity types (DATE, TIME, CARDINAL, EVENT, LAW, PRODUCT, etc.)
    corpus_non_actor_keys: set[str] = {
        _normalize_key(em.entity_name)
        for a in articles
        for em in a.entity_modifiers
        if em.entity_type in _SPACY_NON_ACTOR_NER_LABELS
    }


    raw_candidates: dict = defaultdict(list)
    key_article_set: dict = defaultdict(set)
    surface_modifiers: dict = defaultdict(list)

    for article in articles:
        # Path A: NER-backed candidates (with structural validation)
        ner_candidates = _extract_ner_candidates(article)
        # Path B: SVO-derived candidates (structural validation only)
        svo_candidates = _extract_svo_candidates(article)

        # Merge: if the same surface already comes from NER, skip the SVO
        # duplicate so that the NER-backed entry retains its type label.
        # Also exclude spans known across the corpus to be non-actor types.
        ner_surfaces: set[str] = {_normalize_key(c[0]) for c in ner_candidates}
        deduplicated_svo = [
            c for c in svo_candidates
            if _normalize_key(c[0]) not in ner_surfaces
            and _normalize_key(c[0]) not in corpus_non_actor_keys
        ]

        all_candidates = ner_candidates + deduplicated_svo


        for surface, ner_type, modifiers in all_candidates:
            nkey = _normalize_key(surface)
            mentions = _match_candidate_to_svo(surface, modifiers, article)
            raw_candidates[nkey].append((surface, ner_type, mentions))
            key_article_set[nkey].add(article.article_id)
            surface_modifiers[(article.article_id, surface)].extend(modifiers)

    validated = []
    seen_surf_type: set = set()

    for nkey, candidate_list in sorted(raw_candidates.items()):
        cross_article_freq = len(key_article_set[nkey])
        for surface, ner_type, mentions in candidate_list:
            key = (_normalize_key(surface), ner_type)
            if key in seen_surf_type:
                # Accumulate mentions into existing validated entry
                for entry in validated:
                    if _normalize_key(entry[0]) == _normalize_key(surface) and entry[1] == ner_type:
                        entry[2].extend(mentions)
                continue
            if _validate_actor(surface, ner_type, mentions, total_articles, cross_article_freq):
                seen_surf_type.add(key)
                validated.append([surface, ner_type, list(mentions)])

    canonical_actors = _canonicalize_actors(validated)

    canonical_actors = [
        a for a in canonical_actors if len(a.article_ids) >= min_outlets
    ]

    all_stats = []
    for actor in canonical_actors:
        for aid in article_ids:
            actor_modifiers = []
            for surf in actor.surface_mentions:
                actor_modifiers.extend(surface_modifiers.get((aid, surf), []))
            actor_modifiers = _deduplicate_ordered(actor_modifiers)
            stats = _aggregate_role_stats(actor, aid, actor_modifiers)
            all_stats.append(stats)

    canonical_actors.sort(
        key=lambda a: (-_compute_importance(a, total_articles), a.canonical_name)
    )

    return canonical_actors, all_stats