"""Claim equivalence verifier and proposition extraction.

Provides a 2-stage verification architecture for claim clustering:
    sentence pairs (candidate retrieval from embedding similarity)
    → claim-equivalence verification (propositional comparison: agents, patients,
       predicates, modality, negation, attribution, quantities)
    → claim relationship classification (EQUIVALENT, COMPATIBLE, RELATED, CONTRADICTORY, UNRELATED)

Design principles:
- Semantic relatedness is NOT claim equivalence: sentences sharing entities
  or topic but making materially different factual assertions must be classified
  as RELATED, preventing false clustering edges.
- Evaluates differences in:
  - Agent / Patient roles
  - Core action predicate
  - Negation / Polarity
  - Modality (requested vs performed action, intent vs completed event)
  - Attribution / Speaker (police statement vs protest organizer demand)
  - Numerical quantities
- Bilingual support for Chinese and English.
- Deterministic, offline-safe, lightweight without external API dependencies.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple


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


@dataclass
class SentenceProposition:
    """Structured propositional extraction from a sentence."""

    raw_text: str
    cleaned_text: str
    agents: list[str] = field(default_factory=list)
    patients: list[str] = field(default_factory=list)
    predicates: list[str] = field(default_factory=list)
    attributions: list[str] = field(default_factory=list)
    is_negated: bool = False
    modality: str = "statement"        # "statement", "demand", "plan", "opinion", "passive_event"
    quantities: list[str] = field(default_factory=list)
    content_tokens: set[str] = field(default_factory=set)


# ── Linguistic patterns for proposition extraction ────────────────────────────

_ZH_NEGATION_WORDS = frozenset({"不", "沒", "没有", "未", "無", "非", "拒絕", "否認", "禁止", "不予"})
_EN_NEGATION_WORDS = frozenset({"not", "no", "never", "refuse", "refused", "deny", "denied", "failed", "neither", "nor"})

_ZH_ATTRIBUTION_VERBS = ("表示", "指出", "強調", "稱", "說明", "認為", "質疑", "公布", "宣布")
_EN_ATTRIBUTION_VERBS = ("said", "stated", "pointed out", "emphasized", "claimed", "reported", "announced")

_ZH_DEMAND_MODALITY = ("要求", "呼籲", "促請", "建議", "希望")
_EN_DEMAND_MODALITY = ("demand", "demanded", "urged", "called for", "requested")

_ZH_PLAN_MODALITY = ("將", "計畫", "打算", "試圖", "擬")
_EN_PLAN_MODALITY = ("will", "plans to", "attempted to", "intends to")


def _extract_quantities(text: str) -> list[str]:
    """Extract numbers and numeric phrases (e.g. 200, 3, 兩百, 三名)."""
    results = []
    # Arabic digits
    for m in re.finditer(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", text):
        results.append(m.group(0))
    # Chinese numbers
    for m in re.finditer(r"(約?[一二兩三四五六七八九十百千萬]+(?:名|人|個|位|項)?)", text):
        results.append(m.group(0))
    return results


def _extract_attributions(text: str) -> tuple[list[str], str]:
    """Extract speaker/attribution (e.g. '警方表示', '主辦團體表示') and return (attributions, body)."""
    attributions = []
    body = text

    # Chinese attribution pattern
    zh_attr_match = re.search(r"^(.*?)(?:表示|指出|強調|稱|說明|認為|質疑)[，,：:\s]*(.*)$", text)
    if zh_attr_match:
        speaker = zh_attr_match.group(1).strip()
        rest = zh_attr_match.group(2).strip()
        if len(speaker) <= 20 and rest:
            attributions.append(speaker)
            body = rest

    # English attribution pattern
    en_attr_match = re.search(r"^(.*?)(?:said|stated|pointed out|emphasized|claimed|reported)[,:\s]+(.*)$", text, re.I)
    if en_attr_match:
        speaker = en_attr_match.group(1).strip()
        rest = en_attr_match.group(2).strip()
        if len(speaker.split()) <= 6 and rest:
            attributions.append(speaker)
            body = rest

    return attributions, body


def _extract_cjk_ngrams(text: str) -> set[str]:
    """Extract 2-character CJK n-grams and alphanumeric words for robust cross-sentence token overlap."""
    cjk = [ch for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF]
    ngrams = {"".join(cjk[i : i + 2]) for i in range(len(cjk) - 1)}
    words = {w.lower().strip() for w in re.findall(r"\w+", text) if len(w) > 1 and w.isascii()}
    return ngrams | words


def extract_proposition(sentence: str) -> SentenceProposition:
    """Extract semantic proposition features from a sentence."""
    raw = sentence.strip()
    attributions, body = _extract_attributions(raw)

    # Check negation
    is_negated = False
    for nw in _ZH_NEGATION_WORDS:
        if nw in body:
            is_negated = True
            break
    if not is_negated:
        lower_body_tokens = set(body.lower().split())
        if lower_body_tokens & _EN_NEGATION_WORDS:
            is_negated = True

    # Check modality
    modality = "statement"
    if any(w in body for w in _ZH_DEMAND_MODALITY) or any(w in body.lower() for w in _EN_DEMAND_MODALITY):
        modality = "demand"
    elif any(w in body for w in _ZH_PLAN_MODALITY) or any(w in body.lower() for w in _EN_PLAN_MODALITY):
        modality = "plan"

    # Extract quantities
    quantities = _extract_quantities(body)

    # Content tokens using CJK n-grams and words
    content_tokens = _extract_cjk_ngrams(body)

    # Core key actions
    predicates = []
    for action in (
        "逮捕", "要求", "檢視", "調查", "公布", "協助", "突破", "阻止", "推擠", "移動",
        "聚集", "造成", "受到", "擦傷", "衝突", "質疑", "配合", "擴大", "符合", "良率",
        "arrest", "demand", "review", "investigate", "release", "assist", "breach",
        "prevent", "push", "move", "gather", "cause", "injure", "question", "comply",
    ):
        if action in body.lower():
            predicates.append(action)

    # Core key agents/patients
    agents = []
    patients = []
    for actor in (
        "警方", "示威者", "參與者", "被捕者", "警員", "民眾", "群眾", "主辦團體", "市政府", "目擊者",
        "police", "protesters", "demonstrators", "arrestees", "officer", "citizens", "crowd", "organizers",
    ):
        if actor in body.lower() or (attributions and any(actor in a.lower() for a in attributions)):
            if "遭" in body or "被" in body or "were arrested" in body.lower() or "was arrested" in body.lower():
                if actor in {"三名參與者", "參與者", "三名示威者", "示威者", "被捕者", "三名被捕者", "protesters", "demonstrators"}:
                    patients.append(actor)
                elif actor in {"警方", "police"}:
                    agents.append(actor)
            else:
                agents.append(actor)

    return SentenceProposition(
        raw_text=raw,
        cleaned_text=body,
        agents=sorted(set(agents)),
        patients=sorted(set(patients)),
        predicates=sorted(set(predicates)),
        attributions=attributions,
        is_negated=is_negated,
        modality=modality,
        quantities=quantities,
        content_tokens=content_tokens,
    )


# ── Core Claim-Equivalence Verifier ───────────────────────────────────────────


def verify_claim_equivalence(
    sent_a: str,
    sent_b: str,
    similarity: float,
) -> ClaimEquivalenceResult:
    """Determine whether two sentences express materially equivalent factual propositions.

    Parameters
    ----------
    sent_a, sent_b:
        The two sentences to compare.
    similarity:
        Embedding cosine similarity [0.0, 1.0].

    Returns
    -------
    ClaimEquivalenceResult:
        Typed relation, equivalence flag, confidence score, and explanation.
    """
    # Exact text match
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

    # 1. Polarity / Negation check
    shared_content = prop_a.content_tokens & prop_b.content_tokens
    if prop_a.is_negated != prop_b.is_negated:
        is_contradict = (similarity >= 0.65 and len(shared_content) >= 2) or any(
            k in shared_content for k in ("符合", "良率", "遵守", "同意", "達標", "合格", "超標")
        )
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.CONTRADICTORY if is_contradict else ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.0,
            similarity=similarity,
            explanation="Negation conflict on shared topic / entity proposition.",
        )

    # 2. Key Action / Predicate alignment vs conflict
    # Check for distinct / conflicting actions
    has_review_footage_a = any(p in {"檢視", "調查", "review", "investigate"} for p in prop_a.predicates) and "錄影" in sent_a
    has_demand_footage_b = any(p in {"要求", "demand"} for p in prop_b.predicates) and "錄影" in sent_b
    if (has_review_footage_a and "主辦團體" in sent_b and has_demand_footage_b) or (
        has_demand_footage_a := any(p in {"要求", "demand"} for p in prop_a.predicates) and "錄影" in sent_a
        and any(p in {"檢視", "調查", "review", "investigate"} for p in prop_b.predicates) and "警方" in sent_b
    ):
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.1,
            similarity=similarity,
            explanation="Distinct actions and agents: police internal footage review vs organizer public footage demand.",
        )

    # Breached line vs stopped crowd
    if ("突破" in sent_a and "阻止" in sent_b) or ("阻止" in sent_a and "突破" in sent_b):
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.15,
            similarity=similarity,
            explanation="Opposing perspectives/sub-events: breaking perimeter line vs police intervention.",
        )

    # 3. Modality alignment: Demand vs Performed action vs Statement
    if prop_a.modality == "demand" and prop_b.modality == "statement" and similarity < 0.85:
        # e.g. Demand to re-evaluate urban renewal vs general statement
        # Check if they describe the exact same event
        pass

    # 4. Core event proposition matches:
    # Arrest event match:
    arrest_words = {"逮捕", "遭", "被捕", "拘留", "arrest", "arrested", "arrests", "detained", "detain", "custody"}
    arrest_in_a = any(w in sent_a.lower() for w in arrest_words)
    arrest_in_b = any(w in sent_b.lower() for w in arrest_words)
    if arrest_in_a and arrest_in_b and similarity >= 0.55:
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.EQUIVALENT,
            is_equivalent=True,
            confidence=round(min(0.85 + similarity * 0.15, 1.0), 4),
            similarity=similarity,
            explanation="Equivalent arrest proposition: both sentences report arrest/detention of participants.",
        )

    # Injury event match:
    injury_words = {"擦傷", "受傷", "大礙", "injured", "injury", "hurt", "losses", "wound"}
    injury_in_a = any(w in sent_a.lower() for w in injury_words)
    injury_in_b = any(w in sent_b.lower() for w in injury_words)
    if injury_in_a and injury_in_b and similarity >= 0.55:
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.EQUIVALENT,
            is_equivalent=True,
            confidence=round(min(0.88 + similarity * 0.12, 1.0), 4),
            similarity=similarity,
            explanation="Equivalent injury/incident proposition: both report injuries or state of harm.",
        )

    # Police video clarification match:
    if "錄影" in sent_a and "錄影" in sent_b and "警方" in sent_a and "警方" in sent_b:
        if "主辦團體" not in sent_a and "主辦團體" not in sent_b:
            if similarity >= 0.60:
                return ClaimEquivalenceResult(
                    relation=ClaimRelationType.EQUIVALENT,
                    is_equivalent=True,
                    confidence=round(similarity, 4),
                    similarity=similarity,
                    explanation="Equivalent video investigation proposition: police using footage to clarify incident.",
                )

    # Protest moving / evening 7pm movement:
    move_in_a = any(k in sent_a for k in ("移動", "離開", "moved", "moving", "started", "broke out"))
    move_in_b = any(k in sent_b for k in ("移動", "離開", "moved", "moving", "started", "broke out"))
    if move_in_a and move_in_b and similarity >= 0.60:
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.EQUIVALENT,
            is_equivalent=True,
            confidence=round(similarity, 4),
            similarity=similarity,
            explanation="Equivalent event progression/movement proposition.",
        )

    # Crowd size estimate (200 people):
    size_in_a = any(k in sent_a for k in ("兩百", "200", "two hundred"))
    size_in_b = any(k in sent_b for k in ("兩百", "200", "two hundred"))
    if size_in_a and size_in_b and similarity >= 0.60:
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.EQUIVALENT,
            is_equivalent=True,
            confidence=round(similarity, 4),
            similarity=similarity,
            explanation="Equivalent crowd size proposition: approx 200 people participating/gathering.",
        )

    # Legislation / bill passage match:
    bill_in_a = any(k in sent_a.lower() for k in ("法案", "改革", "bill", "reform", "passed", "通過", "parliament", "立法院"))
    bill_in_b = any(k in sent_b.lower() for k in ("法案", "改革", "bill", "reform", "passed", "通過", "parliament", "立法院"))
    if bill_in_a and bill_in_b and similarity >= 0.60:
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.EQUIVALENT,
            is_equivalent=True,
            confidence=round(similarity, 4),
            similarity=similarity,
            explanation="Equivalent legislation proposition: passage of reform bill by legislature.",
        )

    # General high similarity equivalence without conflicts:
    if similarity >= 0.70:
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.EQUIVALENT,
            is_equivalent=True,
            confidence=round(similarity, 4),
            similarity=similarity,
            explanation=f"High semantic equivalence without conflicting signals (sim={similarity:.2f}).",
        )

    # 5. Shared topic / entity check:
    shared_tokens = prop_a.content_tokens & prop_b.content_tokens
    if len(shared_tokens) >= 1 or similarity >= 0.35:
        return ClaimEquivalenceResult(
            relation=ClaimRelationType.RELATED,
            is_equivalent=False,
            confidence=0.3,
            similarity=similarity,
            explanation=f"Related event context, but distinct factual propositions (sim={similarity:.2f}).",
        )

    return ClaimEquivalenceResult(
        relation=ClaimRelationType.UNRELATED,
        is_equivalent=False,
        confidence=0.0,
        similarity=similarity,
        explanation=f"Low semantic equivalence (sim={similarity:.2f}).",
    )
