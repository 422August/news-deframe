"""Unit tests for claim-equivalence verification and proposition extraction."""

import pytest
from news_deframe.analysis.claim_verifier import (
    ClaimRelationType,
    ClaimEquivalenceResult,
    SentenceProposition,
    extract_proposition,
    verify_claim_equivalence,
)


class TestPropositionExtraction:
    def test_zh_attribution_extraction(self):
        sent = "警方表示，將檢視現場錄影畫面以釐清經過。"
        prop = extract_proposition(sent)
        assert "警方" in prop.attributions
        assert "警方" in prop.agents
        assert prop.is_negated is False

    def test_zh_negation_extraction(self):
        sent = "部分示威者拒絕服從警方的解散命令。"
        prop = extract_proposition(sent)
        assert prop.is_negated is True
        assert "示威者" in prop.agents

    def test_zh_quantity_extraction(self):
        sent = "現場約兩百名民眾要求市政府重新檢討計畫。"
        prop = extract_proposition(sent)
        assert any("兩百" in q for q in prop.quantities)

    def test_en_attribution_and_modality(self):
        sent = "Protest organizers demanded that authorities release the full video footage."
        prop = extract_proposition(sent)
        assert prop.modality == "demand"
        assert "organizers" in prop.agents


class TestClaimEquivalenceVerification:
    def test_exact_match_is_equivalent(self):
        sent = "晶圓製造商昨日宣布擴建先進封裝測試廠。"
        res = verify_claim_equivalence(sent, sent, similarity=1.0)
        assert res.relation == ClaimRelationType.EQUIVALENT
        assert res.is_equivalent is True
        assert res.confidence == 1.0

    def test_negation_conflict_is_contradictory(self):
        sent_a = "晶圓廠發言人證實三奈米製程晶圓良率已突破八成五。"
        sent_b = "晶圓廠發言人否認三奈米製程晶圓良率達到八成五。"
        res = verify_claim_equivalence(sent_a, sent_b, similarity=0.83)
        assert res.relation == ClaimRelationType.CONTRADICTORY
        assert res.is_equivalent is False

    def test_distinct_actions_are_related_not_equivalent(self):
        sent_a = "警方表示將檢視錄影以釐清經過。"
        sent_b = "主辦團體表示將要求警方公布完整執法影像。"
        res = verify_claim_equivalence(sent_a, sent_b, similarity=0.62)
        assert res.relation == ClaimRelationType.RELATED
        assert res.is_equivalent is False

    def test_cross_lingual_equivalent_claims(self):
        sent_a = "中央銀行昨日宣布將基準利率調升一碼。"
        sent_b = "Central bank raised the benchmark interest rate by 25 basis points."
        res = verify_claim_equivalence(sent_a, sent_b, similarity=0.85)
        assert res.relation == ClaimRelationType.EQUIVALENT
        assert res.is_equivalent is True

    def test_unrelated_sentences_are_unrelated(self):
        sent_a = "太空探測器成功傳回木星冰衛星的高解析度雷達影像。"
        sent_b = "中央銀行昨日宣布將基準利率調升一碼。"
        res = verify_claim_equivalence(sent_a, sent_b, similarity=0.03)
        assert res.relation == ClaimRelationType.UNRELATED
        assert res.is_equivalent is False
