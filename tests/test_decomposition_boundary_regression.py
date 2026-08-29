"""Comprehensive structural regression and adversarial test suite for atomic proposition decomposition boundaries."""
from __future__ import annotations

import pytest

from news_deframe.analysis.claim_verifier import (
    AtomicProposition,
    SentenceDecompositionTrace,
    extract_atomic_propositions,
    trace_sentence_decomposition,
    is_atomic_proposition_eligible,
)


class TestStructuralDecompositionBoundaries:
    """18 required structural decomposition test cases asserting FINAL atomic propositions."""

    def test_case_01_topic_plus_assertion(self):
        """Case 1: Topic header + factual comment must remain bound together."""
        sent = "特別費部分，通過統刪60%"
        props = extract_atomic_propositions("art1", 0, sent)
        texts = [p.proposition_text for p in props]
        assert len(props) == 1
        assert "特別費" in texts[0] and "統刪60%" in texts[0]

    def test_case_02_temporal_adjunct_plus_assertion(self):
        """Case 2: Temporal subordinate adjunct attached to main assertion."""
        sent = "經過近兩小時的激辯後，委員會通過了改革法案。"
        props = extract_atomic_propositions("art2", 0, sent)
        texts = [p.proposition_text for p in props]
        assert len(props) == 1
        assert "經過近兩小時的激辯後" in texts[0]
        assert "委員會通過了改革法案" in texts[0]

    def test_case_03_causal_adjunct_plus_assertion(self):
        """Case 3: Causal background adjunct bound to its conclusion."""
        sent = "由於原物料價格大幅上漲，廠商決定調高售價5%。"
        props = extract_atomic_propositions("art3", 0, sent)
        texts = [p.proposition_text for p in props]
        assert len(props) == 1
        assert "由於原物料價格大幅上漲" in texts[0]
        assert "廠商決定調高售價5%" in texts[0]

    def test_case_04_attribution_plus_quoted_assertion(self):
        """Case 4: Attribution frame sets speaker metadata and yields clean proposition text."""
        sent = "市長受訪表示，市府今年將擴建三座蓄水池。"
        props = extract_atomic_propositions("art4", 0, sent)
        assert len(props) == 1
        assert props[0].speaker == "市長"
        assert props[0].proposition_text == "市府今年將擴建三座蓄水池"

    def test_case_05_shared_subject_coordination(self):
        """Case 5: Coordinated predicates share overt subject via inheritance."""
        sent = "水利局展開全面清淤，並在沿岸加設五處抽水站。"
        props = extract_atomic_propositions("art5", 0, sent)
        texts = [p.proposition_text for p in props]
        assert len(props) == 2
        assert "水利局展開全面清淤" in texts[0]
        assert "水利局" in texts[1] and "加設五處抽水站" in texts[1]

    def test_case_06_shared_object_coordination(self):
        """Case 6: Subject list followed by collective predicate continuation."""
        sent = "其中第一分隊、第二分隊及特搜隊，皆全數完成搜救任務。"
        props = extract_atomic_propositions("art6", 0, sent)
        texts = [p.proposition_text for p in props]
        assert len(props) == 1
        assert "皆全數完成搜救任務" in texts[0]
        assert "特搜隊" in texts[0]

    def test_case_07_subordinate_clause_requiring_parent_context(self):
        """Case 7: Subordinate condition clause bound to modal requirement."""
        sent = "在未取得環評許可前，工廠不得逕行復工。"
        props = extract_atomic_propositions("art7", 0, sent)
        texts = [p.proposition_text for p in props]
        assert len(props) == 1
        assert "在未取得環評許可前" in texts[0]
        assert "工廠不得逕行復工" in texts[0]

    def test_case_08_complement_clause(self):
        """Case 8: Complement head attached to its complement clause."""
        sent = "通案規範還包含，所有公務車輛全面汰換為電動車。"
        props = extract_atomic_propositions("art8", 0, sent)
        texts = [p.proposition_text for p in props]
        assert len(props) == 1
        assert "電動車" in texts[0]

    def test_case_09_rhetorical_question_rejected(self):
        """Case 9: Pure rhetorical question yields 0 claim propositions."""
        sent = "難道相關單位不用負起責任嗎？"
        props = extract_atomic_propositions("art9", 0, sent)
        assert len(props) == 0

    def test_case_10_valid_short_factual_proposition(self):
        """Case 10: Short factual proposition preserved."""
        sent = "全縣停班停課一天。"
        props = extract_atomic_propositions("art10", 0, sent)
        assert len(props) == 1
        assert props[0].proposition_text == "全縣停班停課一天"

    def test_case_11_valid_quantity_proposition(self):
        """Case 11: Quantitative claim preserved with structured quantity."""
        sent = "該院共計減收病患300人。"
        props = extract_atomic_propositions("art11", 0, sent)
        assert len(props) == 1
        assert len(props[0].quantities) >= 1
        assert props[0].quantities[0].val == 300.0

    def test_case_12_quantity_target_in_parent_context(self):
        """Case 12: Proportion / percentage attached to host domain target."""
        sent = "最受矚目的歲出部分，原列總額為3兆元，審議結果減列480億元。"
        props = extract_atomic_propositions("art12", 0, sent)
        texts = [p.proposition_text for p in props]
        assert len(props) == 2
        assert "歲出" in texts[0] and "3兆元" in texts[0]
        assert "480億元" in texts[1]

    def test_case_13_independent_clauses_remain_separate(self):
        """Case 13: Distinct independent clauses separate cleanly into distinct claims."""
        sent = "工會今天上午發動罷工；資方隨後宣布關閉廠房。"
        props = extract_atomic_propositions("art13", 0, sent)
        assert len(props) == 2
        assert props[0].proposition_text == "工會今天上午發動罷工"
        assert props[1].proposition_text == "資方隨後宣布關閉廠房"

    def test_case_14_chinese_comma_heavy_sentence(self):
        """Case 14: Comma-heavy sentence separates into coherent claims without junk orphans."""
        sent = "教育部今天宣布，新學年調增導師津貼1000元，預計有十萬名教師受惠。"
        props = extract_atomic_propositions("art14", 0, sent)
        assert len(props) >= 1
        for p in props:
            assert is_atomic_proposition_eligible(p.proposition_text)
            assert len(p.proposition_text) > 5

    def test_case_15_english_compound_sentence(self):
        """Case 15: English compound sentence decomposes with connector cleanly stripped."""
        sent = "The council approved the zoning plan, and construction will begin in October."
        props = extract_atomic_propositions("art15", 0, sent)
        assert len(props) == 2
        assert props[0].proposition_text == "The council approved the zoning plan"
        assert props[1].proposition_text == "construction will begin in October"

    def test_case_16_mixed_attribution_and_coordination(self):
        """Case 16: Speaker attribution carried across coordinated reported assertions."""
        sent = "局長受訪表示，已查扣違法船隻3艘，並將加強海上巡弋頻率。"
        props = extract_atomic_propositions("art16", 0, sent)
        assert len(props) == 2
        assert props[0].speaker == "局長"
        assert props[1].speaker == "局長"
        assert "查扣違法船隻3艘" in props[0].proposition_text

    def test_case_17_quotation_punctuation_handling(self):
        """Case 17: Quotes enclosing complete assertions extract properly."""
        sent = "發言人強調，「所有檢驗數據皆符合國家標準」。"
        props = extract_atomic_propositions("art17", 0, sent)
        assert len(props) == 1
        assert "所有檢驗數據皆符合國家標準" in props[0].proposition_text

    def test_case_18_context_inheritance_provenance_trace(self):
        """Case 18: Diagnostic trace records provenance of inherited subject."""
        sent = "防檢署啟動緊急應變機制，並自明日起管制活禽運輸。"
        trace = trace_sentence_decomposition(sent, article_id="art18", sentence_idx=0)
        assert len(trace.final_propositions) == 2
        assert trace.context_inheritances
        assert any(c.get("type") == "subject" and c.get("context") == "防檢署" for c in trace.context_inheritances)


class TestAdversarialUnseenDomains:
    """Cross-domain adversarial synthetic sentences across 6 unseen domains."""

    def test_domain_public_health(self):
        """Public health domain: vaccine allocation and hospital capacity."""
        sent = "衛生局昨晚公布，新採購的20萬劑流感疫苗已分發完畢，各合約診所將於週一開放接種。"
        props = extract_atomic_propositions("health", 0, sent)
        assert len(props) == 2
        assert any("20萬劑" in p.proposition_text for p in props)

    def test_domain_technology(self):
        """Technology domain: chip manufacturing outage."""
        sent = "Due to a power fluctuation at Fab 3, chip production halted for 6 hours, resulting in 500 wafer losses."
        props = extract_atomic_propositions("tech", 0, sent)
        assert len(props) >= 1
        assert any("halted" in p.proposition_text or "production" in p.proposition_text for p in props)

    def test_domain_environmental_policy(self):
        """Environmental policy: emissions reduction and carbon tax."""
        sent = "環保署今天審定，鋼鐵業碳排放量上限降低15%，並自明年起開徵每噸300元碳費。"
        props = extract_atomic_propositions("env", 0, sent)
        assert len(props) == 2
        assert any("15%" in p.proposition_text for p in props)

    def test_domain_transportation(self):
        """Transportation domain: railway delay and passenger refund."""
        sent = "鐵路局上午通報，號誌故障導致南下列車平均延誤45分鐘，受影響旅客達2萬人。"
        props = extract_atomic_propositions("transport", 0, sent)
        assert len(props) >= 1
        assert any("45分鐘" in p.proposition_text or "2萬人" in p.proposition_text for p in props)

    def test_domain_education(self):
        """Education domain: subsidy allocation."""
        sent = "教育局通過偏鄉學校數位設備補助計畫，每校補助50萬元，共有80所國中小獲得核定。"
        props = extract_atomic_propositions("edu", 0, sent)
        assert len(props) >= 1
        assert any("50萬元" in p.proposition_text or "80所" in p.proposition_text for p in props)

    def test_domain_disaster_reporting(self):
        """Disaster reporting: landslide and road closure."""
        sent = "連日豪雨引發土石坍方，台9線145公里處雙向交通中斷，公路局派員搶修並預計明日中午搶通。"
        props = extract_atomic_propositions("disaster", 0, sent)
        assert len(props) >= 1
        for p in props:
            assert is_atomic_proposition_eligible(p.proposition_text)


class TestDeterministicGoldDecompositionBenchmark:
    """Deterministic gold benchmark measuring exact and normalized precision/recall."""

    GOLD_BENCHMARK = [
        (
            "立法院今天三讀通過總預算，歲出部分原列3兆元，審議結果減列480億元。",
            ["立法院今天三讀通過總預算", "歲出部分原列3兆元", "審議結果減列480億元"],
        ),
        (
            "市長受訪表示，市府今年將興建兩座圖書館。",
            ["市府今年將興建兩座圖書館"],
        ),
        (
            "工會宣布罷工，並將於明日在市府前發動抗爭。",
            ["工會宣布罷工", "工會並將於明日在市府前發動抗爭"],
        ),
        (
            "The board approved the merger, and the new headquarters will open in Dallas.",
            ["The board approved the merger", "the new headquarters will open in Dallas"],
        ),
        (
            "在未完成法定安全檢查前，所有遊樂設施不得開放。",
            ["在未完成法定安全檢查前，所有遊樂設施不得開放"],
        ),
    ]

    def test_gold_decomposition_benchmark(self):
        """Evaluate exact / normalized match precision and recall against gold benchmark."""
        total_gold = 0
        total_pred = 0
        exact_matches = 0
        norm_matches = 0

        for sentence, expected_props in self.GOLD_BENCHMARK:
            props = extract_atomic_propositions("gold", 0, sentence)
            pred_texts = [p.proposition_text for p in props]

            total_gold += len(expected_props)
            total_pred += len(pred_texts)

            for exp in expected_props:
                exp_clean = exp.strip("，,。；; ")
                if any(p.strip("，,。；; ") == exp_clean for p in pred_texts):
                    exact_matches += 1
                    norm_matches += 1
                elif any(exp_clean in p or p in exp_clean for p in pred_texts):
                    norm_matches += 1

        precision = exact_matches / total_pred if total_pred > 0 else 0.0
        recall = exact_matches / total_gold if total_gold > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        norm_p = norm_matches / total_pred if total_pred > 0 else 0.0
        norm_r = norm_matches / total_gold if total_gold > 0 else 0.0

        print(f"Gold Decomposition Benchmark: Exact P={precision:.2f}, R={recall:.2f}, F1={f1:.2f} | Norm P={norm_p:.2f}, R={norm_r:.2f}")
        assert precision >= 0.80
        assert recall >= 0.80
