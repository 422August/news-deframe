"""Multi-domain synthetic gold evaluation datasets for deterministic NLP evaluation.

Covers 6 distinct domains with unseen vocabulary in both Chinese and English:
1. Semiconductor Technology & Supply Chain
2. Public Health & Vaccine Logistics
3. Environmental Regulation & Carbon Policy
4. Aerospace & Satellite Operations
5. Banking Solvency & Financial Oversight
6. Renewable Energy Grid Infrastructure
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SVOExtractionGoldItem:
    """Gold annotation for SVO extraction."""

    sentence: str
    lang: str
    expected_subjects: list[str]
    expected_predicates: list[str]
    expected_objects: list[str]
    is_passive: bool
    domain: str


@dataclass(frozen=True)
class PredicateNormalizationGoldItem:
    """Gold annotation for predicate normalization."""

    raw_token: str
    sentence_context: str
    lang: str
    expected_normalized: str
    is_valid_predicate: bool
    domain: str


@dataclass(frozen=True)
class ActorValidationGoldItem:
    """Gold annotation for actor vs non-actor candidate classification."""

    surface: str
    ner_type: str
    sentence_context: str
    expected_is_actor: bool
    semantic_category: str  # "ACTOR", "LOCATION", "EVENT", "ACTION", "ABSTRACT", "INJURY", "FRAGMENT"
    domain: str


@dataclass(frozen=True)
class ClaimRelationGoldItem:
    """Gold annotation for claim pair relation classification."""

    sent_a: str
    sent_b: str
    expected_relation: str  # "EQUIVALENT", "COMPATIBLE", "RELATED", "CONTRADICTORY", "UNRELATED"
    domain: str


@dataclass(frozen=True)
class MultiArticleClusteringGoldItem:
    """Gold annotation for multi-article event claim clustering."""

    domain: str
    articles: list[dict[str, Any]]  # [{"article_id": str, "sentences": list[str]}]
    expected_clusters: list[list[str]]  # List of clusters (lists of sentence strings)
    expected_actors: list[str]


# ── Gold SVO Items ────────────────────────────────────────────────────────────

GOLD_SVO_ITEMS: list[SVOExtractionGoldItem] = [
    # Domain 1: Semiconductor
    SVOExtractionGoldItem(
        sentence="晶圓製造商昨日宣布擴建先進封裝測試廠。",
        lang="zh",
        expected_subjects=["晶圓製造商"],
        expected_predicates=["宣布", "擴建"],
        expected_objects=["先進封裝測試廠"],
        is_passive=False,
        domain="semiconductor",
    ),
    SVOExtractionGoldItem(
        sentence="新型微處理器晶片遭到出口管制機構全面查扣。",
        lang="zh",
        expected_subjects=["新型微處理器晶片"],
        expected_predicates=["查扣"],
        expected_objects=["出口管制機構"],
        is_passive=True,
        domain="semiconductor",
    ),
    SVOExtractionGoldItem(
        sentence="The semiconductor foundry completed its yield improvement program.",
        lang="en",
        expected_subjects=["semiconductor foundry"],
        expected_predicates=["complete", "completed"],
        expected_objects=["yield improvement program"],
        is_passive=False,
        domain="semiconductor",
    ),
    # Domain 2: Public Health
    SVOExtractionGoldItem(
        sentence="衛生監管局迅速核准新型抗體疫苗的臨床試驗。",
        lang="zh",
        expected_subjects=["衛生監管局"],
        expected_predicates=["核准"],
        expected_objects=["臨床試驗"],
        is_passive=False,
        domain="public_health",
    ),
    SVOExtractionGoldItem(
        sentence="冷鏈配送車隊遭暴風雪嚴重延誤。",
        lang="zh",
        expected_subjects=["冷鏈配送車隊"],
        expected_predicates=["延誤"],
        expected_objects=["暴風雪"],
        is_passive=True,
        domain="public_health",
    ),
    SVOExtractionGoldItem(
        sentence="Epidemiologists monitored the mutation rate of the pathogen.",
        lang="en",
        expected_subjects=["Epidemiologists"],
        expected_predicates=["monitor", "monitored"],
        expected_objects=["mutation rate"],
        is_passive=False,
        domain="public_health",
    ),
    # Domain 3: Environmental Regulation
    SVOExtractionGoldItem(
        sentence="環保署對違規排污石化廠開立巨額罰單。",
        lang="zh",
        expected_subjects=["環保署"],
        expected_predicates=["開立"],
        expected_objects=["巨額罰單"],
        is_passive=False,
        domain="environmental_reg",
    ),
    SVOExtractionGoldItem(
        sentence="Forest rangers planted twenty thousand native saplings.",
        lang="en",
        expected_subjects=["Forest rangers"],
        expected_predicates=["plant", "planted"],
        expected_objects=["twenty thousand native saplings"],
        is_passive=False,
        domain="environmental_reg",
    ),
    # Domain 4: Space Exploration
    SVOExtractionGoldItem(
        sentence="航太工程團隊成功將通訊衛星送入預定軌道。",
        lang="zh",
        expected_subjects=["航太工程團隊"],
        expected_predicates=["送入"],
        expected_objects=["預定軌道"],
        is_passive=False,
        domain="space_exploration",
    ),
    SVOExtractionGoldItem(
        sentence="Ground controllers restored telemetry communication with the lunar probe.",
        lang="en",
        expected_subjects=["Ground controllers"],
        expected_predicates=["restore", "restored"],
        expected_objects=["telemetry communication"],
        is_passive=False,
        domain="space_exploration",
    ),
    # Domain 5: Financial Regulation
    SVOExtractionGoldItem(
        sentence="中央銀行接管陷入流動性危機的區域商銀。",
        lang="zh",
        expected_subjects=["中央銀行"],
        expected_predicates=["接管"],
        expected_objects=["區域商銀"],
        is_passive=False,
        domain="banking_crisis",
    ),
    SVOExtractionGoldItem(
        sentence="Financial regulators suspended offshore derivatives trading.",
        lang="en",
        expected_subjects=["Financial regulators"],
        expected_predicates=["suspend", "suspended"],
        expected_objects=["offshore derivatives trading"],
        is_passive=False,
        domain="banking_crisis",
    ),
    # Domain 6: Energy Grid
    SVOExtractionGoldItem(
        sentence="電力調度中心在兩小時內恢復主要變電站供電。",
        lang="zh",
        expected_subjects=["電力調度中心"],
        expected_predicates=["恢復"],
        expected_objects=["供電"],
        is_passive=False,
        domain="energy_grid",
    ),
    SVOExtractionGoldItem(
        sentence="Offshore wind technicians repaired the submarine transmission cable.",
        lang="en",
        expected_subjects=["Offshore wind technicians"],
        expected_predicates=["repair", "repaired"],
        expected_objects=["submarine transmission cable"],
        is_passive=False,
        domain="energy_grid",
    ),
]


# ── Gold Predicate Items ──────────────────────────────────────────────────────

GOLD_PREDICATE_ITEMS: list[PredicateNormalizationGoldItem] = [
    # Broken Chinese tokens repaired
    PredicateNormalizationGoldItem(
        raw_token="正調",
        sentence_context="專家正調查超導材料的臨界溫度。",
        lang="zh",
        expected_normalized="調查",
        is_valid_predicate=True,
        domain="semiconductor",
    ),
    PredicateNormalizationGoldItem(
        raw_token="活",
        sentence_context="志工在社區活動中心協助老人接種。",
        lang="zh",
        expected_normalized="",
        is_valid_predicate=False,
        domain="public_health",
    ),
    PredicateNormalizationGoldItem(
        raw_token="辦團",
        sentence_context="主辦團體呼籲各界遵守秩序。",
        lang="zh",
        expected_normalized="",
        is_valid_predicate=False,
        domain="environmental_reg",
    ),
    PredicateNormalizationGoldItem(
        raw_token="遭",
        sentence_context="運載火箭遭強風吹離發射台。",
        lang="zh",
        expected_normalized="吹離",
        is_valid_predicate=True,
        domain="space_exploration",
    ),
    PredicateNormalizationGoldItem(
        raw_token="平穩",
        sentence_context="貨幣市場維持平穩運作。",
        lang="zh",
        expected_normalized="",
        is_valid_predicate=False,
        domain="banking_crisis",
    ),
    PredicateNormalizationGoldItem(
        raw_token="逐步",
        sentence_context="輸電網路逐步擴大供電範圍。",
        lang="zh",
        expected_normalized="",
        is_valid_predicate=False,
        domain="energy_grid",
    ),
    # Valid standard predicates
    PredicateNormalizationGoldItem(
        raw_token="核准",
        sentence_context="主管機關核准併購申請案。",
        lang="zh",
        expected_normalized="核准",
        is_valid_predicate=True,
        domain="banking_crisis",
    ),
    PredicateNormalizationGoldItem(
        raw_token="deployed",
        sentence_context="The military deployed defensive satellites into geostationary orbit.",
        lang="en",
        expected_normalized="deploy",
        is_valid_predicate=True,
        domain="space_exploration",
    ),
    PredicateNormalizationGoldItem(
        raw_token="allocated",
        sentence_context="The central committee allocated emergency funds to healthcare providers.",
        lang="en",
        expected_normalized="allocate",
        is_valid_predicate=True,
        domain="public_health",
    ),
]


# ── Gold Actor Validation Items ───────────────────────────────────────────────

GOLD_ACTOR_ITEMS: list[ActorValidationGoldItem] = [
    # Genuine Actors
    ActorValidationGoldItem("晶圓製造商", "ORG", "晶圓製造商宣布擴建廠房。", True, "ACTOR", "semiconductor"),
    ActorValidationGoldItem("衛生監管局", "ORG", "衛生監管局核准新藥上市。", True, "ACTOR", "public_health"),
    ActorValidationGoldItem("冷鏈配送員", "PERSON", "冷鏈配送員運送疫苗。", True, "ACTOR", "public_health"),
    ActorValidationGoldItem("環保署", "ORG", "環保署開出罰單。", True, "ACTOR", "environmental_reg"),
    ActorValidationGoldItem("航太工程師", "PERSON", "航太工程師監測軌道參數。", True, "ACTOR", "space_exploration"),
    ActorValidationGoldItem("中央銀行", "ORG", "中央銀行宣布調升利率。", True, "ACTOR", "banking_crisis"),
    ActorValidationGoldItem("電力調度員", "PERSON", "電力調度員重啟發電機組。", True, "ACTOR", "energy_grid"),
    ActorValidationGoldItem("Forest rangers", "PERSON", "Forest rangers patrolled the reserve.", True, "ACTOR", "environmental_reg"),
    ActorValidationGoldItem("Financial regulators", "ORG", "Financial regulators published the audit.", True, "ACTOR", "banking_crisis"),
    ActorValidationGoldItem("Ground controllers", "PERSON", "Ground controllers acquired the signal.", True, "ACTOR", "space_exploration"),

    # Non-Actors: Locations & Facilities
    ActorValidationGoldItem("先進封裝測試廠", "FAC", "晶圓製造商昨日宣布擴建先進封裝測試廠。", False, "LOCATION", "semiconductor"),
    ActorValidationGoldItem("科學園區外圍", "LOC", "示威隊伍抵達科學園區外圍。", False, "LOCATION", "semiconductor"),
    ActorValidationGoldItem("臨床試驗中心", "FAC", "新藥在臨床試驗中心進行測試。", False, "LOCATION", "public_health"),
    ActorValidationGoldItem("國家自然保護區", "LOC", "保護區內禁止非法伐木。", False, "LOCATION", "environmental_reg"),
    ActorValidationGoldItem("太空發射基地", "FAC", "火箭在太空發射基地整備。", False, "LOCATION", "space_exploration"),
    ActorValidationGoldItem("主要變電所", "FAC", "變電所發生短路跳脫。", False, "LOCATION", "energy_grid"),

    # Non-Actors: Events & Actions
    ActorValidationGoldItem("併購審查會議", "EVENT", "雙方出席併購審查會議。", False, "EVENT", "banking_crisis"),
    ActorValidationGoldItem("排污查核行動", "EVENT", "環保署展開排污查核行動。", False, "EVENT", "environmental_reg"),
    ActorValidationGoldItem("突發停電事件", "EVENT", "停電事件影響三萬戶居民。", False, "EVENT", "energy_grid"),
    ActorValidationGoldItem("衛星入軌程序", "EVENT", "團隊順利完成衛星入軌程序。", False, "EVENT", "space_exploration"),
    ActorValidationGoldItem("疫苗臨床試驗", "EVENT", "試驗成果符合預期標準。", False, "EVENT", "public_health"),

    # Non-Actors: Abstract concepts & states & injuries
    ActorValidationGoldItem("晶片良率評估報告", "WORK_OF_ART", "主管檢視晶片良率評估報告。", False, "ABSTRACT", "semiconductor"),
    ActorValidationGoldItem("碳權交易配額", "LAW", "廠商依法申報碳權交易配額。", False, "ABSTRACT", "environmental_reg"),
    ActorValidationGoldItem("流動性風險指標", "ABSTRACT", "銀行監控各項流動性風險指標。", False, "ABSTRACT", "banking_crisis"),
    ActorValidationGoldItem("輕微電弧灼傷", "INJURY", "工程師受到輕微電弧灼傷。", False, "INJURY", "energy_grid"),
    ActorValidationGoldItem("輕度低溫凍傷", "INJURY", "配送員出現輕度低溫凍傷。", False, "INJURY", "public_health"),

    # Non-Actors: Broken syntactic fragments
    ActorValidationGoldItem("活", "VERB_ACTION", "活動期間志工協助現場引導。", False, "FRAGMENT", "public_health"),
    ActorValidationGoldItem("正調", "VERB_ACTION", "專家正調查異常振幅原因。", False, "FRAGMENT", "space_exploration"),
    ActorValidationGoldItem("體均", "FRAGMENT", "機體均符合標準規範。", False, "FRAGMENT", "semiconductor"),
]


# ── Gold Claim Relation Classification Items ──────────────────────────────────

GOLD_CLAIM_RELATION_ITEMS: list[ClaimRelationGoldItem] = [
    # Equivalent pairs (bilingual and monolingual paraphrases)
    ClaimRelationGoldItem(
        sent_a="晶圓製造商昨日宣布擴建先進封裝測試廠。",
        sent_b="該晶圓大廠昨日公布擴充先進封裝產能的建設計畫。",
        expected_relation="EQUIVALENT",
        domain="semiconductor",
    ),
    ClaimRelationGoldItem(
        sent_a="衛生監管局迅速核准新型抗體疫苗的臨床試驗。",
        sent_b="官方監管機構已核准該款抗體疫苗展開人體臨床試驗。",
        expected_relation="EQUIVALENT",
        domain="public_health",
    ),
    ClaimRelationGoldItem(
        sent_a="中央銀行昨日宣布將基準利率調升一碼。",
        sent_b="Central bank raised the benchmark interest rate by 25 basis points.",
        expected_relation="EQUIVALENT",
        domain="banking_crisis",
    ),
    ClaimRelationGoldItem(
        sent_a="電力工程人員在兩小時內全面恢復主要變電站供電。",
        sent_b="Grid technicians fully restored power to the main substation within two hours.",
        expected_relation="EQUIVALENT",
        domain="energy_grid",
    ),

    # Related but distinct (topic match, different factual claims)
    ClaimRelationGoldItem(
        sent_a="環保署對違規排污石化廠開立兩千萬元巨額罰單。",
        sent_b="環保團體要求環保署撤銷該石化廠的排污許可證。",
        expected_relation="RELATED",
        domain="environmental_reg",
    ),
    ClaimRelationGoldItem(
        sent_a="航太團隊確認運載火箭第一節助推器成功分離。",
        sent_b="地面控制中心表示衛星太陽能板已完全展開運作。",
        expected_relation="RELATED",
        domain="space_exploration",
    ),
    ClaimRelationGoldItem(
        sent_a="區域商業銀行否認面臨流動性枯竭危機。",
        sent_b="中央銀行指派監理小組進駐該區域商業銀行進行查核。",
        expected_relation="RELATED",
        domain="banking_crisis",
    ),
    ClaimRelationGoldItem(
        sent_a="冷鏈車隊運送十萬劑疫苗抵達偏遠山區診所。",
        sent_b="衛生局呼籲六十五歲以上長者儘速預約接種疫苗。",
        expected_relation="RELATED",
        domain="public_health",
    ),

    # Contradictory pairs
    ClaimRelationGoldItem(
        sent_a="石化廠管理層聲明排放水質完全符合國家環保標準。",
        sent_b="石化廠排放水質重金屬檢測超標二十倍未符合國家標準。",
        expected_relation="CONTRADICTORY",
        domain="environmental_reg",
    ),
    ClaimRelationGoldItem(
        sent_a="晶圓廠發言人證實三奈米製程晶圓良率已突破八成五。",
        sent_b="晶圓廠發言人否認三奈米製程晶圓良率達到八成五。",
        expected_relation="CONTRADICTORY",
        domain="semiconductor",
    ),

    # Unrelated pairs
    ClaimRelationGoldItem(
        sent_a="太空探測器成功傳回木星冰衛星的高解析度雷達影像。",
        sent_b="中央銀行昨日宣布將基準利率調升一碼。",
        expected_relation="UNRELATED",
        domain="cross_domain",
    ),
    ClaimRelationGoldItem(
        sent_a="冷鏈配送車隊遭暴風雪嚴重延誤。",
        sent_b="離岸風電水下電纜修復工程順利完工。",
        expected_relation="UNRELATED",
        domain="cross_domain",
    ),

    # ── Category A: Same topic, different claims ──────────────────────────────
    ClaimRelationGoldItem(
        sent_a="The financial regulator issued a record fine to the bank for compliance violations.",
        sent_b="The financial regulator revoked the bank's operating license following the audit.",
        expected_relation="RELATED",
        domain="banking_crisis",
    ),
    ClaimRelationGoldItem(
        sent_a="電網修復工程已完工，供電恢復正常。",
        sent_b="電網供電中斷原因仍在調查中，預計需數日釐清。",
        expected_relation="RELATED",
        domain="energy_grid",
    ),
    ClaimRelationGoldItem(
        sent_a="Clinical trials for the new vaccine began last Monday.",
        sent_b="Phase three vaccine trials are expected to conclude by year-end.",
        expected_relation="RELATED",
        domain="public_health",
    ),
    ClaimRelationGoldItem(
        sent_a="通訊衛星已成功進入預定軌道並開始運作。",
        sent_b="通訊衛星發射延誤三週，主因為氣象條件不佳。",
        expected_relation="RELATED",
        domain="space_exploration",
    ),

    # ── Category B: Same event, different speakers, different assertions ───────
    ClaimRelationGoldItem(
        sent_a="環保團體批評主管機關審查時程過長，導致污染持續惡化。",
        sent_b="主管機關表示審查作業按既定程序進行，預計六個月內完成。",
        expected_relation="RELATED",
        domain="environmental_reg",
    ),
    ClaimRelationGoldItem(
        sent_a="The manufacturer stated the new battery met all safety certification benchmarks.",
        sent_b="The regulatory agency requested additional performance data before approving the battery.",
        expected_relation="RELATED",
        domain="technology",
    ),
    ClaimRelationGoldItem(
        sent_a="在野黨質疑預算審查缺乏透明度，要求公開完整細目。",
        sent_b="執政黨強調預算案已依法完成三讀程序，符合程序規定。",
        expected_relation="RELATED",
        domain="governance",
    ),

    # ── Category C: Same proposition, paraphrased ─────────────────────────────
    ClaimRelationGoldItem(
        sent_a="先進半導體製造商昨日發生供電中斷事故。",
        sent_b="晶圓大廠昨日遭遇無預警停電事故。",
        expected_relation="EQUIVALENT",
        domain="semiconductor",
    ),
    ClaimRelationGoldItem(
        sent_a="Offshore wind technicians repaired the submarine transmission cable.",
        sent_b="Marine engineers completed repairs on the undersea power cable.",
        expected_relation="EQUIVALENT",
        domain="energy_grid",
    ),
    ClaimRelationGoldItem(
        sent_a="The health authority approved emergency use of the antiviral treatment.",
        sent_b="Emergency authorization for the antiviral drug was granted by health regulators.",
        expected_relation="EQUIVALENT",
        domain="public_health",
    ),

    # ── Category D: Same proposition, compatible additional detail ────────────
    ClaimRelationGoldItem(
        sent_a="生技研發團隊成功開發免冷鏈保存的新型疫苗佐劑。",
        sent_b="生技研發團隊在國家實驗室成功開發出免冷鏈保存、常溫穩定的新型疫苗佐劑配方。",
        expected_relation="COMPATIBLE",
        domain="public_health",
    ),
    ClaimRelationGoldItem(
        sent_a="The central bank raised interest rates by 25 basis points.",
        sent_b="The central bank's monetary policy committee voted unanimously to raise interest rates by 25 basis points.",
        expected_relation="COMPATIBLE",
        domain="banking_crisis",
    ),

    # ── Category G: Negation ──────────────────────────────────────────────────
    ClaimRelationGoldItem(
        sent_a="環保署確認石化廠排放數據符合標準。",
        sent_b="環保署否認石化廠排放數據符合標準。",
        expected_relation="CONTRADICTORY",
        domain="environmental_reg",
    ),
    ClaimRelationGoldItem(
        sent_a="The clinical trial confirmed the vaccine prevented severe disease.",
        sent_b="The clinical trial did not confirm that the vaccine prevented severe disease.",
        expected_relation="CONTRADICTORY",
        domain="public_health",
    ),

    # ── Category H: Modality ──────────────────────────────────────────────────
    ClaimRelationGoldItem(
        sent_a="監管機構已完成對銀行的現場審查。",
        sent_b="監管機構計畫下個月對銀行展開現場審查。",
        expected_relation="RELATED",
        domain="banking_crisis",
    ),
    ClaimRelationGoldItem(
        sent_a="The agency completed the environmental impact assessment.",
        sent_b="Environmental groups demanded that the agency complete the impact assessment.",
        expected_relation="RELATED",
        domain="environmental_reg",
    ),
    ClaimRelationGoldItem(
        sent_a="廠方已完成廢水處理設施升級工程。",
        sent_b="環保團體要求廠方儘速完成廢水處理設施升級。",
        expected_relation="RELATED",
        domain="environmental_reg",
    ),

    # ── Category I: Quantity disagreement ─────────────────────────────────────
    ClaimRelationGoldItem(
        sent_a="工程團隊預估本次晶圓報廢損失約兩千片。",
        sent_b="市調機構指出受損晶圓數量估計達一萬片。",
        expected_relation="CONTRADICTORY",
        domain="semiconductor",
    ),
    ClaimRelationGoldItem(
        sent_a="The new wafer process achieved a yield rate of 85 percent.",
        sent_b="The new wafer process achieved a yield rate of 42 percent.",
        expected_relation="CONTRADICTORY",
        domain="semiconductor",
    ),

    # ── Category J: Different agent (same action, different actor) ────────────
    ClaimRelationGoldItem(
        sent_a="中央銀行宣布調升基準利率。",
        sent_b="財政部宣布調升基準利率。",
        expected_relation="RELATED",
        domain="banking_crisis",
    ),
    ClaimRelationGoldItem(
        sent_a="The environmental agency fined the petrochemical plant for violations.",
        sent_b="The municipal government fined the petrochemical plant for violations.",
        expected_relation="RELATED",
        domain="environmental_reg",
    ),

    # ── Category L: Chinese-language pairs ────────────────────────────────────
    ClaimRelationGoldItem(
        sent_a="廠區緊急備用發電機在五分鐘內全面啟動。",
        sent_b="備用發電機迅速在五分鐘內啟動維持關鍵機台運轉。",
        expected_relation="EQUIVALENT",
        domain="energy_grid",
    ),
    ClaimRelationGoldItem(
        sent_a="衛生署表示將優先審核該項專利技術。",
        sent_b="國際非政府組織呼籲儘速將此技術授權開發中國家。",
        expected_relation="RELATED",
        domain="public_health",
    ),

    # ── Category M: Non-political domains ────────────────────────────────────
    ClaimRelationGoldItem(
        sent_a="A freight train derailed near the river bridge, blocking the main line.",
        sent_b="The freight train left the tracks at the bridge crossing, halting main line services.",
        expected_relation="EQUIVALENT",
        domain="transportation",
    ),
    ClaimRelationGoldItem(
        sent_a="The earthquake measuring 6.4 struck the coastal region at dawn.",
        sent_b="Rescue teams deployed to the coastal region following the seismic event.",
        expected_relation="RELATED",
        domain="disaster",
    ),
    ClaimRelationGoldItem(
        sent_a="研究團隊完成新型固態電池充放電循環測試，循環壽命達三千次。",
        sent_b="固態電池測試結果顯示循環壽命達到三千次充放電週期。",
        expected_relation="EQUIVALENT",
        domain="technology",
    ),
    ClaimRelationGoldItem(
        sent_a="The research team demonstrated that the new material conducts electricity at room temperature without resistance.",
        sent_b="Scientists confirmed room-temperature superconductivity in the newly synthesized compound.",
        expected_relation="EQUIVALENT",
        domain="science",
    ),
]


# ── Gold False-Merge Pair Items ───────────────────────────────────────────────
# Pairs that must NEVER be classified as EQUIVALENT or COMPATIBLE.
# These directly test the false-merge safeguard.


@dataclass(frozen=True)
class FalseMergeGoldItem:
    """Gold annotation for false-merge safeguard testing.

    Both sentences share a topic or entity, but must not be merged as the
    same claim.  expected_equivalent must always be False.
    """

    sent_a: str
    sent_b: str
    reason: str   # Human-readable description of why these must not merge
    category: str  # A, B, G, H, I, J, etc.
    domain: str


GOLD_FALSE_MERGE_PAIRS: list[FalseMergeGoldItem] = [
    # Category A: Same topic, different claims
    FalseMergeGoldItem(
        sent_a="The financial regulator imposed a fine on the bank for compliance failures.",
        sent_b="The financial regulator cleared the bank of all compliance violations.",
        reason="Same actor + topic, but opposite factual outcomes (penalty vs clearance).",
        category="A",
        domain="banking_crisis",
    ),
    FalseMergeGoldItem(
        sent_a="監管機關已對廠商完成行政調查並結案。",
        sent_b="監管機關宣布對廠商展開全面行政調查。",
        reason="Investigation completed vs initiated — opposite temporal facts.",
        category="A",
        domain="environmental_reg",
    ),
    # Category B: Different speakers, different assertions
    FalseMergeGoldItem(
        sent_a="Researchers argued that the pipeline delay was caused by funding shortfalls.",
        sent_b="Government officials stated that the pipeline delay resulted from adverse weather conditions.",
        reason="Same delay event, two different causal claims from different speakers.",
        category="B",
        domain="technology",
    ),
    FalseMergeGoldItem(
        sent_a="反對黨質疑行政部門的預算審查程序缺乏透明度。",
        sent_b="行政部門強調預算審查均依法定程序完成，資料均已公開。",
        reason="Same budget review context; one criticises transparency, one defends it.",
        category="B",
        domain="governance",
    ),
    # Category G: Negation conflict
    FalseMergeGoldItem(
        sent_a="The vaccine trial confirmed efficacy against the new variant.",
        sent_b="The vaccine trial did not confirm efficacy against the new variant.",
        reason="Positive vs negated claim on the same proposition.",
        category="G",
        domain="public_health",
    ),
    FalseMergeGoldItem(
        sent_a="環保署確認該廠廢水排放達到法定標準。",
        sent_b="環保署否認該廠廢水排放達到法定標準。",
        reason="Affirmative vs negated form of identical proposition.",
        category="G",
        domain="environmental_reg",
    ),
    # Category H: Modality — completed fact vs future plan
    FalseMergeGoldItem(
        sent_a="The grid operator restored power to all affected substations.",
        sent_b="The grid operator plans to restore power to the affected substations by next week.",
        reason="Completed fact vs future plan — different temporal status of same event.",
        category="H",
        domain="energy_grid",
    ),
    FalseMergeGoldItem(
        sent_a="研究機構完成新型疫苗的臨床三期試驗。",
        sent_b="研究機構計畫在明年啟動新型疫苗的臨床三期試驗。",
        reason="Completed trial vs future plan to begin trial.",
        category="H",
        domain="public_health",
    ),
    # Category I: Quantity conflict
    FalseMergeGoldItem(
        sent_a="The flood damaged approximately 200 homes in the affected zone.",
        sent_b="The flood damaged approximately 2,000 homes in the affected zone.",
        reason="Same event, same proposition skeleton, but 10× quantity conflict.",
        category="I",
        domain="disaster",
    ),
    FalseMergeGoldItem(
        sent_a="衛星在距地面三百公里軌道順利運作。",
        sent_b="衛星在距地面三萬六千公里軌道順利運作。",
        reason="Same satellite, same action, orbit altitude differs 120×.",
        category="I",
        domain="space_exploration",
    ),
    # Category J: Different agent (same action type)
    FalseMergeGoldItem(
        sent_a="The health ministry approved the new treatment protocol.",
        sent_b="The hospital association approved the new treatment protocol.",
        reason="Same action (approval of protocol), materially different agents.",
        category="J",
        domain="public_health",
    ),
    FalseMergeGoldItem(
        sent_a="中央氣象局發布海上颱風警報。",
        sent_b="地方政府發布海上颱風警報。",
        reason="Same warning action, different issuing agents (national vs local).",
        category="J",
        domain="disaster",
    ),
    # Cross-category: High embedding similarity but different propositions
    FalseMergeGoldItem(
        sent_a="The semiconductor firm expanded its chip fabrication capacity.",
        sent_b="The semiconductor firm reduced its chip fabrication workforce.",
        reason="Same actor + domain; opposite business decisions — expansion vs reduction.",
        category="A",
        domain="semiconductor",
    ),
    FalseMergeGoldItem(
        sent_a="航太公司宣布衛星發射任務成功完成。",
        sent_b="航太公司宣布衛星發射任務因技術問題暫停。",
        reason="Same actor + event type; opposite outcomes — success vs suspension.",
        category="A",
        domain="space_exploration",
    ),
]


# ── Gold Multi-Article Event Corpora for Clustering Evaluation ─────────────────

GOLD_CLUSTERING_CORPORA: list[MultiArticleClusteringGoldItem] = [
    MultiArticleClusteringGoldItem(
        domain="semiconductor_fab_incident",
        articles=[
            {
                "article_id": "tech_daily",
                "sentences": [
                    "先進半導體製造商昨日發生供電中斷事故。",
                    "廠區緊急備用發電機在五分鐘內全面啟動。",
                    "工程團隊預估本次晶圓報廢損失約兩千片。",
                ],
            },
            {
                "article_id": "market_wire",
                "sentences": [
                    "半導體晶圓廠因電力跳脫導致產線短暫停擺。",
                    "廠區備用發電系統在五分鐘內順利接管供電。",
                    "市調機構指出受損晶圓數量估計達兩千片。",
                ],
            },
            {
                "article_id": "global_chip_news",
                "sentences": [
                    "晶圓大廠昨日遭遇無預警停電事故。",
                    "備用發電機迅速在五分鐘內啟動維持關鍵機台運轉。",
                    "產業分析師認為此事件不會對第三季出貨造成長期衝擊。",
                ],
            },
        ],
        expected_clusters=[
            # Cluster 1: Power interruption event (3 outlets)
            [
                "先進半導體製造商昨日發生供電中斷事故。",
                "半導體晶圓廠因電力跳脫導致產線短暫停擺。",
                "晶圓大廠昨日遭遇無預警停電事故。",
            ],
            # Cluster 2: Backup generators started within 5 mins (3 outlets)
            [
                "廠區緊急備用發電機在五分鐘內全面啟動。",
                "廠區備用發電系統在五分鐘內順利接管供電。",
                "備用發電機迅速在五分鐘內啟動維持關鍵機台運轉。",
            ],
            # Cluster 3: Scrap wafer loss 2000 units (2 outlets)
            [
                "工程團隊預估本次晶圓報廢損失約兩千片。",
                "市調機構指出受損晶圓數量估計達兩千片。",
            ],
            # Cluster 4: Long term shipment impact analysis (1 outlet outlier)
            [
                "產業分析師認為此事件不會對第三季出貨造成長期衝擊。",
            ],
        ],
        expected_actors=["半導體製造商", "工程團隊", "市調機構", "產業分析師"],
    ),
    MultiArticleClusteringGoldItem(
        domain="vaccine_coldchain_breakthrough",
        articles=[
            {
                "article_id": "health_times",
                "sentences": [
                    "生技研發團隊成功開發免冷鏈保存的新型疫苗佐劑。",
                    "臨床實驗顯示常溫保存三個月後效價依然維持九成以上。",
                    "衛生署表示將優先審核該項專利技術。",
                ],
            },
            {
                "article_id": "medical_journal_digest",
                "sentences": [
                    "研發團隊宣布研發出可在室溫穩定存放的新型疫苗配方。",
                    "測試數據證實常溫放置三個月抗體保護力維持九成以上。",
                    "國際非政府組織呼籲儘速將此技術授權開發中國家。",
                ],
            },
        ],
        expected_clusters=[
            # Cluster 1: Room temperature vaccine formulation developed (2 outlets)
            [
                "生技研發團隊成功開發免冷鏈保存的新型疫苗佐劑。",
                "研發團隊宣布研發出可在室溫穩定存放的新型疫苗配方。",
            ],
            # Cluster 2: Maintained >90% efficacy after 3 months at room temp (2 outlets)
            [
                "臨床實驗顯示常溫保存三個月後效價依然維持九成以上。",
                "測試數據證實常溫放置三個月抗體保護力維持九成以上。",
            ],
            # Cluster 3: Regulatory fast track (1 outlet)
            [
                "衛生署表示將優先審核該項專利技術。",
            ],
            # Cluster 4: NGO licensing demand (1 outlet)
            [
                "國際非政府組織呼籲儘速將此技術授權開發中國家。",
            ],
        ],
        expected_actors=["生技研發團隊", "衛生署", "國際非政府組織"],
    ),
]
