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
