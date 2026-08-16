"""Tests for event analysis across Chinese, English, and mixed corpora.

Ensures the multi-article workflow handles:
- Traditional / Simplified Chinese article corpora
- English article corpora
- Mixed Chinese and English corpora in the same event
- Bilingual entity and SVO extraction
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from news_deframe.parser.spacy_loader import detect_language
from news_deframe.parser.article_loader import discover_articles
from news_deframe.analysis.event import run_event_analysis
from news_deframe.schemas import (
    EntityModifier,
    ParsedArticle,
    SVORecord,
)


def _make_keyword_embedder(cluster_keywords: list[list[str]]):
    dim = max(len(cluster_keywords) + 2, 8)

    def _embed(sentences: list[str]) -> np.ndarray:
        embs = np.zeros((len(sentences), dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            assigned = False
            for group_idx, keywords in enumerate(cluster_keywords):
                if any(kw.lower() in s.lower() for kw in keywords):
                    embs[i, group_idx] = 1.0
                    assigned = True
                    break
            if not assigned:
                embs[i, -1] = 1.0
        return embs

    return _embed


class TestChineseCorpora:
    def test_chinese_event_analysis(self):
        """Analyze a Chinese event corpus with multiple outlets."""
        # 3 Chinese articles
        art_a = ParsedArticle(
            article_id="cna_news",
            sentences=["警方逮捕了三名抗議者。", "現場發生衝突。"],
            svo_records=[
                SVORecord(
                    sentence="警方逮捕了三名抗議者。",
                    verb="逮捕",
                    subjects=["警方"],
                    objects=["三名抗議者"],
                    is_passive=False,
                ),
                SVORecord(
                    sentence="現場發生衝突。",
                    verb="發生",
                    subjects=["衝突"],
                    objects=[],
                    is_passive=False,
                ),
            ],
            entity_modifiers=[
                EntityModifier(entity_name="警方", entity_type="ORG", modifiers=["執法人員"]),
                EntityModifier(entity_name="抗議者", entity_type="PERSON", modifiers=["示威群眾"]),
            ],
        )

        art_b = ParsedArticle(
            article_id="pts_news",
            sentences=["三名抗議者遭警方逮捕。", "活動和平進行。"],
            svo_records=[
                SVORecord(
                    sentence="三名抗議者遭警方逮捕。",
                    verb="逮捕",
                    subjects=["三名抗議者"],
                    objects=["警方"],
                    is_passive=True,
                    voice_markers=["遭"],
                )
            ],
            entity_modifiers=[
                EntityModifier(entity_name="抗議者", entity_type="PERSON", modifiers=["和平"]),
            ],
        )

        embed_fn = _make_keyword_embedder([
            ["逮捕", "警方", "抗議者"],
            ["衝突", "發生"],
            ["和平", "活動"],
        ])

        analysis = run_event_analysis(
            event_id="taipei_protest_2026",
            articles=[art_a, art_b],
            threshold=0.6,
            embed_fn=embed_fn,
        )

        assert analysis.event_id == "taipei_protest_2026"
        assert len(analysis.claim_clusters) >= 1
        assert "警方" in analysis.entity_outlet_matrix.entity_names or "抗議者" in analysis.entity_outlet_matrix.entity_names


class TestEnglishCorpora:
    def test_english_event_analysis(self):
        """Analyze an English event corpus with multiple outlets."""
        art_a = ParsedArticle(
            article_id="reuters",
            sentences=["Police arrested three suspects on Friday.", "The investigation is ongoing."],
            svo_records=[
                SVORecord(
                    sentence="Police arrested three suspects on Friday.",
                    verb="arrest",
                    subjects=["Police"],
                    objects=["three suspects"],
                    is_passive=False,
                )
            ],
            entity_modifiers=[
                EntityModifier(entity_name="Police", entity_type="ORG", modifiers=["metropolitan"]),
            ],
        )

        art_b = ParsedArticle(
            article_id="bbc",
            sentences=["Three suspects were arrested by police.", "Officials confirmed the detention."],
            svo_records=[
                SVORecord(
                    sentence="Three suspects were arrested by police.",
                    verb="arrest",
                    subjects=["Three suspects"],
                    objects=["police"],
                    is_passive=True,
                    voice_markers=["were"],
                )
            ],
            entity_modifiers=[
                EntityModifier(entity_name="police", entity_type="ORG", modifiers=[]),
            ],
        )

        embed_fn = _make_keyword_embedder([
            ["arrest", "police", "suspects"],
            ["investigation", "ongoing"],
            ["officials", "confirmed", "detention"],
        ])

        analysis = run_event_analysis(
            event_id="london_incident",
            articles=[art_a, art_b],
            threshold=0.6,
            embed_fn=embed_fn,
        )

        assert analysis.event_id == "london_incident"
        assert len(analysis.claim_clusters) >= 1
        assert analysis.consensus_view.total_articles == 2


class TestMixedCorpora:
    def test_mixed_language_event_analysis(self):
        """Event analysis with one Chinese article and one English article."""
        art_zh = ParsedArticle(
            article_id="cna_zh",
            sentences=["立法院通過改革法案。"],
            svo_records=[
                SVORecord(
                    sentence="立法院通過改革法案。",
                    verb="通過",
                    subjects=["立法院"],
                    objects=["改革法案"],
                    is_passive=False,
                )
            ],
            entity_modifiers=[
                EntityModifier(entity_name="立法院", entity_type="ORG", modifiers=[]),
            ],
        )

        art_en = ParsedArticle(
            article_id="taipei_times_en",
            sentences=["Parliament passed the reform bill on Tuesday."],
            svo_records=[
                SVORecord(
                    sentence="Parliament passed the reform bill on Tuesday.",
                    verb="pass",
                    subjects=["Parliament"],
                    objects=["the reform bill"],
                    is_passive=False,
                )
            ],
            entity_modifiers=[
                EntityModifier(entity_name="Parliament", entity_type="ORG", modifiers=[]),
            ],
        )

        # Multilingual sentence transformer aligns Chinese and English semantics
        embed_fn = _make_keyword_embedder([
            ["立法院", "parliament", "通過", "pass", "法案", "bill"],
        ])

        analysis = run_event_analysis(
            event_id="parliament_reform",
            articles=[art_zh, art_en],
            threshold=0.6,
            embed_fn=embed_fn,
        )

        assert analysis.event_id == "parliament_reform"
        assert len(analysis.claim_clusters) == 1
        assert analysis.claim_clusters[0].coverage_count == 2
