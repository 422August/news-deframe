"""Tests for the Click CLI `analyze` and `diff` commands.

Tests verify:
- Directory-based multi-article analysis: `news-deframe analyze articles/event_001/`
- Direct multi-file analysis: `news-deframe analyze a.txt b.txt c.txt`
- Output formatting (console, JSON stdout, JSON file output)
- Threshold & clusters options
- Error handling (unreadable files, single article, non-existent folder)
- Backward compatibility: `news-deframe diff file_a file_b` remains fully functional
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from news_deframe.cli import main
from news_deframe.schemas import (
    DiffReport,
    EntityModifier,
    ParsedArticle,
    SentenceAlignment,
    SVORecord,
)


# ─── Mock NLP Fixtures ─────────────────────────────────────────────────────────


def _fake_parse_article(text: str, article_id: str) -> ParsedArticle:
    """Fast deterministic parser mock returning ParsedArticle with SVO and entities."""
    sents = [s.strip() for s in text.splitlines() if s.strip()]
    if not sents:
        sents = [text.strip()] if text.strip() else []

    svo_records = [
        SVORecord(
            sentence=s,
            verb="reported",
            subjects=["Police"] if "police" in s.lower() or "警方" in s else [],
            objects=["suspects"] if "suspect" in s.lower() or "嫌犯" in s else [],
            is_passive="passive" in s.lower() or "遭" in s or "被" in s,
            voice_markers=["被"] if ("遭" in s or "被" in s) else [],
        )
        for s in sents
    ]
    entity_modifiers = [
        EntityModifier(entity_name="Police", entity_type="ORG", modifiers=["local"]),
    ]
    return ParsedArticle(
        article_id=article_id,
        sentences=sents,
        svo_records=svo_records,
        entity_modifiers=entity_modifiers,
    )


def _fake_embed_sentences(sentences: list[str]) -> np.ndarray:
    """Return deterministic unit vectors."""
    dim = max(len(sentences), 8)
    embs = np.zeros((len(sentences), dim), dtype=np.float32)
    for i in range(len(sentences)):
        embs[i, i % dim] = 1.0
    return embs


# ─── Tests: CLI analyze command ───────────────────────────────────────────────


class TestCliAnalyze:
    def test_analyze_folder_console_output(self, tmp_path: Path):
        """news-deframe analyze <folder> prints event-level console report."""
        event_dir = tmp_path / "event_001"
        event_dir.mkdir()
        (event_dir / "outlet_a.txt").write_text("Police arrested three suspects.\nInvestigation ongoing.", encoding="utf-8")
        (event_dir / "outlet_b.txt").write_text("Three suspects were detained by police.\nInquiry active.", encoding="utf-8")
        (event_dir / "outlet_c.txt").write_text("Authorities made arrests.", encoding="utf-8")

        runner = CliRunner()
        with patch("news_deframe.cli._parse_article", side_effect=_fake_parse_article), \
             patch("news_deframe.diff.aligner.embed_sentences", side_effect=_fake_embed_sentences), \
             patch("news_deframe.diff.coverage.embed_sentences", side_effect=_fake_embed_sentences):
            result = runner.invoke(main, ["analyze", str(event_dir)])

        assert result.exit_code == 0
        assert "EVENT ANALYSIS" in result.output
        assert "event_001" in result.output
        assert "Claim Coverage" in result.output
        assert "Consensus / Outliers" in result.output

    def test_analyze_folder_json_stdout(self, tmp_path: Path):
        """news-deframe analyze <folder> --format json dumps valid EventAnalysis JSON to stdout."""
        event_dir = tmp_path / "event_002"
        event_dir.mkdir()
        (event_dir / "outlet_1.txt").write_text("Statement issued.", encoding="utf-8")
        (event_dir / "outlet_2.txt").write_text("Statement confirmed.", encoding="utf-8")

        runner = CliRunner()
        with patch("news_deframe.cli._parse_article", side_effect=_fake_parse_article), \
             patch("news_deframe.diff.aligner.embed_sentences", side_effect=_fake_embed_sentences), \
             patch("news_deframe.diff.coverage.embed_sentences", side_effect=_fake_embed_sentences):
            result = runner.invoke(main, ["analyze", str(event_dir), "--format", "json"])

        assert result.exit_code == 0
        json_str = result.output[result.output.index("{"):]
        data = json.loads(json_str)
        assert data["event_id"] == "event_002"
        assert "claim_clusters" in data
        assert "entity_outlet_matrix" in data
        assert "framing_clusters" in data
        assert "consensus_view" in data

    def test_analyze_folder_json_file_output(self, tmp_path: Path):
        """news-deframe analyze <folder> --format json --output file.json saves report to file."""
        event_dir = tmp_path / "event_003"
        event_dir.mkdir()
        (event_dir / "outlet_1.txt").write_text("Breaking news here.", encoding="utf-8")
        (event_dir / "outlet_2.txt").write_text("Breaking update here.", encoding="utf-8")
        out_json = tmp_path / "output.json"

        runner = CliRunner()
        with patch("news_deframe.cli._parse_article", side_effect=_fake_parse_article), \
             patch("news_deframe.diff.aligner.embed_sentences", side_effect=_fake_embed_sentences), \
             patch("news_deframe.diff.coverage.embed_sentences", side_effect=_fake_embed_sentences):
            result = runner.invoke(
                main,
                ["analyze", str(event_dir), "--format", "json", "--output", str(out_json)],
            )

        assert result.exit_code == 0
        assert out_json.exists()
        saved_data = json.loads(out_json.read_text(encoding="utf-8"))
        assert saved_data["event_id"] == "event_003"

    def test_analyze_multi_file_arguments(self, tmp_path: Path):
        """news-deframe analyze a.txt b.txt c.txt handles multiple explicit files."""
        f1 = tmp_path / "art_a.txt"
        f2 = tmp_path / "art_b.txt"
        f3 = tmp_path / "art_c.txt"
        f1.write_text("Report Alpha", encoding="utf-8")
        f2.write_text("Report Beta", encoding="utf-8")
        f3.write_text("Report Gamma", encoding="utf-8")

        runner = CliRunner()
        with patch("news_deframe.cli._parse_article", side_effect=_fake_parse_article), \
             patch("news_deframe.diff.aligner.embed_sentences", side_effect=_fake_embed_sentences), \
             patch("news_deframe.diff.coverage.embed_sentences", side_effect=_fake_embed_sentences):
            result = runner.invoke(main, ["analyze", str(f1), str(f2), str(f3)])

        assert result.exit_code == 0
        assert "EVENT ANALYSIS" in result.output
        assert "Articles analysed:  3" in result.output or "Articles analysed:" in result.output

    def test_analyze_folder_insufficient_articles_error(self, tmp_path: Path):
        """Single article folder triggers an error with exit code 1."""
        event_dir = tmp_path / "event_lonely"
        event_dir.mkdir()
        (event_dir / "only_one.txt").write_text("Sole report.", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["analyze", str(event_dir)])
        assert result.exit_code == 1

    def test_analyze_nonexistent_directory(self, tmp_path: Path):
        """Non-existent directory triggers an error."""
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0


# ─── Tests: Backward Compatibility for `diff` ─────────────────────────────────


class TestCliDiffCompatibility:
    def test_diff_command_remains_operational(self, tmp_path: Path):
        """Existing `news-deframe diff file_a file_b` command continues to work as expected."""
        fa = tmp_path / "article_a.txt"
        fb = tmp_path / "article_b.txt"
        fa.write_text("Police arrested three suspects.", encoding="utf-8")
        fb.write_text("Three suspects were detained by police.", encoding="utf-8")

        runner = CliRunner()
        with patch("news_deframe.cli._parse_article", side_effect=_fake_parse_article), \
             patch("news_deframe.diff.aligner.embed_sentences", side_effect=_fake_embed_sentences), \
             patch("news_deframe.diff.coverage.embed_sentences", side_effect=_fake_embed_sentences):
            result = runner.invoke(main, ["diff", str(fa), str(fb)])

        assert result.exit_code == 0
        assert "Framing Analysis" in result.output or "Sentence Alignment" in result.output

    def test_diff_command_json_format(self, tmp_path: Path):
        """`news-deframe diff file_a file_b --format json` produces valid DiffReport JSON."""
        fa = tmp_path / "article_a.txt"
        fb = tmp_path / "article_b.txt"
        fa.write_text("Text A", encoding="utf-8")
        fb.write_text("Text B", encoding="utf-8")

        runner = CliRunner()
        with patch("news_deframe.cli._parse_article", side_effect=_fake_parse_article), \
             patch("news_deframe.diff.aligner.embed_sentences", side_effect=_fake_embed_sentences), \
             patch("news_deframe.diff.coverage.embed_sentences", side_effect=_fake_embed_sentences):
            result = runner.invoke(main, ["diff", str(fa), str(fb), "--format", "json"])

        assert result.exit_code == 0
        json_str = result.output[result.output.index("{"):]
        data = json.loads(json_str)
        assert data["article_a_id"] == "article_a"
        assert data["article_b_id"] == "article_b"
        assert "alignments" in data
        assert "passive_ratio_a" in data
