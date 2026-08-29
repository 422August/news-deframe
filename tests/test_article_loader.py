"""Deterministic offline tests for article discovery and loading (news_deframe.parser.article_loader).
"""
from __future__ import annotations

import os
import pytest
from pathlib import Path

from news_deframe.parser.article_loader import (
    ArticleFile,
    ArticleLoadError,
    discover_articles,
    load_article_files,
)


class TestDiscoverArticles:
    def test_folder_discovery_and_ordering(self, tmp_path: Path):
        """Discovered files must be sorted deterministically in lexicographical order."""
        (tmp_path / "outlet_c.txt").write_text("Article C content.", encoding="utf-8")
        (tmp_path / "outlet_a.txt").write_text("Article A content.", encoding="utf-8")
        (tmp_path / "outlet_b.txt").write_text("Article B content.", encoding="utf-8")

        articles = discover_articles(tmp_path)
        assert len(articles) == 3
        assert [a.article_id for a in articles] == ["outlet_a", "outlet_b", "outlet_c"]
        assert articles[0].text == "Article A content."
        assert articles[1].text == "Article B content."
        assert articles[2].text == "Article C content."

    def test_non_recursive_by_default(self, tmp_path: Path):
        """Nested directories should not be searched by default."""
        (tmp_path / "outlet_1.txt").write_text("Article 1", encoding="utf-8")
        (tmp_path / "outlet_2.txt").write_text("Article 2", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "outlet_3.txt").write_text("Article 3", encoding="utf-8")

        articles = discover_articles(tmp_path, recursive=False)
        assert len(articles) == 2
        assert [a.article_id for a in articles] == ["outlet_1", "outlet_2"]

    def test_recursive_discovery(self, tmp_path: Path):
        """Recursive discovery should include files in sub-folders."""
        (tmp_path / "outlet_1.txt").write_text("Article 1", encoding="utf-8")
        (tmp_path / "outlet_2.txt").write_text("Article 2", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "outlet_3.txt").write_text("Article 3", encoding="utf-8")

        articles = discover_articles(tmp_path, recursive=True)
        assert len(articles) == 3
        assert [a.article_id for a in articles] == ["outlet_1", "outlet_2", "outlet_3"]

    def test_ignoring_unsupported_files(self, tmp_path: Path):
        """Non-.txt files (e.g. .pdf, .json, .md, .png) must be ignored."""
        (tmp_path / "outlet_a.txt").write_text("Article A", encoding="utf-8")
        (tmp_path / "outlet_b.txt").write_text("Article B", encoding="utf-8")
        (tmp_path / "readme.md").write_text("# Readme", encoding="utf-8")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")

        articles = discover_articles(tmp_path)
        assert len(articles) == 2
        assert [a.article_id for a in articles] == ["outlet_a", "outlet_b"]

    def test_ignoring_hidden_files(self, tmp_path: Path):
        """Hidden files starting with a dot (like .gitkeep, .DS_Store) must be ignored."""
        (tmp_path / "outlet_a.txt").write_text("Article A", encoding="utf-8")
        (tmp_path / "outlet_b.txt").write_text("Article B", encoding="utf-8")
        (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
        (tmp_path / ".hidden.txt").write_text("Hidden article", encoding="utf-8")

        articles = discover_articles(tmp_path)
        assert len(articles) == 2
        assert [a.article_id for a in articles] == ["outlet_a", "outlet_b"]

    def test_empty_folder_raises_value_error(self, tmp_path: Path):
        """An empty folder has < 2 articles and must raise ValueError."""
        with pytest.raises(ValueError, match="usable article"):
            discover_articles(tmp_path)

    def test_one_article_folder_raises_value_error(self, tmp_path: Path):
        """A folder with only 1 usable article must raise ValueError."""
        (tmp_path / "outlet_a.txt").write_text("Single article", encoding="utf-8")
        with pytest.raises(ValueError, match="Found only 1 usable article"):
            discover_articles(tmp_path)

    def test_nonexistent_directory_raises_not_a_directory(self, tmp_path: Path):
        """Nonexistent path raises NotADirectoryError."""
        fake_path = tmp_path / "does_not_exist"
        with pytest.raises(NotADirectoryError):
            discover_articles(fake_path)

    def test_file_passed_as_directory_raises_not_a_directory(self, tmp_path: Path):
        """Passing a file path instead of a folder raises NotADirectoryError."""
        f = tmp_path / "file.txt"
        f.write_text("Content", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            discover_articles(f)

    def test_invalid_utf8_file_skipped_with_warning(self, tmp_path: Path):
        """Files containing invalid UTF-8 bytes should trigger a warning and be skipped."""
        (tmp_path / "outlet_a.txt").write_text("Valid article A", encoding="utf-8")
        (tmp_path / "outlet_b.txt").write_text("Valid article B", encoding="utf-8")
        (tmp_path / "outlet_bad.txt").write_bytes(b"\xff\xfe\xfa\xfb Invalid binary bytes")

        with pytest.warns(UserWarning, match="not valid UTF-8"):
            articles = discover_articles(tmp_path)

        assert len(articles) == 2
        assert [a.article_id for a in articles] == ["outlet_a", "outlet_b"]

    def test_empty_or_whitespace_file_skipped(self, tmp_path: Path):
        """Files containing zero bytes or only whitespace should be skipped."""
        (tmp_path / "outlet_a.txt").write_text("Valid article A", encoding="utf-8")
        (tmp_path / "outlet_b.txt").write_text("Valid article B", encoding="utf-8")
        (tmp_path / "outlet_empty.txt").write_text("   \n\t  ", encoding="utf-8")

        with pytest.warns(UserWarning, match="empty or contains only whitespace"):
            articles = discover_articles(tmp_path)

        assert len(articles) == 2
        assert [a.article_id for a in articles] == ["outlet_a", "outlet_b"]


class TestLoadArticleFiles:
    def test_explicit_files_loaded_in_order(self, tmp_path: Path):
        p1 = tmp_path / "z.txt"
        p2 = tmp_path / "a.txt"
        p1.write_text("Article Z", encoding="utf-8")
        p2.write_text("Article A", encoding="utf-8")

        articles = load_article_files([p1, p2])
        assert len(articles) == 2
        assert articles[0].article_id == "z"
        assert articles[1].article_id == "a"

    def test_fewer_than_two_files_raises(self, tmp_path: Path):
        p1 = tmp_path / "a.txt"
        p1.write_text("Article A", encoding="utf-8")
        with pytest.raises(ValueError, match="At least 2"):
            load_article_files([p1])

    def test_unsupported_file_in_list_emits_warning(self, tmp_path: Path):
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p3 = tmp_path / "c.pdf"
        p1.write_text("Article A", encoding="utf-8")
        p2.write_text("Article B", encoding="utf-8")
        p3.write_bytes(b"%PDF-1.4")

        with pytest.warns(UserWarning, match="Unsupported file type"):
            articles = load_article_files([p1, p2, p3])

        assert len(articles) == 2
        assert [a.article_id for a in articles] == ["a", "b"]


class TestCleanArticleText:
    """Tests for input hygiene and scraped webpage noise stripping."""

    def test_strips_webpage_noise_and_preserves_content(self):
        from news_deframe.parser.article_loader import clean_article_text

        raw_article = (
            "某市今晨發生重大火警，警消緊急疏散百名住戶。（中央社）\n"
            "\n"
            "0\n"
            "2026年3月15日\n"
            "加入為 Google 偏好來源\n"
            "（另開新視窗）\n"
            "\n"
            "火勢在兩小時內獲得控制，現場無人傷亡。\n"
            "廣告\n"
            "Ad\n"
            "【精選建案】市中心景觀名邸 限量席次\n"
            "請繼續往下閱讀...\n"
            "\n"
            "市府宣布啟動災後調查與安置作業（14:30）\n"
            "延伸閱讀\n"
            "三立新聞網 setn.com\n"
            "追蹤中央社\n"
        )

        cleaned = clean_article_text(raw_article)
        expected_lines = [
            "某市今晨發生重大火警，警消緊急疏散百名住戶。",
            "火勢在兩小時內獲得控制，現場無人傷亡。",
            "市府宣布啟動災後調查與安置作業",
        ]
        assert cleaned == "\n".join(expected_lines)

    def test_english_metadata_stripping(self):
        from news_deframe.parser.article_loader import clean_article_text

        raw_en = (
            "A fire broke out in the warehouse on Tuesday morning. (Photo: Reuters)\n"
            "Follow us on Twitter for live updates\n"
            "Read more\n"
            "Firefighters contained the blaze within three hours.\n"
            "Sponsored\n"
        )
        cleaned = clean_article_text(raw_en)
        expected = (
            "A fire broke out in the warehouse on Tuesday morning.\n"
            "Firefighters contained the blaze within three hours."
        )
        assert cleaned == expected
