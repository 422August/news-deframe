"""Article discovery and loading utilities.

Folder Loading
--------------
By default, ``discover_articles`` performs **non-recursive** discovery of
``.txt`` files inside the given directory.  Hidden files (names starting with
``.``) and zero-byte files are silently skipped.  Files are sorted
lexicographically so that the discovery order is always deterministic
regardless of filesystem ordering.

Convention
----------
::

    articles/
    └── event_001/
        ├── outlet_a.txt
        ├── outlet_b.txt
        └── outlet_c.txt

The folder name (``event_001``) is used as the *event ID* and each filename
stem (``outlet_a``, …) is used as the *article / outlet ID*.

The program does **not** require the corpus to live inside the repository.
Paths outside the project root are fully supported::

    news-deframe analyze /path/to/my/event_articles/
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

#: File extensions recognised as article files.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt"})

_NOISE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 1. Pure whitespace/bullets/counters/separators
    re.compile(r"^[\d\s\-_–—|/•·*#~`=+\.,:;!?，。！？；：、\(\)（）\[\]【】\"\'“”‘’]+$"),
    # 2. Advertisement / sponsored text / promo cards
    re.compile(r"^(?:廣告|Ad|AD|Advertisement|Sponsored|贊助商連結|贊助|廣編特輯|業配)[：:\s]*$", re.IGNORECASE),
    re.compile(r"^【[^】]+】\s*[^。！？!?]{2,50}$"),
    # 3. Read more / related articles / navigation / window instructions
    re.compile(r"^(?:請繼續往下閱讀|延伸閱讀|推薦閱讀|相關新聞|相關報導|熱門新聞|點我看更多|更多新聞|最新消息|熱門話題|Read more|Related articles|Recommended|You may also like)[.。…\s]*$", re.IGNORECASE),
    re.compile(r"^[（(【\[\s]*(?:另開新視窗|另開視窗|點擊看大圖|點擊放大|點圖放大|點此觀看|詳見影片|點我看|點這裡|click here|open in new window)[）)】\]\s]*$", re.IGNORECASE),
    # 4. Social / follow / subscribe / share
    re.compile(r"^(?:透過|加入|加入為|追蹤|訂閱|按讚|分享|關注|Follow|Subscribe to|Sign up for)\s*.+$", re.IGNORECASE),
    # 5. Standalone publisher / media outlet signature without narrative sentence
    re.compile(r"^[\w\u4e00-\u9fff\s\.-]{2,30}(?:新聞網|日報|電子報|通訊社|廣播公司|電視台|新聞|News|Times|Post|Daily)(?:\s+[\w\.-]+)?$", re.IGNORECASE),
    # 6. Standalone author signature / date / timestamp / editorial code
    re.compile(r"^(?:文|記者|特派員|攝影|撰文|編譯|責任編輯|編輯|Author|By)[\s／/：:][^\s]{2,15}(?:／[^\s]{2,15})?$", re.IGNORECASE),
    re.compile(r"^\d{4}[年\.-]\d{1,2}[月\.-]\d{1,2}日?(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$"),
    re.compile(r"^[（(【\[\s]*(?:圖|翻攝|攝影|資料照|中央社|共同社|NHK|路透|美聯社|法新社|Photo|Image|Credit|Source|圖取自|翻攝自|截圖自)[：:\s／/].*[）)】\]\s]*$", re.IGNORECASE),
)

_TRAILING_METADATA_PATTERN = re.compile(
    r"\s*[（(【\[](?:翻攝|圖取自|共同社|中央社|美聯社|路透|法新社|編譯|記者|攝影|資料照|NHK|Photo|Image|Credit|Source|X|Twitter|Facebook|FB)[\s\S]*?[）)】\]]\s*(?:\d{5,8})?$",
    re.IGNORECASE,
)

_TRAILING_TIMESTAMP_PATTERN = re.compile(r"\s*[(（]\d{1,2}:\d{2}[)）]$")


def clean_article_text(raw_text: str) -> str:
    """Strip generic webpage noise, advertisements, follow prompts, and metadata from article text."""
    cleaned_lines: list[str] = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s or any(p.match(s) for p in _NOISE_LINE_PATTERNS):
            continue
        s = _TRAILING_METADATA_PATTERN.sub("", s)
        s = _TRAILING_TIMESTAMP_PATTERN.sub("", s)
        s = s.strip()
        if s and not any(p.match(s) for p in _NOISE_LINE_PATTERNS):
            cleaned_lines.append(s)
    return "\n".join(cleaned_lines)


# ── Data class ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArticleFile:
    """A successfully loaded article file ready for NLP processing."""

    path: Path
    """Absolute path to the source file."""

    article_id: str
    """Outlet / article identifier derived from the filename stem."""

    text: str
    """Decoded UTF-8 text content (guaranteed non-empty)."""


# ── Errors ────────────────────────────────────────────────────────────────────


class ArticleLoadError(ValueError):
    """Raised when a candidate file cannot be used as an article."""


# ── Internal helpers ──────────────────────────────────────────────────────────


def _is_hidden(path: Path) -> bool:
    """Return True if *path*'s name starts with a dot."""
    return path.name.startswith(".")


def _is_supported(path: Path) -> bool:
    """Return True when the file's suffix is in SUPPORTED_EXTENSIONS."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _load_file(path: Path) -> ArticleFile:
    """Read and validate a single candidate file.

    Parameters
    ----------
    path:
        Path to the candidate article file.

    Returns
    -------
    ArticleFile

    Raises
    ------
    ArticleLoadError
        When the file is unreadable, not valid UTF-8, or empty after stripping.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArticleLoadError(f"Cannot read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ArticleLoadError(
            f"{path} is not valid UTF-8 — skipping (use UTF-8 encoded text files)."
        ) from exc

    stripped = text.strip()
    if not stripped:
        raise ArticleLoadError(f"{path} is empty or contains only whitespace.")

    return ArticleFile(
        path=path.resolve(),
        article_id=path.stem,
        text=stripped,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def discover_articles(
    folder: Path,
    *,
    recursive: bool = False,
) -> list[ArticleFile]:
    """Discover and load all supported article files in *folder*.

    By default the search is **non-recursive** — only direct children of
    *folder* are considered.  Pass ``recursive=True`` to traverse sub-folders.

    Files are returned in deterministic lexicographic order by path.

    Parameters
    ----------
    folder:
        Directory to search.
    recursive:
        When *True*, search all descendant directories as well.

    Returns
    -------
    list[ArticleFile]
        Successfully loaded articles, sorted by path.

    Raises
    ------
    NotADirectoryError
        When *folder* does not exist or is not a directory.
    ValueError
        When fewer than 2 usable articles are found (the minimum required for
        any meaningful multi-article analysis).
    ArticleLoadError
        Individual file errors are collected and emitted as warnings;
        they do not abort the whole discovery unless no usable files remain.
    """
    if not folder.exists():
        raise NotADirectoryError(f"Path does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory, got a file: {folder}")

    glob_pattern = "**/*" if recursive else "*"
    candidates = sorted(
        p for p in folder.glob(glob_pattern)
        if p.is_file() and not _is_hidden(p) and _is_supported(p)
    )

    articles: list[ArticleFile] = []
    errors: list[str] = []

    for path in candidates:
        try:
            articles.append(_load_file(path))
        except ArticleLoadError as exc:
            errors.append(str(exc))

    if errors:
        import warnings
        for msg in errors:
            warnings.warn(msg, stacklevel=3)

    if len(articles) < 2:
        found = len(articles)
        total = len(candidates)
        raise ValueError(
            f"Found only {found} usable article(s) in '{folder}' "
            f"(discovered {total} candidate file(s)).  "
            "At least 2 articles are required for event analysis."
        )

    return articles


def load_article_files(paths: list[Path]) -> list[ArticleFile]:
    """Load a list of explicit file paths as articles.

    Unlike :func:`discover_articles` this function does not perform directory
    scanning — callers supply the exact file list.  All paths must be regular
    files with a supported extension.

    Parameters
    ----------
    paths:
        Ordered list of file paths to load.

    Returns
    -------
    list[ArticleFile]
        Successfully loaded articles in the same order as *paths*.

    Raises
    ------
    ValueError
        When fewer than 2 paths are supplied or fewer than 2 load successfully.
    ArticleLoadError
        For individual file failures (raised immediately — not collected).
    """
    if len(paths) < 2:
        raise ValueError(
            f"At least 2 article files are required; got {len(paths)}."
        )

    articles: list[ArticleFile] = []
    for path in paths:
        if not _is_supported(path):
            import warnings
            warnings.warn(
                f"Unsupported file type '{path.suffix}' for {path} — skipping.",
                stacklevel=2,
            )
            continue
        articles.append(_load_file(path))

    if len(articles) < 2:
        raise ValueError(
            f"Fewer than 2 usable articles after loading {len(paths)} path(s)."
        )

    return articles
