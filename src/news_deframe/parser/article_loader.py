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

from dataclasses import dataclass
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

#: File extensions recognised as article files.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt"})


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
