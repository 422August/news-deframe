"""CLI entry point for news-deframe.

Usage
-----
    news-deframe diff <file_a> <file_b> [--threshold FLOAT] [--format console|json]
    news-deframe analyze <folder_or_files…>  [--threshold FLOAT] [--format console|json]

Examples
--------
    news-deframe diff article_a.txt article_b.txt
    news-deframe diff article_a.txt article_b.txt --threshold 0.5 --format json
    news-deframe analyze articles/event_001/
    news-deframe analyze a.txt b.txt c.txt --threshold 0.5
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from news_deframe.parser.spacy_loader import get_nlp
from news_deframe.parser.svo import extract_svo
from news_deframe.parser.entities import extract_entity_modifiers
from news_deframe.diff.coverage import compute_coverage
from news_deframe.formatters.console import render_diff_report
from news_deframe.formatters.json_export import report_to_json
from news_deframe.schemas import ParsedArticle

err_console = Console(stderr=True)


def _parse_article(text: str, article_id: str) -> ParsedArticle:
    """Run the full NLP pipeline on *text* and return a :class:`ParsedArticle`.

    Language is auto-detected from *text* (CJK proportion heuristic), so
    no ``--lang`` flag is required — Chinese and English articles are handled
    transparently.
    """
    nlp = get_nlp(text)  # language-aware: detects zh vs en from text
    doc = nlp(text)

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    svo_records = extract_svo(doc)
    entity_modifiers = extract_entity_modifiers(doc)

    return ParsedArticle(
        article_id=article_id,
        svo_records=svo_records,
        entity_modifiers=entity_modifiers,
        sentences=sentences,
    )


@click.group()
def main() -> None:
    """news-deframe – structural framing analysis for news articles."""


@main.command("diff")
@click.argument("file_a", type=click.Path(exists=True, path_type=Path))
@click.argument("file_b", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--threshold",
    default=0.60,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Cosine similarity threshold for matching sentences.",
)
@click.option(
    "--format",
    "output_format",
    default="console",
    show_default=True,
    type=click.Choice(["console", "json"], case_sensitive=False),
    help="Output format.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Write output to file (only used with --format json).",
)
def diff_command(
    file_a: Path,
    file_b: Path,
    threshold: float,
    output_format: str,
    output: Path | None,
) -> None:
    """Compare two news articles for structural framing differences.

    FILE_A and FILE_B should be plain-text files (UTF-8) containing news
    articles in Traditional/Simplified Chinese or English.  Language is
    detected automatically per file — no ``--lang`` flag is needed.
    """
    try:
        text_a = file_a.read_text(encoding="utf-8")
        text_b = file_b.read_text(encoding="utf-8")
    except OSError as exc:
        err_console.print(f"[red]Error reading file:[/red] {exc}")
        sys.exit(1)

    try:
        err_console.print("[dim]Loading NLP model…[/dim]")
        article_a = _parse_article(text_a, file_a.stem)
        article_b = _parse_article(text_b, file_b.stem)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    err_console.print("[dim]Computing semantic alignment…[/dim]")
    report = compute_coverage(article_a, article_b, threshold=threshold)

    if output_format == "json":
        json_str = report_to_json(report)
        if output:
            output.write_text(json_str, encoding="utf-8")
            err_console.print(f"[green]Report saved to {output}[/green]")
        else:
            click.echo(json_str)
    else:
        render_diff_report(report, article_a, article_b)


@main.command("analyze")
@click.argument("sources", nargs=-1, required=True, type=click.Path(path_type=Path))
@click.option(
    "--threshold",
    default=0.60,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Cosine similarity threshold for claim clustering.",
)
@click.option(
    "--format",
    "output_format",
    default="console",
    show_default=True,
    type=click.Choice(["console", "json"], case_sensitive=False),
    help="Output format.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Write JSON output to file (only used with --format json).",
)
@click.option(
    "--clusters",
    "n_clusters",
    default=None,
    type=click.IntRange(1, 20),
    help="Number of framing clusters (default: min(3, n_articles)).",
)
def analyze_command(
    sources: tuple[Path, ...],
    threshold: float,
    output_format: str,
    output: Path | None,
    n_clusters: int | None,
) -> None:
    """Analyse multiple articles about the same event.

    SOURCES may be:

    \b
      • A single directory containing .txt article files:
          news-deframe analyze articles/event_001/

    \b
      • Two or more explicit .txt file paths:
          news-deframe analyze a.txt b.txt c.txt

    Files are read as UTF-8.  Article IDs are derived from filename stems.
    The event ID defaults to the folder name (folder input) or
    'event' (multi-file input).

    \b
    Examples:
      news-deframe analyze articles/event_001/
      news-deframe analyze articles/event_001/ --format json -o report.json
      news-deframe analyze a.txt b.txt c.txt --threshold 0.5
    """
    from news_deframe.parser.article_loader import discover_articles, load_article_files
    from news_deframe.analysis.event import run_event_analysis
    from news_deframe.formatters.event_console import render_event_analysis

    # ── Resolve sources ───────────────────────────────────────────────────────
    source_list = list(sources)
    if len(source_list) == 1 and source_list[0].is_dir():
        folder = source_list[0]
        event_id = folder.name
        err_console.print(
            f"[dim]Discovering articles in '{folder}' (non-recursive)…[/dim]"
        )
        import warnings
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                article_files = discover_articles(folder)
            except (NotADirectoryError, ValueError) as exc:
                err_console.print(f"[red]{exc}[/red]")
                sys.exit(1)

        for w in caught_warnings:
            err_console.print(f"[yellow]Warning:[/yellow] {w.message}")

    else:
        event_id = "event"
        err_console.print(
            f"[dim]Loading {len(source_list)} explicit file path(s)…[/dim]"
        )
        try:
            article_files = load_article_files(source_list)
        except (ValueError, Exception) as exc:
            err_console.print(f"[red]{exc}[/red]")
            sys.exit(1)

    err_console.print(
        f"[dim]Loaded {len(article_files)} article(s).  Parsing…[/dim]"
    )

    # ── Parse articles ────────────────────────────────────────────────────────
    articles: list[ParsedArticle] = []
    for af in article_files:
        err_console.print(f"[dim]  Parsing {af.article_id}…[/dim]")
        try:
            parsed = _parse_article(af.text, af.article_id)
            articles.append(parsed)
        except RuntimeError as exc:
            err_console.print(f"[red]Error parsing {af.article_id}: {exc}[/red]")
            sys.exit(1)

    # ── Run event analysis ────────────────────────────────────────────────────
    err_console.print("[dim]Running event analysis…[/dim]")
    try:
        analysis = run_event_analysis(
            event_id=event_id,
            articles=articles,
            threshold=threshold,
            n_framing_clusters=n_clusters,
        )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Analysis error: {exc}[/red]")
        sys.exit(1)

    # ── Output ────────────────────────────────────────────────────────────────
    if output_format == "json":
        json_str = analysis.model_dump_json(indent=2)
        if output:
            output.write_text(json_str, encoding="utf-8")
            err_console.print(f"[green]Report saved to {output}[/green]")
        else:
            click.echo(json_str)
    else:
        render_event_analysis(analysis)


if __name__ == "__main__":
    main()
