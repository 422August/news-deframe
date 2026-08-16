"""CLI entry point for news-deframe.

Usage
-----
    news-deframe diff <file_a> <file_b> [--threshold FLOAT] [--format console|json]

Examples
--------
    news-deframe diff article_a.txt article_b.txt
    news-deframe diff article_a.txt article_b.txt --threshold 0.5 --format json
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


if __name__ == "__main__":
    main()
