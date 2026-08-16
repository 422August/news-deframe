"""Rich terminal formatter for DiffReport.

Renders three panels:
1.  Sentence Alignment Table  – side-by-side with similarity score and passive flag.
2.  Unshared Claims           – sentences unique to each article.
3.  Entity Modifier Contrast  – named entities and their descriptive modifiers.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from news_deframe.schemas import DiffReport, ParsedArticle

console = Console()


def _passive_badge(is_passive: bool) -> Text:
    if is_passive:
        return Text("● PASSIVE", style="bold red")
    return Text("○ active", style="dim green")


def render_diff_report(
    report: DiffReport,
    article_a: ParsedArticle,
    article_b: ParsedArticle,
) -> None:
    """Print a rich-formatted comparison of two articles to stdout.

    Parameters
    ----------
    report:
        The computed :class:`DiffReport`.
    article_a, article_b:
        Parsed articles (used for entity modifier sections).
    """
    c = Console()

    # ── Header ──────────────────────────────────────────────────────────────
    c.print(
        Panel(
            f"[bold]Comparing:[/bold]  [cyan]{report.article_a_id}[/cyan]  vs  [yellow]{report.article_b_id}[/yellow]\n"
            f"Passive ratio A: [red]{report.passive_ratio_a:.1%}[/red]   "
            f"Passive ratio B: [red]{report.passive_ratio_b:.1%}[/red]",
            title="[bold white]news-deframe · Framing Analysis[/bold white]",
            border_style="bright_blue",
            expand=True,
        )
    )

    # ── Alignment Table ──────────────────────────────────────────────────────
    align_table = Table(
        title="Sentence Alignment",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
        header_style="bold magenta",
    )
    align_table.add_column(f"[cyan]{report.article_a_id}[/cyan]", ratio=45, no_wrap=False)
    align_table.add_column("Sim", ratio=6, justify="center")
    align_table.add_column(f"[yellow]{report.article_b_id}[/yellow]", ratio=45, no_wrap=False)
    align_table.add_column("Voice A", ratio=12, justify="center")

    # Build a quick lookup: sentence_a → is_passive from SVO records
    passive_map_a: dict[str, bool] = {
        r.sentence: r.is_passive for r in article_a.svo_records
    }

    for alignment in report.alignments:
        score = alignment.similarity_score
        if score >= 0.8:
            score_style = "bold green"
        elif score >= 0.6:
            score_style = "yellow"
        else:
            score_style = "dim red"

        sent_b_text = alignment.sent_b if alignment.sent_b else Text("[not covered]", style="dim italic")
        is_passive = passive_map_a.get(alignment.sent_a, False)

        align_table.add_row(
            alignment.sent_a,
            Text(f"{score:.2f}", style=score_style),
            sent_b_text,
            _passive_badge(is_passive),
        )

    c.print(align_table)

    # ── Unshared Claims ──────────────────────────────────────────────────────
    if report.unshared_claims_a or report.unshared_claims_b:
        claims_table = Table(
            title="Unshared Claims (Omissions)",
            box=box.SIMPLE_HEAVY,
            expand=True,
            header_style="bold white",
        )
        claims_table.add_column(
            f"Only in [cyan]{report.article_a_id}[/cyan]",
            ratio=50,
            style="cyan",
        )
        claims_table.add_column(
            f"Only in [yellow]{report.article_b_id}[/yellow]",
            ratio=50,
            style="yellow",
        )

        max_rows = max(len(report.unshared_claims_a), len(report.unshared_claims_b))
        for i in range(max_rows):
            cell_a = report.unshared_claims_a[i] if i < len(report.unshared_claims_a) else ""
            cell_b = report.unshared_claims_b[i] if i < len(report.unshared_claims_b) else ""
            claims_table.add_row(cell_a, cell_b)

        c.print(claims_table)

    # ── Framing & Evaluative Descriptors ─────────────────────────────────────
    em_table = Table(
        title="Framing & Evaluative Descriptors",
        box=box.SIMPLE,
        expand=True,
        header_style="bold white",
    )
    em_table.add_column("Target", style="bold")
    em_table.add_column("Type", justify="center")
    em_table.add_column(f"Modifiers in [cyan]{report.article_a_id}[/cyan]", style="cyan")
    em_table.add_column(f"Modifiers in [yellow]{report.article_b_id}[/yellow]", style="yellow")

    # Merge framing descriptors by name+type
    entity_map_a: dict[tuple[str, str], list[str]] = {
        (em.entity_name, em.entity_type): em.modifiers for em in article_a.entity_modifiers
    }
    entity_map_b: dict[tuple[str, str], list[str]] = {
        (em.entity_name, em.entity_type): em.modifiers for em in article_b.entity_modifiers
    }
    all_keys = sorted(entity_map_a.keys() | entity_map_b.keys())

    _TYPE_TAG: dict[str, str] = {
        "VERB_ACTION": "ACTION",
        "EVENT_NOUN": "EVENT",
        "PERSON": "PERSON",
        "PER": "PERSON",
        "ORG": "ORG",
        "GPE": "GPE",
        "LOC": "LOC",
        "NORP": "NORP",
        "FAC": "FAC",
        "EVENT": "EVENT",
    }

    rows_added = 0
    for key in all_keys:
        name, label = key
        mods_a = entity_map_a.get(key, [])
        mods_b = entity_map_b.get(key, [])

        # Omit entries where neither article has any modifiers
        if not mods_a and not mods_b:
            continue

        tag = _TYPE_TAG.get(label, label)
        tagged_name = f"[{tag}] {name}"
        em_table.add_row(
            tagged_name,
            label,
            ", ".join(mods_a) if mods_a else "—",
            ", ".join(mods_b) if mods_b else "—",
        )
        rows_added += 1

    if rows_added:
        c.print(em_table)
