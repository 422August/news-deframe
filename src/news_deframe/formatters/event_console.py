"""Rich terminal formatter for EventAnalysis.

Produces a structured, human-readable event-level report with sections:

1.  Header \u2014 event ID and article count.
2.  Claim Coverage \u2014 per-cluster summary with coverage ratio.
3.  Consensus / Outliers \u2014 frequency categories and absent outlets.
4.  Entity \u00d7 Outlet Framing \u2014 agent/patient ratio table.
5.  Framing Clusters \u2014 which articles share structural profiles.
6.  Detailed Claims \u2014 full source sentences per cluster.

Output is intended to remain readable for corpora of 10\u201320 articles.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from news_deframe.analysis.schemas import (
    ClaimCluster,
    ConsensusView,
    EntityOutletMatrix,
    EventAnalysis,
    FramingCluster,
)

_CATEGORY_STYLE: dict[str, str] = {
    "Widely shared": "bold green",
    "Commonly reported": "green",
    "Minority coverage": "yellow",
    "Rare claim": "dim",
}


def _ratio_bar(ratio: float, width: int = 20) -> str:
    """Return a simple ASCII bar representing *ratio* in [0, 1]."""
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def render_event_analysis(analysis: EventAnalysis, *, console: Console | None = None) -> None:
    """Print a rich-formatted event analysis to stdout.

    Parameters
    ----------
    analysis:
        The computed :class:`EventAnalysis`.
    console:
        Optional Rich console instance.  Creates a new stdout console if omitted.
    """
    c = console or Console()

    n_articles = len(analysis.article_ids)

    # ── Header ───────────────────────────────────────────────────────────────
    header_text = (
        f"[bold]Event:[/bold]  [cyan]{analysis.event_id}[/cyan]\n"
        f"[bold]Articles analysed:[/bold]  {n_articles}\n"
        f"[bold]Claim clusters:[/bold]  {len(analysis.claim_clusters)}\n"
        f"[bold]Framing clusters:[/bold]  {len(analysis.framing_clusters)}"
    )
    c.print(
        Panel(
            header_text,
            title="[bold white]NEWS-DEFRAME  EVENT ANALYSIS[/bold white]",
            border_style="bright_blue",
            expand=True,
        )
    )

    # ── Claim Coverage ───────────────────────────────────────────────────────
    c.rule("[bold]Claim Coverage[/bold]")
    if not analysis.claim_clusters:
        c.print("[dim]No claim clusters found.[/dim]")
    else:
        cov_table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        cov_table.add_column("Cluster", style="bold", width=8)
        cov_table.add_column("Coverage", width=10, justify="right")
        cov_table.add_column("Bar", width=22)
        cov_table.add_column("Representative claim", no_wrap=False)

        for cluster in analysis.claim_clusters[:30]:  # cap for readability
            ratio_text = f"{cluster.coverage_count}/{cluster.total_articles}"
            bar = _ratio_bar(cluster.coverage_ratio)
            rep = cluster.representative[:120] + ("\u2026" if len(cluster.representative) > 120 else "")
            cov_table.add_row(cluster.cluster_id, ratio_text, bar, rep)

        c.print(cov_table)

    # ── Consensus / Outliers ─────────────────────────────────────────────────
    c.rule("[bold]Consensus / Outliers[/bold]")
    consensus = analysis.consensus_view
    if not consensus.claims:
        c.print("[dim]No consensus data.[/dim]")
    else:
        con_table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold white",
            expand=True,
        )
        con_table.add_column("Cluster", width=8)
        con_table.add_column("Category", width=20)
        con_table.add_column("Coverage", width=10, justify="right")
        con_table.add_column("Absent from", no_wrap=False)

        for claim in consensus.claims:
            style = _CATEGORY_STYLE.get(claim.coverage_category, "")
            absent_str = ", ".join(claim.outlets_absent) if claim.outlets_absent else "\u2014"
            con_table.add_row(
                claim.cluster_id,
                Text(claim.coverage_category, style=style),
                f"{claim.coverage_count}/{claim.total_articles}",
                absent_str,
            )

        c.print(con_table)

    # ── Entity × Outlet Framing ──────────────────────────────────────────────
    c.rule("[bold]Entity \u00d7 Outlet Framing[/bold]")
    matrix = analysis.entity_outlet_matrix
    if not matrix.entity_names or not matrix.article_ids:
        c.print("[dim]No entity framing data.[/dim]")
    else:
        em_table = Table(
            title="Agent ratio  |  Patient ratio  (subject / total, object / total)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        em_table.add_column("Entity", style="bold", min_width=18)
        for aid in matrix.article_ids:
            em_table.add_column(aid, justify="center", min_width=12)

        # Map (entity, article) -> profile
        profile_map = {
            (p.entity_name, p.article_id): p for p in matrix.profiles
        }

        for entity_name in matrix.entity_names[:20]:  # cap rows for readability
            cells = []
            for aid in matrix.article_ids:
                profile = profile_map.get((entity_name, aid))
                if profile and profile.total_mentions > 0:
                    cells.append(
                        f"ag {profile.agent_ratio:.2f}\npt {profile.patient_ratio:.2f}"
                    )
                else:
                    cells.append("[dim]\u2014[/dim]")
            em_table.add_row(entity_name, *cells)

        c.print(em_table)

    # ── Framing Clusters ─────────────────────────────────────────────────────
    c.rule("[bold]Framing Clusters[/bold]")
    if not analysis.framing_clusters:
        c.print("[dim]No framing clusters found.[/dim]")
    else:
        for fc in analysis.framing_clusters:
            members = ", ".join(fc.article_ids)
            centroid_str = "  ".join(
                f"{k}: {v:.2f}" for k, v in fc.centroid_description.items()
            )
            c.print(
                Panel(
                    f"[bold]Members:[/bold] {members}\n"
                    f"[dim]Centroid \u2014 {centroid_str}[/dim]",
                    title=f"[bold cyan]{fc.label}[/bold cyan]",
                    border_style="dim",
                )
            )

    # ── Detailed Claims ──────────────────────────────────────────────────────
    c.rule("[bold]Detailed Claims[/bold]")
    if not analysis.claim_clusters:
        c.print("[dim]No detailed claim data.[/dim]")
    else:
        for cluster in analysis.claim_clusters[:20]:  # cap for readability
            sources_str = "\n".join(
                f"  [{src.article_id}] (sim={src.similarity:.2f})  {src.text[:100]}"
                for src in cluster.sources
            )
            c.print(
                Panel(
                    f"[bold]Representative:[/bold]\n  {cluster.representative}\n\n"
                    f"[bold]Coverage:[/bold] {cluster.coverage_count}/{cluster.total_articles}  "
                    f"({cluster.coverage_ratio:.0%})\n\n"
                    f"[bold]Sources:[/bold]\n{sources_str}",
                    title=f"[bold]{cluster.cluster_id}[/bold]",
                    border_style="blue",
                )
            )
