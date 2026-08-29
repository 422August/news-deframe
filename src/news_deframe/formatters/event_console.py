"""Rich terminal formatter for EventAnalysis.

Produces a structured, human-readable event-level report designed for
social-science and media research use.

Presentation levels:
1. Default: Concise research-oriented summary (compact header, merged claim
   coverage, actor-oriented framing blocks, framing cluster membership,
   and research interpretation notes).
2. --details: Claim-level evidence and source sentences for manual inspection.
3. --verbose: Technical diagnostics (similarity scores, framing centroids,
   exact ratio denominators, full actor lists).
"""
from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from news_deframe.analysis.schemas import (
    ClaimCluster,
    ClaimConsensus,
    EntityOutletMatrix,
    EventAnalysis,
    FramingCluster,
)

_CATEGORY_STYLE: dict[str, str] = {
    "Widely shared": "bold green",
    "Commonly reported": "green",
    "Minority coverage": "yellow",
    "Single-outlet": "dim",
    "Rare claim": "dim",
}


def _display_category(claim: ClaimConsensus | None, coverage_count: int) -> str:
    """Return a researcher-friendly display label for a claim's coverage."""
    if coverage_count == 1:
        return "Single-outlet"
    if claim is not None:
        return claim.coverage_category
    return "Shared"


def render_event_analysis(
    analysis: EventAnalysis,
    *,
    details: bool = False,
    verbose: bool = False,
    console: Console | None = None,
) -> None:
    """Print a rich-formatted event analysis to stdout.

    Parameters
    ----------
    analysis:
        The computed :class:`EventAnalysis`.
    details:
        When True, display detailed claim evidence (source sentences,
        present/absent outlets) for manual inspection.
    verbose:
        When True, display engineering diagnostics (centroid feature values,
        claim similarity scores, exact ratio denominators, and all actors).
    console:
        Optional Rich console instance.  Creates a new stdout console if omitted.
    """
    c = console or Console()
    total_articles = len(analysis.article_ids)

    # ── 1. Compact Header ──────────────────────────────────────────────────
    header_content = (
        f"[bold]Event:[/bold] [cyan]{escape(analysis.event_id)}[/cyan]   •   "
        f"[bold]Articles:[/bold] {total_articles}   •   "
        f"[bold]Claim clusters:[/bold] {len(analysis.claim_clusters)}   •   "
        f"[bold]Framing clusters:[/bold] {len(analysis.framing_clusters)}"
    )
    c.print(
        Panel(
            header_content,
            title="[bold white]news-deframe · Event Analysis[/bold white]",
            border_style="bright_blue",
            expand=True,
        )
    )

    # ── 2. Integrated Claim Coverage ───────────────────────────────────────
    c.rule("[bold]Claim Coverage[/bold]")
    claims = analysis.claim_clusters
    if not claims:
        c.print("[dim]No claim clusters found.[/dim]\n")
    else:
        # Compact summary counts above table
        shared_all = sum(1 for cl in claims if cl.coverage_count == total_articles)
        shared_majority = sum(
            1
            for cl in claims
            if cl.coverage_count < total_articles
            and cl.coverage_ratio >= 0.5
            and cl.coverage_count > 1
        )
        single_outlet = sum(1 for cl in claims if cl.coverage_count == 1)

        c.print(f"Shared by all outlets:     [bold]{shared_all}[/bold]")
        c.print(f"Shared by majority:        [bold]{shared_majority}[/bold]")
        c.print(f"Single-outlet claims:      [bold]{single_outlet}[/bold]\n")

        # Build lookup for consensus metadata
        consensus_map: dict[str, ClaimConsensus] = {
            cc.cluster_id: cc for cc in analysis.consensus_view.claims
        }

        cov_table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        cov_table.add_column("Claim", style="bold", width=7)
        cov_table.add_column("Coverage", width=10, justify="right")
        cov_table.add_column("Category", width=20)
        cov_table.add_column("Missing", width=16)
        cov_table.add_column("Representative", no_wrap=False)

        # Deterministic order: coverage descending, then cluster_id
        sorted_claims = sorted(
            claims, key=lambda cl: (-cl.coverage_count, cl.cluster_id)
        )

        for cl in sorted_claims:
            con = consensus_map.get(cl.cluster_id)
            cat_display = _display_category(con, cl.coverage_count)
            cat_style = _CATEGORY_STYLE.get(cat_display, "")
            ratio_text = f"{cl.coverage_count}/{cl.total_articles}"
            missing_str = (
                ", ".join(con.outlets_absent) if con and con.outlets_absent else "—"
            )
            rep_clean = cl.representative.strip()

            cov_table.add_row(
                cl.cluster_id,
                ratio_text,
                Text(cat_display, style=cat_style),
                missing_str,
                rep_clean,
            )

        c.print(cov_table)
        c.print("")

    # ── 3. Actor Framing by Outlet ─────────────────────────────────────────
    c.rule("[bold]Actor Framing by Outlet[/bold]")
    matrix = analysis.entity_outlet_matrix
    if not matrix.entity_names or not matrix.article_ids:
        c.print("[dim]No actor framing data.[/dim]\n")
    else:
        c.print(
            "[dim]Ratios are calculated from occurrences with an identifiable agent/patient role.[/dim]"
        )
        if verbose:
            c.print(
                "[dim]Exact denominator: role_occurrence_count = agent_count + patient_count (passive_patient_ratio denom = patient_count).[/dim]"
            )
        c.print("")

        # Map (entity, article) -> profile
        profile_map = {
            (p.entity_name, p.article_id): p for p in matrix.profiles
        }

        # Actor limit for default mode (uses existing importance ranking)
        default_actor_limit = 5
        if not verbose and len(matrix.entity_names) > default_actor_limit:
            actors_to_display = matrix.entity_names[:default_actor_limit]
            c.print(
                f"[dim]Showing top {default_actor_limit} of {len(matrix.entity_names)} actors "
                f"(ranked by cross-outlet importance). Use --verbose to view all.[/dim]\n"
            )
        else:
            actors_to_display = matrix.entity_names

        for entity_name in actors_to_display:
            c.print(f"[bold]Actor:[/bold] [bold yellow]{escape(entity_name)}[/bold yellow]")

            act_table = Table(
                box=box.SIMPLE,
                show_header=True,
                header_style="bold cyan",
                expand=False,
                pad_edge=False,
            )
            act_table.add_column("Outlet", style="bold", width=12)
            act_table.add_column("Agent", justify="right", width=10)
            act_table.add_column("Patient", justify="right", width=10)
            act_table.add_column("Role observations", justify="right", width=20)
            if verbose:
                act_table.add_column("Passive Pt", justify="right", width=14)

            actions_lines: list[str] = []
            modifiers_lines: list[str] = []

            for aid in matrix.article_ids:
                profile = profile_map.get((entity_name, aid))
                if profile and profile.total_mentions > 0:
                    ag_str = f"{profile.agent_ratio:.2f}"
                    pt_str = f"{profile.patient_ratio:.2f}"
                    obs_str = str(profile.total_mentions)
                    if verbose:
                        pass_str = f"{profile.passive_ratio:.2f} ({profile.passive_count})"
                        act_table.add_row(aid, ag_str, pt_str, obs_str, pass_str)
                    else:
                        act_table.add_row(aid, ag_str, pt_str, obs_str)

                    if profile.associated_verbs:
                        verbs_str = ", ".join(profile.associated_verbs)
                        actions_lines.append(f"  [bold]{escape(aid)}:[/bold] {escape(verbs_str)}")
                    if profile.modifiers:
                        mods_str = ", ".join(profile.modifiers)
                        modifiers_lines.append(f"  [bold]{escape(aid)}:[/bold] {escape(mods_str)}")
                else:
                    if verbose:
                        act_table.add_row(aid, "—", "—", "0", "—")
                    else:
                        act_table.add_row(aid, "—", "—", "0")

            c.print(act_table)
            if actions_lines:
                c.print("[bold dim]Associated actions:[/bold dim]")
                for aline in actions_lines:
                    c.print(aline)
            if verbose and modifiers_lines:
                c.print("[bold dim]Evaluative modifiers:[/bold dim]")
                for mline in modifiers_lines:
                    c.print(mline)
            c.print("")

    # ── 4. Framing Clusters ────────────────────────────────────────────────
    c.rule("[bold]Framing Clusters[/bold]")
    if not analysis.framing_clusters:
        c.print("[dim]No framing clusters found.[/dim]\n")
    else:
        # Small corpus notice if every article forms its own cluster
        if len(analysis.framing_clusters) == total_articles and all(
            len(fc.article_ids) == 1 for fc in analysis.framing_clusters
        ):
            c.print(
                "[dim]Note: Each article forms a separate structural cluster in this corpus.[/dim]"
            )

        for fc in analysis.framing_clusters:
            members = ", ".join(fc.article_ids)
            if verbose:
                centroid_str = "  ".join(
                    f"{k}: {v:.2f}" for k, v in fc.centroid_description.items()
                )
                c.print(
                    Panel(
                        f"[bold]Members:[/bold] {escape(members)}\n"
                        f"[dim]Centroid — {centroid_str}[/dim]",
                        title=f"[bold cyan]{escape(fc.label)}[/bold cyan]",
                        border_style="dim",
                    )
                )
            else:
                c.print(f"  [bold cyan]• {escape(fc.label)}:[/bold cyan] {escape(members)}")
        c.print("")

    # ── 5. Detailed Claims (--details) ─────────────────────────────────────
    if details:
        c.rule("[bold]Detailed Claim Evidence[/bold]")
        if not claims:
            c.print("[dim]No detailed claim data.[/dim]\n")
        else:
            consensus_map = {
                cc.cluster_id: cc for cc in analysis.consensus_view.claims
            }
            for cluster in claims:
                con = consensus_map.get(cluster.cluster_id)
                present_str = (
                    ", ".join(con.outlets_present)
                    if (con and con.outlets_present)
                    else (", ".join(cluster.article_ids) or "—")
                )
                absent_str = (
                    ", ".join(con.outlets_absent)
                    if (con and con.outlets_absent)
                    else "—"
                )

                sources_lines: list[str] = []
                for src in cluster.sources:
                    tag = escape(f"[{src.article_id}]")
                    src_txt = escape(src.text)
                    if verbose:
                        sources_lines.append(
                            f"  • {tag} (sim={src.similarity:.2f}) {src_txt}"
                        )
                    else:
                        sources_lines.append(
                            f"  • {tag} {src_txt}"
                        )
                sources_str = "\n".join(sources_lines)

                panel_body = (
                    f"[bold]Representative:[/bold]\n  {escape(cluster.representative)}\n\n"
                    f"[bold]Coverage:[/bold] {cluster.coverage_count}/{cluster.total_articles}  "
                    f"({cluster.coverage_ratio:.0%})\n"
                    f"[bold]Present outlets:[/bold] {escape(present_str)}\n"
                    f"[bold]Absent outlets:[/bold]  {escape(absent_str)}\n\n"
                    f"[bold]Source sentences:[/bold]\n{sources_str}"
                )
                c.print(
                    Panel(
                        panel_body,
                        title=f"[bold cyan]{escape(cluster.cluster_id)}[/bold cyan]",
                        border_style="blue",
                    )
                )
            c.print("")

    # ── 6. Research Interpretation Notes ───────────────────────────────────
    c.rule("[bold]Research Interpretation Notes[/bold]")
    c.print(
        "[dim]"
        "• Claim coverage reflects reporting frequency across outlets, not factual verification.\n"
        "• Absence of a claim indicates a reporting difference, not intentional omission.\n"
        "• Agent/patient ratios describe grammatical positioning in extracted clauses.\n"
        "• Framing clusters group articles by structural feature similarity only."
        "[/dim]\n"
    )
