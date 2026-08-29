"""CLI runner for deterministic NLP evaluation framework."""

from __future__ import annotations

import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from news_deframe.evaluation.evaluator import run_evaluation


def main() -> None:
    """Run full NLP evaluation and render rich metrics tables."""
    console = Console()
    console.print("\n[bold cyan]─── Running news-deframe NLP Quality Benchmark ───[/bold cyan]\n")

    report = run_evaluation()

    # Main Metrics Table
    table = Table(title="news-deframe NLP Subsystem Performance", show_header=True, header_style="bold magenta")
    table.add_column("NLP Task / Subsystem", style="bold", width=38)
    table.add_column("Support", justify="right", width=10)
    table.add_column("Precision", justify="right", width=12)
    table.add_column("Recall", justify="right", width=12)
    table.add_column("F1 / Acc", justify="right", width=12)

    table.add_row(
        "SVO Participant Extraction",
        str(report.svo_metrics.support),
        f"{report.svo_metrics.precision:.4f}",
        f"{report.svo_metrics.recall:.4f}",
        f"[green]{report.svo_metrics.f1:.4f}[/green]",
    )
    table.add_row(
        "SVO Voice Detection (Passive)",
        str(report.svo_passive_metrics.support),
        f"{report.svo_passive_metrics.precision:.4f}",
        f"{report.svo_passive_metrics.recall:.4f}",
        f"[green]{report.svo_passive_metrics.f1:.4f}[/green]",
    )
    table.add_row(
        "Predicate Validity Filter",
        str(report.predicate_validation_metrics.support),
        f"{report.predicate_validation_metrics.precision:.4f}",
        f"{report.predicate_validation_metrics.recall:.4f}",
        f"[green]{report.predicate_validation_metrics.f1:.4f}[/green]",
    )
    table.add_row(
        "Predicate Lemma Normalization",
        "-",
        "-",
        "-",
        f"[green]{report.predicate_normalization_accuracy:.4f}[/green]",
    )
    table.add_row(
        "Actor vs Non-Actor Discrimination",
        str(report.actor_discrimination_metrics.support),
        f"{report.actor_discrimination_metrics.precision:.4f}",
        f"{report.actor_discrimination_metrics.recall:.4f}",
        f"[green]{report.actor_discrimination_metrics.f1:.4f}[/green]",
    )
    table.add_row(
        "Claim Equivalence Verification",
        str(report.claim_relation_metrics.support),
        f"{report.claim_relation_metrics.precision:.4f}",
        f"{report.claim_relation_metrics.recall:.4f}",
        f"[green]{report.claim_relation_metrics.f1:.4f}[/green]",
    )

    console.print(table)
    console.print()

    # Multi-Document Claim Clustering Table
    clust_table = Table(title="Multi-Article Event Claim Clustering (Unseen Corpora)", show_header=True, header_style="bold blue")
    clust_table.add_column("Corpus Domain", style="bold", width=35)
    clust_table.add_column("Gold Clusts", justify="right", width=12)
    clust_table.add_column("Pred Clusts", justify="right", width=12)
    clust_table.add_column("Pairwise P", justify="right", width=12)
    clust_table.add_column("Pairwise R", justify="right", width=12)
    clust_table.add_column("Pairwise F1", justify="right", width=12)
    clust_table.add_column("Rand Index", justify="right", width=12)

    domains = ["Semiconductor Fab Incident", "Vaccine Coldchain Breakthrough"]
    for idx, cm in enumerate(report.clustering_metrics):
        dname = domains[idx] if idx < len(domains) else f"Corpus {idx + 1}"
        clust_table.add_row(
            dname,
            str(cm.gold_cluster_count),
            str(cm.predicted_cluster_count),
            f"{cm.pairwise_precision:.4f}",
            f"{cm.pairwise_recall:.4f}",
            f"[green]{cm.pairwise_f1:.4f}[/green]",
            f"[green]{cm.rand_index:.4f}[/green]",
        )

    console.print(clust_table)
    console.print()

    # Claim Relation Confusion Matrix Table
    conf_table = Table(title="Claim Relation Confusion Matrix", show_header=True, header_style="bold yellow")
    conf_table.add_column("Gold \\ Pred", style="bold", width=18)
    labels = ["EQUIVALENT", "COMPATIBLE", "RELATED", "CONTRADICTORY", "UNRELATED"]
    for l in labels:
        conf_table.add_column(l[:8], justify="right", width=10)

    for gold_l in labels:
        row = [gold_l]
        for pred_l in labels:
            cnt = report.claim_relation_confusion_matrix.get(gold_l, {}).get(pred_l, 0)
            if gold_l == pred_l and cnt > 0:
                row.append(f"[bold green]{cnt}[/bold green]")
            elif cnt > 0:
                row.append(f"[bold red]{cnt}[/bold red]")
            else:
                row.append("0")
        conf_table.add_row(*row)

    console.print(conf_table)
    console.print()

    score_color = "bold green" if report.overall_score >= 85.0 else "bold yellow"
    console.print(
        Panel.fit(
            f"Composite NLP Quality Benchmark Score: [{score_color}]{report.overall_score} / 100.0[/{score_color}]",
            border_style="green",
        )
    )
    console.print()


if __name__ == "__main__":
    main()
