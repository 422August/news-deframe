"""Deterministic NLP evaluation metrics calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Any


@dataclass
class ClassificationMetrics:
    """Standard classification evaluation metrics."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    support: int
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0


@dataclass
class ClusteringMetrics:
    """Clustering evaluation metrics (Pairwise Precision, Recall, F1, Rand Index)."""

    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    rand_index: float
    gold_cluster_count: int
    predicted_cluster_count: int


def calculate_binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> ClassificationMetrics:
    """Calculate binary classification precision, recall, f1, and accuracy."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)

    total = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 1.0

    return ClassificationMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        support=total,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def calculate_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix: matrix[true_label][pred_label] = count."""
    matrix = {tl: {pl: 0 for pl in labels} for tl in labels}
    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1
    return matrix


def calculate_clustering_metrics(
    all_items: list[str],
    gold_clusters: list[list[str]],
    predicted_clusters: list[list[str]],
) -> ClusteringMetrics:
    """Calculate pairwise precision, recall, F1, and Rand index for clustering."""
    # Build pair sets
    gold_pairs = set()
    for g_clust in gold_clusters:
        for i in range(len(g_clust)):
            for j in range(i + 1, len(g_clust)):
                item_a, item_b = sorted([g_clust[i], g_clust[j]])
                gold_pairs.add((item_a, item_b))

    pred_pairs = set()
    for p_clust in predicted_clusters:
        for i in range(len(p_clust)):
            for j in range(i + 1, len(p_clust)):
                item_a, item_b = sorted([p_clust[i], p_clust[j]])
                pred_pairs.add((item_a, item_b))

    tp = len(gold_pairs & pred_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    n_items = len(all_items)
    total_possible_pairs = n_items * (n_items - 1) // 2
    tn = total_possible_pairs - (tp + fp + fn)

    p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gold_pairs and not pred_pairs else 0.0)
    r = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not gold_pairs else 0.0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    rand_idx = (tp + tn) / total_possible_pairs if total_possible_pairs > 0 else 1.0

    return ClusteringMetrics(
        pairwise_precision=round(p, 4),
        pairwise_recall=round(r, 4),
        pairwise_f1=round(f1, 4),
        rand_index=round(rand_idx, 4),
        gold_cluster_count=len(gold_clusters),
        predicted_cluster_count=len(predicted_clusters),
    )
