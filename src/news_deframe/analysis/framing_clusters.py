"""Unsupervised article framing cluster construction.

Approach
--------
1.  Build a numeric feature vector for each article from available structural
    signals (passive ratio, mean entity agency, claim coverage participation,
    sentence count, entity count).
2.  Normalise the feature matrix to zero mean, unit variance.
3.  Cluster articles using k-means (via ``sklearn``) or a deterministic
    agglomerative fallback when sklearn is unavailable.
4.  Return neutral labels: "Framing Cluster 1", "Framing Cluster 2", \u2026

No ideological labels are inferred.  The centroid description exposes mean
feature values so a human analyst can understand why articles were grouped.

The algorithm is intentionally modular \u2014 the ``cluster_fn`` parameter of
:func:`cluster_articles` can accept any callable with the signature::

    def cluster_fn(feature_matrix: np.ndarray, n_clusters: int) -> list[int]: ...

which maps the ``(n_articles, n_features)`` matrix to a list of integer
cluster assignments (0-indexed).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from news_deframe.schemas import ParsedArticle
from news_deframe.analysis.schemas import (
    ArticleFramingFeatures,
    ClaimCluster,
    EntityOutletMatrix,
    FramingCluster,
)

# Feature names in the order they appear in the feature vector
_FEATURE_NAMES = [
    "passive_ratio",
    "mean_agent_ratio",
    "mean_patient_ratio",
    "entity_count_norm",
    "sentence_count_norm",
]


def _build_feature_vector(
    article: ParsedArticle,
    matrix: EntityOutletMatrix,
    claim_clusters: list[ClaimCluster],
    all_article_ids: list[str],
) -> ArticleFramingFeatures:
    """Extract a numeric framing feature record for *article*."""
    # Passive ratio from SVO records
    svo = article.svo_records
    passive_ratio = (
        sum(1 for r in svo if r.is_passive) / len(svo) if svo else 0.0
    )

    # Mean agent / patient ratios across profiles for this article
    profiles = [p for p in matrix.profiles if p.article_id == article.article_id]
    if profiles:
        mean_agent = sum(p.agent_ratio for p in profiles) / len(profiles)
        mean_patient = sum(p.patient_ratio for p in profiles) / len(profiles)
    else:
        mean_agent = 0.0
        mean_patient = 0.0

    # Claim coverage participation: 1.0 if article appears in cluster, else 0.0
    claim_vec = [
        1.0 if article.article_id in c.article_ids else 0.0
        for c in claim_clusters
    ]

    return ArticleFramingFeatures(
        article_id=article.article_id,
        passive_ratio=round(passive_ratio, 4),
        mean_agent_ratio=round(mean_agent, 4),
        mean_patient_ratio=round(mean_patient, 4),
        entity_count=len({p.entity_name for p in profiles}),
        sentence_count=len(article.sentences),
        claim_coverage_vector=claim_vec,
    )


def _build_feature_matrix(
    features: list[ArticleFramingFeatures],
) -> np.ndarray:
    """Convert feature records into a numeric ``(n_articles, n_features)`` matrix."""
    rows = []
    for f in features:
        # Normalise counts by dividing by a scale factor (max possible value)
        # so they are approximately in [0, 1] without fitting a scaler.
        entity_norm = f.entity_count / max(f.entity_count, 10)
        sent_norm = f.sentence_count / max(f.sentence_count, 30)
        base = [
            f.passive_ratio,
            f.mean_agent_ratio,
            f.mean_patient_ratio,
            entity_norm,
            sent_norm,
        ]
        row = base + f.claim_coverage_vector
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def _kmeans_cluster(feature_matrix: np.ndarray, n_clusters: int) -> list[int]:
    """K-means clustering using sklearn (preferred) with numpy fallback."""
    try:
        from sklearn.cluster import KMeans  # type: ignore[import]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)
        km = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init="auto",
        )
        km.fit(scaled)
        return [int(label) for label in km.labels_]
    except ImportError:
        pass

    # Deterministic fallback: assign based on L2 distance to random-seeded centroids
    rng = np.random.default_rng(42)
    n = len(feature_matrix)
    if n <= n_clusters:
        return list(range(n))

    centroid_indices = rng.choice(n, size=n_clusters, replace=False)
    centroids = feature_matrix[centroid_indices].copy()

    for _ in range(50):  # max iterations
        diffs = feature_matrix[:, None, :] - centroids[None, :, :]  # (n, k, d)
        dists = np.sum(diffs ** 2, axis=-1)                          # (n, k)
        labels = np.argmin(dists, axis=1)                             # (n,)
        new_centroids = np.array(
            [
                feature_matrix[labels == k].mean(axis=0)
                if np.any(labels == k)
                else centroids[k]
                for k in range(n_clusters)
            ],
            dtype=np.float32,
        )
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return [int(label) for label in labels]


def cluster_articles(
    articles: list[ParsedArticle],
    matrix: EntityOutletMatrix,
    claim_clusters: list[ClaimCluster],
    *,
    n_clusters: int | None = None,
    cluster_fn: Callable[[np.ndarray, int], list[int]] | None = None,
) -> list[FramingCluster]:
    """Cluster articles into framing groups based on structural features.

    Parameters
    ----------
    articles:
        Parsed articles in the event corpus.
    matrix:
        Entity \u00d7 outlet matrix (used for agent/patient ratio features).
    claim_clusters:
        Claim clusters (used for coverage participation features).
    n_clusters:
        Number of framing clusters to produce.  Defaults to
        ``min(3, len(articles))``.
    cluster_fn:
        Optional clustering algorithm override with signature
        ``(feature_matrix, n_clusters) -> list[int]``.

    Returns
    -------
    list[FramingCluster]
        Clusters sorted by cluster ID, each with a neutral label.
    """
    n = len(articles)
    if n < 2:
        return []

    k = n_clusters if n_clusters is not None else min(3, n)
    k = max(1, min(k, n))

    all_article_ids = [a.article_id for a in articles]

    feature_records = [
        _build_feature_vector(a, matrix, claim_clusters, all_article_ids)
        for a in articles
    ]
    feature_matrix = _build_feature_matrix(feature_records)

    fn = cluster_fn or _kmeans_cluster
    labels = fn(feature_matrix, k)

    # Build cluster objects
    clusters_map: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters_map.setdefault(label, []).append(i)

    result: list[FramingCluster] = []
    for cluster_idx, (label_int, indices) in enumerate(
        sorted(clusters_map.items()), start=1
    ):
        member_ids = [articles[i].article_id for i in indices]

        # Centroid description: mean of base features for this cluster
        members = feature_matrix[indices]
        n_features = len(_FEATURE_NAMES)
        base = members[:, :n_features].mean(axis=0) if len(members) else np.zeros(n_features)
        centroid_desc = {
            name: round(float(val), 4)
            for name, val in zip(_FEATURE_NAMES, base)
        }

        result.append(
            FramingCluster(
                cluster_id=cluster_idx,
                label=f"Framing Cluster {cluster_idx}",
                article_ids=sorted(member_ids),
                centroid_description=centroid_desc,
            )
        )

    result.sort(key=lambda fc: fc.cluster_id)
    return result
