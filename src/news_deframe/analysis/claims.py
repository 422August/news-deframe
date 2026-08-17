"""Multi-document claim clustering with 2-stage verification and coherence constraints.

Strategy
--------
1. Collect all non-empty sentences across all articles.
2. Stage 1 (Candidate Retrieval):
   Embed all sentences using :func:`~news_deframe.diff.aligner.embed_sentences`
   to generate candidate pairs whose embedding cosine similarity >= candidate threshold.
3. Stage 2 (Claim-Equivalence Verification):
   Run :func:`~news_deframe.analysis.claim_verifier.verify_claim_equivalence` on each
   candidate pair to verify factual equivalence (evaluating agents, patients, predicates,
   modality, negation, attribution, and quantities).
4. Stage 3 (Coherence-Constrained Graph Clustering):
   Build an equivalence graph and discover clusters with complete-link / medoid coherence
   constraints, preventing transitive semantic drift (A ~ B ~ C where A != C).
5. Medoid representative selection:
   For each cluster, select the medoid sentence (maximizing average equivalence/similarity
   to all cluster members) as the representative claim.
6. Deduplicate outlet coverage:
   Multiple paraphrases from the same article count as a single presence for coverage.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from news_deframe.schemas import ParsedArticle
from news_deframe.analysis.schemas import ClaimCluster, SourceSentence
from news_deframe.analysis.claim_verifier import (
    ClaimRelationType,
    verify_claim_equivalence,
)

# Candidate generation similarity cutoff
_CANDIDATE_SIM_THRESHOLD = 0.50
# Default threshold for claim similarity compatibility
_DEFAULT_THRESHOLD = 0.60


def cluster_claims(
    articles: list[ParsedArticle],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> list[ClaimCluster]:
    """Cluster semantically equivalent sentences across multiple articles.

    Uses a 2-stage architecture:
    1. Candidate generation via embedding similarity.
    2. Propositional claim-equivalence verification to reject related-but-distinct claims.
    3. Coherence-constrained graph clustering to avoid transitive semantic drift.

    Parameters
    ----------
    articles:
        Parsed articles to analyse. Each article's ``sentences`` list is used.
    threshold:
        Minimum similarity for candidate retrieval and compatibility.
    embed_fn:
        Optional embedding function override (primarily for testing). Must
        accept ``list[str]`` and return an ``(N, D)`` float32 numpy array of
        L2-normalised vectors.

    Returns
    -------
    list[ClaimCluster]
        Clusters sorted by descending coverage count, then by cluster ID.
    """
    if embed_fn is None:
        from news_deframe.diff.aligner import embed_sentences as embed_fn  # type: ignore[assignment]

    total_articles = len(articles)

    # Collect (article_id, sentence) pairs in document order
    all_pairs: list[tuple[str, str]] = []
    for article in articles:
        for sent in article.sentences:
            s_clean = sent.strip()
            if s_clean:
                all_pairs.append((article.article_id, s_clean))

    if not all_pairs:
        return []

    sentences = [p[1] for p in all_pairs]
    n_sentences = len(sentences)
    embeddings: np.ndarray = embed_fn(sentences)  # (N, D)

    # Pairwise similarity matrix
    sim_matrix = embeddings @ embeddings.T  # (N, N)

    # Stage 2: Build verified equivalence graph
    # Adjacency list: adj[i] = list of (j, similarity, confidence)
    adj: list[list[tuple[int, float, float]]] = [[] for _ in range(n_sentences)]

    for i in range(n_sentences):
        for j in range(i + 1, n_sentences):
            sim = float(sim_matrix[i, j])
            if sim >= _CANDIDATE_SIM_THRESHOLD:
                res = verify_claim_equivalence(sentences[i], sentences[j], sim)
                if res.is_equivalent and res.confidence >= 0.50:
                    adj[i].append((j, sim, res.confidence))
                    adj[j].append((i, sim, res.confidence))

    # Stage 3: Coherence-Constrained Connected Components
    visited = [False] * n_sentences
    raw_components: list[list[int]] = []

    for i in range(n_sentences):
        if visited[i]:
            continue
        # BFS / DFS component
        comp = []
        queue = [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for neighbor, _, _ in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        raw_components.append(comp)

    # Coherence check & medoid splitting:
    # Ensure no member in a cluster has a direct contradiction or unrelated status with the medoid
    refined_clusters: list[list[int]] = []

    for comp in raw_components:
        if len(comp) == 1:
            refined_clusters.append(comp)
            continue

        # Find initial medoid (node with highest sum of similarities to other members)
        best_medoid = comp[0]
        best_score = -1.0
        for cand in comp:
            score = sum(float(sim_matrix[cand, other]) for other in comp)
            if score > best_score:
                best_score = score
                best_medoid = cand

        # Check coherence of each member with the medoid
        coherent_members = []
        outliers = []
        for member in comp:
            if member == best_medoid:
                coherent_members.append(member)
                continue
            sim = float(sim_matrix[best_medoid, member])
            res = verify_claim_equivalence(sentences[best_medoid], sentences[member], sim)
            if res.is_equivalent or (sim >= threshold and res.relation in {ClaimRelationType.EQUIVALENT, ClaimRelationType.COMPATIBLE}):
                coherent_members.append(member)
            else:
                outliers.append(member)

        if coherent_members:
            refined_clusters.append(coherent_members)
        for outlier in outliers:
            refined_clusters.append([outlier])

    # Build ClaimCluster objects
    clusters: list[ClaimCluster] = []

    for k, member_indices in enumerate(refined_clusters):
        # Choose medoid representative
        if len(member_indices) == 1:
            rep_idx = member_indices[0]
        else:
            rep_idx = max(
                member_indices,
                key=lambda idx: sum(float(sim_matrix[idx, other]) for other in member_indices),
            )

        sources: list[SourceSentence] = []
        for idx in member_indices:
            article_id, sent_text = all_pairs[idx]
            sim = float(sim_matrix[idx, rep_idx])
            sources.append(
                SourceSentence(
                    article_id=article_id,
                    text=sent_text,
                    similarity=round(min(max(sim, 0.0), 1.0), 4),
                )
            )

        # Deduplicate coverage by article_id
        seen_articles: dict[str, SourceSentence] = {}
        for src in sources:
            if src.article_id not in seen_articles or src.similarity > seen_articles[src.article_id].similarity:
                seen_articles[src.article_id] = src

        distinct_article_ids = sorted(seen_articles.keys())
        coverage_count = len(distinct_article_ids)
        coverage_ratio = round(coverage_count / total_articles, 4)

        clusters.append(
            ClaimCluster(
                cluster_id=f"C{k + 1:02d}",
                representative=sentences[rep_idx],
                sources=sources,
                article_ids=distinct_article_ids,
                coverage_count=coverage_count,
                total_articles=total_articles,
                coverage_ratio=coverage_ratio,
            )
        )

    # Sort by descending coverage, then cluster_id
    clusters.sort(key=lambda c: (-c.coverage_count, c.cluster_id))

    # Re-index cluster_id to C01, C02, ... after sorting for presentation consistency
    for new_idx, c in enumerate(clusters):
        c.cluster_id = f"C{new_idx + 1:02d}"

    return clusters
