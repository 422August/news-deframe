"""Multi-document claim clustering.

Strategy
--------
1.  Collect all sentences from all articles.
2.  Embed them using the shared :func:`~news_deframe.diff.aligner.embed_sentences`
    function (same model as the ``diff`` pipeline).
3.  Group sentences into clusters with a greedy threshold-based approach:

    - Iterate sentences in document order.
    - If a sentence has cosine similarity \u2265 *threshold* to an existing
      cluster's representative, add it to that cluster.
    - Otherwise open a new cluster with that sentence as the representative.

4.  Deduplicate outlet coverage \u2014 multiple paraphrases from the same article
    count as a single presence for coverage calculation.

The clustering algorithm is intentionally modular: the
:func:`cluster_claims` function accepts a *similarity_fn* parameter so the
algorithm can be swapped in future without changing calling code.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from news_deframe.schemas import ParsedArticle
from news_deframe.analysis.schemas import ClaimCluster, SourceSentence

# Default threshold for claim similarity
_DEFAULT_THRESHOLD = 0.60


def cluster_claims(
    articles: list[ParsedArticle],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> list[ClaimCluster]:
    """Cluster semantically similar sentences across multiple articles.

    Parameters
    ----------
    articles:
        Parsed articles to analyse.  Each article's ``sentences`` list is used.
    threshold:
        Minimum cosine similarity for two sentences to be grouped together.
    embed_fn:
        Optional embedding function override (primarily for testing).  Must
        accept ``list[str]`` and return an ``(N, D)`` float32 numpy array of
        L2-normalised vectors.  Defaults to the shared
        :func:`~news_deframe.diff.aligner.embed_sentences` function.

    Returns
    -------
    list[ClaimCluster]
        Clusters sorted by descending coverage count, then by cluster ID.
        Clusters with zero sentences are not returned.
    """
    if embed_fn is None:
        from news_deframe.diff.aligner import embed_sentences as embed_fn  # type: ignore[assignment]

    total_articles = len(articles)

    # Collect (article_id, sentence) pairs \u2014 flat list
    all_pairs: list[tuple[str, str]] = []
    for article in articles:
        for sent in article.sentences:
            if sent.strip():
                all_pairs.append((article.article_id, sent.strip()))

    if not all_pairs:
        return []

    sentences = [p[1] for p in all_pairs]
    embeddings: np.ndarray = embed_fn(sentences)  # (N, D)

    # Greedy clustering \u2014 representative = first sentence in each cluster
    cluster_reps: list[int] = []          # indices into sentences[]
    cluster_rep_embs: list[np.ndarray] = []
    assignments: list[int] = []           # cluster index for each sentence

    for i, emb in enumerate(embeddings):
        if not cluster_reps:
            cluster_reps.append(i)
            cluster_rep_embs.append(emb)
            assignments.append(0)
            continue

        # Compute cosine similarities to all current representatives
        rep_matrix = np.stack(cluster_rep_embs)         # (K, D)
        sims = rep_matrix @ emb                          # (K,)
        best_k = int(np.argmax(sims))
        best_score = float(sims[best_k])

        if best_score >= threshold:
            assignments.append(best_k)
        else:
            assignments.append(len(cluster_reps))
            cluster_reps.append(i)
            cluster_rep_embs.append(emb)

    # Build ClaimCluster objects
    num_clusters = len(cluster_reps)
    cluster_sources: list[list[SourceSentence]] = [[] for _ in range(num_clusters)]

    for i, (article_id, sent) in enumerate(all_pairs):
        k = assignments[i]
        rep_idx = cluster_reps[k]
        similarity = float(np.dot(embeddings[i], embeddings[rep_idx]))
        cluster_sources[k].append(
            SourceSentence(
                article_id=article_id,
                text=sent,
                similarity=round(min(max(similarity, 0.0), 1.0), 4),
            )
        )

    all_article_ids = [a.article_id for a in articles]

    clusters: list[ClaimCluster] = []
    for k, rep_idx in enumerate(cluster_reps):
        sources = cluster_sources[k]
        if not sources:
            continue

        # Deduplicate: one article counts once regardless of how many sentences
        # from it appear in the cluster.
        seen_articles: dict[str, SourceSentence] = {}
        for src in sources:
            if src.article_id not in seen_articles:
                seen_articles[src.article_id] = src
            else:
                # Keep the source with the higher similarity as representative
                if src.similarity > seen_articles[src.article_id].similarity:
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

    # Sort by descending coverage, then cluster_id for stability
    clusters.sort(key=lambda c: (-c.coverage_count, c.cluster_id))

    return clusters
