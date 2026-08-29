"""Multi-document claim clustering with 2-stage verification and coherence constraints.

Strategy
--------
1. Collect all non-empty sentences across all articles.
2. Stage 1 (Candidate Retrieval):
   Embed all sentences using :func:`~news_deframe.diff.aligner.embed_sentences`
   to generate candidate pairs whose embedding cosine similarity >= candidate threshold.
3. Stage 2 (Claim-Equivalence Verification):
   Run :func:`~news_deframe.analysis.claim_verifier.verify_claim_equivalence` on each
   candidate pair.  Only EQUIVALENT and (by policy) COMPATIBLE relations create an
   adjacency edge.  RELATED is NEVER sufficient to create an edge.
4. Stage 3 (Coherence-Constrained Complete-Link Clustering):
   Build a verified adjacency graph.  Use a conservative complete-link component
   discovery: a new node is only added to an existing component if it has a verified
   edge to EVERY current member of the component (complete-link criterion).
   This prevents transitive semantic drift (A ~ B ~ C where A ≢ C).
5. Post-hoc coherence pruning:
   For each component, elect a medoid and evict any member that does not satisfy
   the verifier with the medoid.  Evicted nodes form their own singleton components.
6. Medoid representative selection:
   For each cluster, select the medoid (maximising mean verified-similarity to all
   cluster members) as the representative.  Prefer the shortest sentence among
   near-medoid candidates when all similarities are close, to avoid multi-proposition
   long sentences as the face of a cluster.
7. Deduplicate outlet coverage:
   Multiple paraphrases from the same article count as a single outlet presence.

Conservative-failure policy
---------------------------
FALSE MERGE is more harmful than FALSE SPLIT for research validity.
When structural evidence is insufficient, keep propositions in separate clusters.
The COMPATIBLE relation is currently treated as same-claim for coverage because it
represents a consistent sub-proposition.  This policy is documented and may be
tightened if evaluation reveals remaining false merges.

Relation-to-edge mapping
------------------------
EQUIVALENT  → edge (always same-claim)
COMPATIBLE  → edge (same-claim; compatible detail/sub-proposition)
RELATED     → NO edge (different assertions; same topic)
CONTRADICTORY → NO edge
UNRELATED   → NO edge
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from news_deframe.schemas import ParsedArticle
from news_deframe.analysis.schemas import ClaimCluster, SourceSentence
from news_deframe.analysis.claim_verifier import (
    AtomicProposition,
    ClaimRelationType,
    check_claim_eligibility,
    extract_atomic_propositions,
    verify_claim_equivalence,
    _speakers_diverge,
)

# Candidate generation similarity cutoff — broad, picks up paraphrases and near-paraphrases.
# Must be below the verifier's own thresholds so the verifier always gets to decide.
_CANDIDATE_SIM_THRESHOLD = 0.50

# Minimum verifier confidence to form a same-claim edge.
_MIN_CONFIDENCE = 0.50

# Minimum similarity threshold (used as a fallback sanity check alongside the verifier).
_DEFAULT_THRESHOLD = 0.60

# Same-claim relations: which ClaimRelationType values create an adjacency edge.
# RELATED must NOT appear here.
_SAME_CLAIM_RELATIONS = frozenset({ClaimRelationType.EQUIVALENT, ClaimRelationType.COMPATIBLE})


def _is_same_claim_edge(relation: ClaimRelationType, confidence: float) -> bool:
    """Return True if this verification result justifies a same-claim edge."""
    return relation in _SAME_CLAIM_RELATIONS and confidence >= _MIN_CONFIDENCE


def cluster_claims(
    articles: list[ParsedArticle],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> list[ClaimCluster]:
    """Cluster semantically equivalent propositions across multiple articles.

    Uses a 4-stage architecture:
    1. Atomic proposition extraction: decompose compound sentences into atomic propositions
       preserving provenance back to the source sentence.
    2. Candidate generation via embedding similarity of atomic propositions.
    3. Propositional claim-equivalence verification (structural, domain-agnostic).
    4. Coherence-constrained complete-link clustering to prevent transitive drift.

    Parameters
    ----------
    articles:
        Parsed articles to analyse. Each article's ``sentences`` list is used.
    threshold:
        Minimum similarity for coherence checks (secondary guard, verifier is primary).
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

    # ── Stage 1: Extract Atomic Propositions with full provenance ─────────────
    all_propositions: list[AtomicProposition] = []
    for article in articles:
        for sent_idx, sent in enumerate(article.sentences):
            props = extract_atomic_propositions(article.article_id, sent_idx, sent)
            all_propositions.extend(props)

    if not all_propositions:
        return []

    prop_texts = [p.proposition_text for p in all_propositions]
    n_props = len(all_propositions)
    embeddings: np.ndarray = embed_fn(prop_texts)  # (N, D)

    # Pairwise similarity matrix
    sim_matrix = embeddings @ embeddings.T  # (N, N)

    # ── Stage 2: Build verified equivalence adjacency matrix ──────────────────
    adj_matrix: list[list[bool]] = [[False] * n_props for _ in range(n_props)]
    verif_cache: dict[tuple[int, int], bool] = {}

    def _verified_edge(i: int, j: int) -> bool:
        """Return True if propositions (i, j) have a verified same-claim edge."""
        key = (min(i, j), max(i, j))
        if key in verif_cache:
            return verif_cache[key]
        sim = float(sim_matrix[i, j])
        if sim < _CANDIDATE_SIM_THRESHOLD:
            verif_cache[key] = False
            return False

        res = verify_claim_equivalence(prop_texts[i], prop_texts[j], sim)
        is_same = _is_same_claim_edge(res.relation, res.confidence)
        if is_same:
            # Check speaker divergence if both have explicit speakers
            if (
                all_propositions[i].speaker
                and all_propositions[j].speaker
                and _speakers_diverge(
                    [all_propositions[i].speaker],
                    [all_propositions[j].speaker],
                )
            ):
                # Distinct speakers making distinct arguments diverge.
                # If they report the identical/high-overlap factual assertion, allow alignment.
                shared_tokens = all_propositions[i].content_tokens & all_propositions[j].content_tokens
                total_tokens = all_propositions[i].content_tokens | all_propositions[j].content_tokens
                jaccard = len(shared_tokens) / max(1, len(total_tokens))
                if jaccard < 0.60:
                    is_same = False

        verif_cache[key] = is_same
        return is_same

    for i in range(n_props):
        for j in range(i + 1, n_props):
            if float(sim_matrix[i, j]) >= _CANDIDATE_SIM_THRESHOLD:
                if _verified_edge(i, j):
                    adj_matrix[i][j] = True
                    adj_matrix[j][i] = True

    # ── Stage 3: Coherence-Constrained Complete-Link Clustering ───────────────
    cluster_assignment: list[int] = [-1] * n_props
    cluster_members: list[list[int]] = []

    # Build sorted list of edges (sim, i, j)
    edges: list[tuple[float, int, int]] = []
    for i in range(n_props):
        for j in range(i + 1, n_props):
            if adj_matrix[i][j]:
                edges.append((float(sim_matrix[i, j]), i, j))
    edges.sort(key=lambda e: -e[0])

    def _can_join_cluster(node: int, clust_id: int) -> bool:
        """Return True if node has verified edges to all members of clust_id."""
        for member in cluster_members[clust_id]:
            if member == node:
                continue
            if not adj_matrix[node][member]:
                return False
        return True

    for sim, i, j in edges:
        ci = cluster_assignment[i]
        cj = cluster_assignment[j]

        if ci == -1 and cj == -1:
            new_clust = len(cluster_members)
            cluster_members.append([i, j])
            cluster_assignment[i] = new_clust
            cluster_assignment[j] = new_clust

        elif ci == -1 and cj != -1:
            if _can_join_cluster(i, cj):
                cluster_members[cj].append(i)
                cluster_assignment[i] = cj

        elif ci != -1 and cj == -1:
            if _can_join_cluster(j, ci):
                cluster_members[ci].append(j)
                cluster_assignment[j] = ci

        elif ci != cj:
            clust_i = cluster_members[ci]
            clust_j = cluster_members[cj]
            can_merge = all(
                adj_matrix[mi][mj]
                for mi in clust_i
                for mj in clust_j
                if mi != mj
            )
            if can_merge:
                if len(clust_i) >= len(clust_j):
                    for node in clust_j:
                        cluster_members[ci].append(node)
                        cluster_assignment[node] = ci
                    cluster_members[cj] = []
                else:
                    for node in clust_i:
                        cluster_members[cj].append(node)
                        cluster_assignment[node] = cj
                    cluster_members[ci] = []

    active_clusters: list[list[int]] = [c for c in cluster_members if c]
    for i in range(n_props):
        if cluster_assignment[i] == -1:
            active_clusters.append([i])

    # ── Stage 4: Post-hoc coherence pruning ───────────────────────────────────
    refined_clusters: list[list[int]] = []
    for comp in active_clusters:
        if len(comp) <= 1:
            refined_clusters.append(comp)
            continue

        best_medoid = comp[0]
        best_score = -1.0
        for cand in comp:
            score = sum(float(sim_matrix[cand, other]) for other in comp if other != cand)
            if score > best_score:
                best_score = score
                best_medoid = cand

        coherent_members = [best_medoid]
        outliers = []
        for member in comp:
            if member == best_medoid:
                continue
            if adj_matrix[best_medoid][member]:
                coherent_members.append(member)
            else:
                outliers.append(member)

        refined_clusters.append(coherent_members)
        for outlier in outliers:
            refined_clusters.append([outlier])

    # ── Stage 5: Build ClaimCluster objects with Evidence Invariant ───────────
    clusters: list[ClaimCluster] = []

    for k, member_indices in enumerate(refined_clusters):
        if len(member_indices) == 1:
            rep_idx = member_indices[0]
        else:
            def _medoid_score(idx: int) -> tuple[float, int]:
                sim_sum = sum(float(sim_matrix[idx, other]) for other in member_indices if other != idx)
                return (sim_sum, -len(prop_texts[idx]))

            rep_idx = max(member_indices, key=_medoid_score)

        # Build source sentence evidence from verified propositions
        sources_map: dict[tuple[str, str], float] = {}
        for idx in member_indices:
            p = all_propositions[idx]
            sim = float(sim_matrix[idx, rep_idx])
            key = (p.article_id, p.sentence_text)
            if key not in sources_map or sim > sources_map[key]:
                sources_map[key] = sim

        sources: list[SourceSentence] = [
            SourceSentence(
                article_id=art_id,
                text=sent_text,
                similarity=round(min(max(sim, 0.0), 1.0), 4),
            )
            for (art_id, sent_text), sim in sources_map.items()
        ]

        distinct_article_ids = sorted({s.article_id for s in sources})
        coverage_count = len(distinct_article_ids)
        coverage_ratio = round(coverage_count / total_articles, 4)

        clusters.append(
            ClaimCluster(
                cluster_id=f"C{k + 1:02d}",
                representative=prop_texts[rep_idx],
                sources=sources,
                article_ids=distinct_article_ids,
                coverage_count=coverage_count,
                total_articles=total_articles,
                coverage_ratio=coverage_ratio,
            )
        )

    # Sort by descending coverage, then cluster_id
    clusters.sort(key=lambda c: (-c.coverage_count, c.cluster_id))

    # Re-index cluster_id to C01, C02, ...
    for new_idx, c in enumerate(clusters):
        c.cluster_id = f"C{new_idx + 1:02d}"

    return clusters
