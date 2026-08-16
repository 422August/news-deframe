"""Coverage diffing: identify unshared/omitted sentences between two articles.

A sentence is considered "unshared" when its best cosine-similarity match in the
other article falls below *threshold* (default 0.60).
"""
from __future__ import annotations

import numpy as np

from news_deframe.diff.aligner import embed_sentences, cosine_similarity_matrix
from news_deframe.schemas import DiffReport, ParsedArticle, SentenceAlignment


def compute_coverage(
    article_a: ParsedArticle,
    article_b: ParsedArticle,
    threshold: float = 0.60,
) -> DiffReport:
    """Produce a full :class:`DiffReport` comparing two parsed articles.

    Steps
    -----
    1.  Embed all sentences from both articles.
    2.  Build an (N × M) cosine similarity matrix.
    3.  For each sentence in A, find its best match in B (and vice versa).
    4.  Sentences whose best-match score < *threshold* are "unshared claims".

    Parameters
    ----------
    article_a, article_b:
        Fully parsed articles (output of the parser pipeline).
    threshold:
        Similarity cutoff below which a sentence is considered unshared.

    Returns
    -------
    DiffReport
    """
    sents_a = article_a.sentences
    sents_b = article_b.sentences

    # --- Passive ratios ---
    total_a = len(article_a.svo_records)
    passive_a = sum(1 for r in article_a.svo_records if r.is_passive)
    passive_ratio_a = passive_a / total_a if total_a else 0.0

    total_b = len(article_b.svo_records)
    passive_b = sum(1 for r in article_b.svo_records if r.is_passive)
    passive_ratio_b = passive_b / total_b if total_b else 0.0

    # --- Edge cases ---
    if not sents_a or not sents_b:
        return DiffReport(
            article_a_id=article_a.article_id,
            article_b_id=article_b.article_id,
            alignments=[
                SentenceAlignment(sent_a=s, sent_b=None, similarity_score=0.0)
                for s in sents_a
            ],
            unshared_claims_a=list(sents_a),
            unshared_claims_b=list(sents_b),
            passive_ratio_a=passive_ratio_a,
            passive_ratio_b=passive_ratio_b,
        )

    emb_a = embed_sentences(sents_a)
    emb_b = embed_sentences(sents_b)
    sim: np.ndarray = cosine_similarity_matrix(emb_a, emb_b)  # shape (N, M)

    # --- Alignments: A → B ---
    alignments: list[SentenceAlignment] = []
    for i, sent_a in enumerate(sents_a):
        best_j = int(np.argmax(sim[i]))
        best_score = float(sim[i, best_j])
        alignments.append(
            SentenceAlignment(
                sent_a=sent_a,
                sent_b=sents_b[best_j] if best_score >= threshold else None,
                similarity_score=round(best_score, 4),
            )
        )

    # --- Unshared claims in A (no good match in B) ---
    unshared_a = [
        sents_a[i]
        for i in range(len(sents_a))
        if float(np.max(sim[i])) < threshold
    ]

    # --- Unshared claims in B (no good match in A) ---
    unshared_b = [
        sents_b[j]
        for j in range(len(sents_b))
        if float(np.max(sim[:, j])) < threshold
    ]

    return DiffReport(
        article_a_id=article_a.article_id,
        article_b_id=article_b.article_id,
        alignments=alignments,
        unshared_claims_a=unshared_a,
        unshared_claims_b=unshared_b,
        passive_ratio_a=round(passive_ratio_a, 4),
        passive_ratio_b=round(passive_ratio_b, 4),
    )
