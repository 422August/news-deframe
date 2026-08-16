"""Diff sub-package public API."""
from news_deframe.diff.aligner import align_sentences, embed_sentences, cosine_similarity_matrix
from news_deframe.diff.coverage import compute_coverage

__all__ = [
    "align_sentences",
    "embed_sentences",
    "cosine_similarity_matrix",
    "compute_coverage",
]
