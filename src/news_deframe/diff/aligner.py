"""Sentence embedding and cosine similarity matrix construction.

Uses ``sentence-transformers`` with the ``paraphrase-multilingual-MiniLM-L12-v2``
model, which supports Chinese and >50 other languages without any preprocessing.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from news_deframe.schemas import SentenceAlignment

_EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_lock = threading.Lock()
_model: Optional[Any] = None  # SentenceTransformer, imported lazily


def _get_model() -> "Any":
    """Return a cached SentenceTransformer instance (thread-safe lazy load)."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer  # lazy import
        _model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _model


def embed_sentences(sentences: list[str]) -> NDArray[np.float32]:
    """Encode *sentences* into L2-normalised embedding vectors.

    Parameters
    ----------
    sentences:
        Non-empty list of strings to embed.

    Returns
    -------
    NDArray[np.float32]
        Shape ``(len(sentences), embedding_dim)``.
    """
    if not sentences:
        raise ValueError("Cannot embed an empty list of sentences.")
    model = _get_model()
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def cosine_similarity_matrix(
    embeddings_a: NDArray[np.float32],
    embeddings_b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute pairwise cosine similarity between two sets of normalised embeddings.

    Because both matrices are L2-normalised, cosine similarity equals the dot product.

    Parameters
    ----------
    embeddings_a:
        Shape ``(N, D)``
    embeddings_b:
        Shape ``(M, D)``

    Returns
    -------
    NDArray[np.float32]
        Shape ``(N, M)`` – entry ``[i, j]`` is the similarity of sentence i in A
        with sentence j in B.
    """
    return np.dot(embeddings_a, embeddings_b.T)


def align_sentences(
    sentences_a: list[str],
    sentences_b: list[str],
    threshold: float = 0.60,
) -> list[SentenceAlignment]:
    """Find the best-matching sentence in B for each sentence in A.

    Parameters
    ----------
    sentences_a, sentences_b:
        Tokenised sentences from each article.
    threshold:
        Minimum cosine similarity to consider a match.

    Returns
    -------
    list[SentenceAlignment]
        One entry per sentence in A.  ``sent_b`` is ``None`` when the best
        available score is below *threshold*.
    """
    if not sentences_a or not sentences_b:
        return [
            SentenceAlignment(sent_a=s, sent_b=None, similarity_score=0.0)
            for s in sentences_a
        ]

    emb_a = embed_sentences(sentences_a)
    emb_b = embed_sentences(sentences_b)
    sim_matrix = cosine_similarity_matrix(emb_a, emb_b)

    alignments: list[SentenceAlignment] = []
    for i, sent_a in enumerate(sentences_a):
        best_j = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i, best_j])
        alignments.append(
            SentenceAlignment(
                sent_a=sent_a,
                sent_b=sentences_b[best_j] if best_score >= threshold else None,
                similarity_score=round(best_score, 4),
            )
        )

    return alignments
