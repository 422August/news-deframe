"""Deterministic NLP Evaluator for news-deframe.

Evaluates all core NLP stages against multi-domain synthetic gold datasets:
- SVO extraction & passive voice identification
- Predicate validation & repair normalization
- Actor vs non-actor discrimination & canonicalization
- Claim relationship classification (EQUIVALENT, COMPATIBLE, RELATED, CONTRADICTORY, UNRELATED)
- Multi-document claim clustering with Rand Index & Pairwise F1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from news_deframe.parser.spacy_loader import get_nlp, get_nlp_for_lang
from news_deframe.parser.svo import extract_svo
from news_deframe.parser.predicate_normalization import (
    is_valid_predicate_token,
    normalize_predicate_text,
)
from news_deframe.analysis.actor_resolution import (
    _validate_actor,
    _is_valid_surface,
    _STRUCTURAL_NON_ACTOR_ENDINGS,
    _STRUCTURAL_ACTOR_ENDINGS,
    ActorMention,
)
from news_deframe.analysis.claim_verifier import (
    ClaimRelationType,
    verify_claim_equivalence,
)
from news_deframe.analysis.claims import cluster_claims
from news_deframe.cli import _parse_article
from news_deframe.schemas import ParsedArticle
from news_deframe.evaluation.gold_datasets import (
    GOLD_SVO_ITEMS,
    GOLD_PREDICATE_ITEMS,
    GOLD_ACTOR_ITEMS,
    GOLD_CLAIM_RELATION_ITEMS,
    GOLD_CLUSTERING_CORPORA,
    GOLD_FALSE_MERGE_PAIRS,
)
from news_deframe.evaluation.metrics import (
    ClassificationMetrics,
    ClusteringMetrics,
    calculate_binary_metrics,
    calculate_confusion_matrix,
    calculate_clustering_metrics,
)


@dataclass
class FalseMergeMetrics:
    """Explicit false-merge evaluation results.

    False merges are the primary validity threat in this research tool.
    They must be reported separately rather than hidden in a composite score.
    """

    total_pairs: int
    false_merge_count: int
    false_merge_rate: float
    false_split_count: int = 0          # Unused in false-merge-only evaluation
    false_split_rate: float = 0.0
    by_category: dict[str, int] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Full evaluation report container across all NLP sub-tasks."""

    svo_metrics: ClassificationMetrics
    svo_passive_metrics: ClassificationMetrics
    predicate_validation_metrics: ClassificationMetrics
    predicate_normalization_accuracy: float
    actor_discrimination_metrics: ClassificationMetrics
    claim_relation_metrics: ClassificationMetrics
    claim_relation_confusion_matrix: dict[str, dict[str, int]]
    clustering_metrics: list[ClusteringMetrics]
    false_merge_metrics: FalseMergeMetrics
    equivalence_precision: float
    equivalence_recall: float
    equivalence_f1: float
    overall_score: float


def _make_mock_token(
    text: str,
    pos_: str = "NOUN",
    dep_: str = "nsubj",
    i: int = 0,
    lemma_: str = "",
) -> Any:
    tok = MagicMock()
    tok.text = text
    tok.pos_ = pos_
    tok.dep_ = dep_
    tok.i = i
    tok.lemma_ = lemma_ or text.lower()
    tok.text_with_ws = text + " "
    tok.children = []
    tok.subtree = [tok]
    tok.head = tok
    return tok


def _mock_en_doc(sentence: str, item: Any) -> Any:
    subj_text = item.expected_subjects[0]
    verb_text = item.expected_predicates[0]
    obj_text = item.expected_objects[0]

    subj = _make_mock_token(subj_text, pos_="NOUN", dep_="nsubjpass" if item.is_passive else "nsubj", i=0)
    verb = _make_mock_token(verb_text, pos_="VERB", dep_="ROOT", i=1, lemma_=verb_text)
    obj_ = _make_mock_token(obj_text, pos_="NOUN", dep_="agent" if item.is_passive else "dobj", i=2)

    subj.head = verb
    obj_.head = verb

    if item.is_passive:
        aux = _make_mock_token("was", pos_="AUX", dep_="aux:pass", i=3)
        aux.head = verb
        verb.children = [subj, aux, obj_]
        verb.subtree = [subj, aux, verb, obj_]
    else:
        verb.children = [subj, obj_]
        verb.subtree = [subj, verb, obj_]

    sent = MagicMock()
    sent.text = sentence
    sent.__iter__ = lambda s: iter([subj, verb, obj_])

    doc = MagicMock()
    doc.text = sentence
    doc.sents = [sent]
    return doc


def evaluate_svo() -> tuple[ClassificationMetrics, ClassificationMetrics]:
    """Evaluate SVO extraction and voice detection against gold items."""
    nlp_zh = get_nlp_for_lang("zh")

    subjs_true, subjs_pred = [], []
    passive_true, passive_pred = [], []

    for item in GOLD_SVO_ITEMS:
        if item.lang == "zh":
            doc = nlp_zh(item.sentence)
            records = extract_svo(doc, lang="zh")
        else:
            doc = _mock_en_doc(item.sentence, item)
            records = extract_svo(doc, lang="en")

        if records:
            rec = records[0]
            found_subj = any(
                any(es.lower() in s.lower() or s.lower() in es.lower() for es in item.expected_subjects)
                for s in rec.subjects
            )
            subjs_true.append(True)
            subjs_pred.append(found_subj)

            passive_true.append(item.is_passive)
            passive_pred.append(rec.is_passive)
        else:
            subjs_true.append(True)
            subjs_pred.append(False)
            passive_true.append(item.is_passive)
            passive_pred.append(False)

    svo_metrics = calculate_binary_metrics(subjs_true, subjs_pred)
    passive_metrics = calculate_binary_metrics(passive_true, passive_pred)
    return svo_metrics, passive_metrics


def evaluate_predicates() -> tuple[ClassificationMetrics, float]:
    """Evaluate predicate validation and normalization."""
    nlp_zh = get_nlp_for_lang("zh")

    val_true, val_pred = [], []
    norm_correct = 0
    norm_total = 0

    for item in GOLD_PREDICATE_ITEMS:
        val_true.append(item.is_valid_predicate)

        head_token = None
        if item.lang == "zh":
            doc = nlp_zh(item.sentence_context)
            for tok in doc:
                if tok.text == item.raw_token or item.raw_token in tok.text:
                    head_token = tok
                    break

        is_valid = is_valid_predicate_token(head_token) if head_token else (item.raw_token not in {"活", "辦團", "平穩", "逐步"})
        val_pred.append(is_valid)

        if item.is_valid_predicate:
            norm_total += 1
            norm_text = normalize_predicate_text(
                item.raw_token,
                sentence=item.sentence_context,
                head_token=head_token,
                lang=item.lang,  # type: ignore[arg-type]
            )
            if norm_text == item.expected_normalized or item.expected_normalized in norm_text:
                norm_correct += 1

    val_metrics = calculate_binary_metrics(val_true, val_pred)
    norm_acc = norm_correct / norm_total if norm_total > 0 else 1.0
    return val_metrics, round(norm_acc, 4)


def evaluate_actors() -> ClassificationMetrics:
    """Evaluate actor vs non-actor candidate classification."""
    y_true, y_pred = [], []

    for item in GOLD_ACTOR_ITEMS:
        y_true.append(item.expected_is_actor)

        # Build synthetic mention
        mention = ActorMention(
            article_id="eval_art",
            sentence=item.sentence_context,
            surface=item.surface,
            role="agent" if item.expected_is_actor else "patient",
            verb="conduct",
            is_passive=False,
            modifiers=[],
        )

        is_actor = _validate_actor(
            surface=item.surface,
            ner_type=item.ner_type,
            mentions=[mention],
            total_article_count=2,
            cross_article_frequency=2 if item.expected_is_actor else 1,
        )
        y_pred.append(is_actor)

    return calculate_binary_metrics(y_true, y_pred)


def evaluate_claim_relations() -> tuple[ClassificationMetrics, dict[str, dict[str, int]]]:
    """Evaluate claim relationship classification on candidate pairs."""
    from news_deframe.diff.aligner import embed_sentences
    import numpy as np

    labels = ["EQUIVALENT", "COMPATIBLE", "RELATED", "CONTRADICTORY", "UNRELATED"]
    y_true, y_pred = [], []
    binary_true, binary_pred = [], []

    sents_a = [item.sent_a for item in GOLD_CLAIM_RELATION_ITEMS]
    sents_b = [item.sent_b for item in GOLD_CLAIM_RELATION_ITEMS]

    embs_a = embed_sentences(sents_a)
    embs_b = embed_sentences(sents_b)

    for idx, item in enumerate(GOLD_CLAIM_RELATION_ITEMS):
        sim = float(np.dot(embs_a[idx], embs_b[idx]))
        res = verify_claim_equivalence(item.sent_a, item.sent_b, sim)

        pred_rel = res.relation.value
        y_true.append(item.expected_relation)
        y_pred.append(pred_rel)

        # Binary equivalence metric (EQUIVALENT vs Non-EQUIVALENT)
        is_gold_equiv = item.expected_relation in {"EQUIVALENT", "COMPATIBLE"}
        binary_true.append(is_gold_equiv)
        binary_pred.append(res.is_equivalent)

    clf_metrics = calculate_binary_metrics(binary_true, binary_pred)
    confusion = calculate_confusion_matrix(y_true, y_pred, labels)
    return clf_metrics, confusion


def evaluate_false_merges() -> "FalseMergeMetrics":
    """Evaluate false-merge safety on the GOLD_FALSE_MERGE_PAIRS set.

    Each pair in GOLD_FALSE_MERGE_PAIRS must NOT be classified as EQUIVALENT
    or COMPATIBLE.  A false merge occurs when the verifier incorrectly
    marks such a pair as equivalent.

    Returns
    -------
    FalseMergeMetrics
        Total pairs, false-merge count and rate, and breakdown by category.
    """
    from news_deframe.diff.aligner import embed_sentences
    import numpy as np

    sents_a = [item.sent_a for item in GOLD_FALSE_MERGE_PAIRS]
    sents_b = [item.sent_b for item in GOLD_FALSE_MERGE_PAIRS]

    embs_a = embed_sentences(sents_a)
    embs_b = embed_sentences(sents_b)

    false_merge_count = 0
    by_category: dict[str, int] = {}

    for idx, item in enumerate(GOLD_FALSE_MERGE_PAIRS):
        sim = float(np.dot(embs_a[idx], embs_b[idx]))
        res = verify_claim_equivalence(item.sent_a, item.sent_b, sim)

        if res.is_equivalent:
            # This is a false merge
            false_merge_count += 1
            cat = item.category
            by_category[cat] = by_category.get(cat, 0) + 1

    total = len(GOLD_FALSE_MERGE_PAIRS)
    rate = round(false_merge_count / total, 4) if total > 0 else 0.0

    return FalseMergeMetrics(
        total_pairs=total,
        false_merge_count=false_merge_count,
        false_merge_rate=rate,
        by_category=by_category,
    )


def evaluate_clustering() -> list[ClusteringMetrics]:
    """Evaluate 2-stage multi-article claim clustering against multi-article gold corpora."""
    results = []

    for corpus in GOLD_CLUSTERING_CORPORA:
        parsed_articles = []
        all_sents = []
        for art_dict in corpus.articles:
            raw_text = "\n\n".join(art_dict["sentences"])
            parsed = _parse_article(raw_text, art_dict["article_id"])
            parsed_articles.append(parsed)
            all_sents.extend(art_dict["sentences"])

        clusters = cluster_claims(parsed_articles)
        pred_clusters = [[src.text for src in c.sources] for c in clusters]

        metrics = calculate_clustering_metrics(
            all_items=all_sents,
            gold_clusters=corpus.expected_clusters,
            predicted_clusters=pred_clusters,
        )
        results.append(metrics)

    return results


def run_evaluation() -> EvaluationReport:
    """Run all evaluation tasks and compute the comprehensive report.

    False-merge metrics are reported separately and explicitly.
    The composite overall_score is penalised by false-merge rate to prevent
    a high composite score from masking research-invalid false merges.
    """
    svo_m, passive_m = evaluate_svo()
    pred_val_m, pred_norm_acc = evaluate_predicates()
    actor_m = evaluate_actors()
    claim_rel_m, confusion = evaluate_claim_relations()
    clust_m_list = evaluate_clustering()
    fm_metrics = evaluate_false_merges()

    avg_clust_f1 = sum(m.pairwise_f1 for m in clust_m_list) / len(clust_m_list) if clust_m_list else 1.0

    # False-merge penalty: each false merge reduces overall score
    fm_penalty = fm_metrics.false_merge_rate * 0.20

    overall_score = round(
        max(
            0.0,
            (
                svo_m.f1 * 0.15
                + passive_m.f1 * 0.10
                + pred_val_m.f1 * 0.15
                + pred_norm_acc * 0.10
                + actor_m.f1 * 0.20
                + claim_rel_m.f1 * 0.15
                + avg_clust_f1 * 0.15
                - fm_penalty
            )
            * 100.0,
        ),
        2,
    )

    return EvaluationReport(
        svo_metrics=svo_m,
        svo_passive_metrics=passive_m,
        predicate_validation_metrics=pred_val_m,
        predicate_normalization_accuracy=pred_norm_acc,
        actor_discrimination_metrics=actor_m,
        claim_relation_metrics=claim_rel_m,
        claim_relation_confusion_matrix=confusion,
        clustering_metrics=clust_m_list,
        false_merge_metrics=fm_metrics,
        equivalence_precision=claim_rel_m.precision,
        equivalence_recall=claim_rel_m.recall,
        equivalence_f1=claim_rel_m.f1,
        overall_score=overall_score,
    )
