"""Deterministic NLP Quality Evaluation Framework for news-deframe."""

from news_deframe.evaluation.evaluator import (
    run_evaluation,
    EvaluationReport,
    FalseMergeMetrics,
    evaluate_false_merges,
)

__all__ = [
    "run_evaluation",
    "EvaluationReport",
    "FalseMergeMetrics",
    "evaluate_false_merges",
]
