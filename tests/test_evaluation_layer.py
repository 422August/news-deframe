"""Unit tests for deterministic NLP evaluation layer."""

import pytest
from news_deframe.evaluation.evaluator import (
    evaluate_svo,
    evaluate_predicates,
    evaluate_actors,
    evaluate_claim_relations,
    evaluate_clustering,
    run_evaluation,
)
from news_deframe.evaluation.metrics import (
    calculate_binary_metrics,
    calculate_confusion_matrix,
    calculate_clustering_metrics,
)


class TestEvaluationMetrics:
    def test_binary_metrics_perfect(self):
        m = calculate_binary_metrics([True, False, True], [True, False, True])
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.accuracy == 1.0

    def test_clustering_metrics_perfect(self):
        gold = [["s1", "s2"], ["s3", "s4"]]
        pred = [["s1", "s2"], ["s3", "s4"]]
        m = calculate_clustering_metrics(["s1", "s2", "s3", "s4"], gold, pred)
        assert m.pairwise_precision == 1.0
        assert m.pairwise_recall == 1.0
        assert m.pairwise_f1 == 1.0
        assert m.rand_index == 1.0

    def test_confusion_matrix_structure(self):
        labels = ["A", "B", "C"]
        cm = calculate_confusion_matrix(["A", "B", "A"], ["A", "A", "A"], labels)
        assert cm["A"]["A"] == 2
        assert cm["B"]["A"] == 1
        assert cm["C"]["C"] == 0


class TestEvaluationPipeline:
    def test_evaluate_svo_runs(self):
        svo_m, passive_m = evaluate_svo()
        assert svo_m.precision >= 0.90
        assert passive_m.f1 >= 0.90

    def test_evaluate_predicates_runs(self):
        val_m, norm_acc = evaluate_predicates()
        assert val_m.f1 >= 0.90
        assert norm_acc >= 0.90

    def test_evaluate_actors_runs(self):
        act_m = evaluate_actors()
        assert act_m.f1 >= 0.90

    def test_evaluate_claim_relations_runs(self):
        clf_m, conf = evaluate_claim_relations()
        assert clf_m.f1 >= 0.90
        assert "EQUIVALENT" in conf

    def test_evaluate_clustering_runs(self):
        results = evaluate_clustering()
        assert len(results) >= 2
        for r in results:
            assert r.rand_index >= 0.85

    def test_full_evaluation_report(self):
        report = run_evaluation()
        assert report.overall_score >= 90.0
