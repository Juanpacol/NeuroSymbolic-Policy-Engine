"""Tests for nspe.eval.metrics: AUROC, threshold fitting, binary metrics."""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.eval.metrics import auroc, best_threshold, binary_metrics


class TestAuroc(TestCase):
    def test_perfect_inverted_and_tied(self):
        labels = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(
            auroc(torch.tensor([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), labels), 1.0
        )
        self.assertEqual(
            auroc(torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), labels), 0.0
        )
        self.assertEqual(auroc(torch.full((6,), 0.5), labels), 0.5)

    def test_hand_computed_case(self):
        # positives at scores 0.4 and 0.8; negatives at 0.1, 0.5, 0.9.
        # Pairs won by positives: 0.4 beats 0.1 -> 1; 0.8 beats 0.1, 0.5 -> 2.
        # 3 of 6 pairs => 0.5.
        scores = torch.tensor([0.1, 0.4, 0.5, 0.8, 0.9])
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        self.assertEqual(auroc(scores, labels), 0.5)

    def test_ties_get_half_credit(self):
        # One positive and one negative sharing a score is half a win.
        scores = torch.tensor([0.5, 0.5])
        labels = torch.tensor([1.0, 0.0])
        self.assertEqual(auroc(scores, labels), 0.5)

    def test_single_class_returns_chance(self):
        self.assertEqual(auroc(torch.rand(10), torch.zeros(10)), 0.5)
        self.assertEqual(auroc(torch.rand(10), torch.ones(10)), 0.5)

    def test_invariant_under_monotone_rescaling(self):
        torch.manual_seed(0)
        scores = torch.rand(64)
        labels = (torch.rand(64) > 0.5).float()
        rescaled = torch.sigmoid(3.0 * torch.logit(scores.clamp(1e-6, 1 - 1e-6)) - 1.5)
        self.assertEqual(auroc(scores, labels), auroc(rescaled, labels))


class TestBestThreshold(TestCase):
    def test_recovers_separable_split(self):
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        threshold, score = best_threshold(scores, labels, objective="f1")
        self.assertEqual(score, 1.0)
        # Predicting scores >= threshold must select exactly the positives.
        self.assertEqual((scores >= threshold).float(), labels)

    def test_accuracy_objective(self):
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        _, score = best_threshold(scores, labels, objective="accuracy")
        self.assertEqual(score, 1.0)

    def test_rejects_unknown_objective(self):
        with self.assertRaises(ValueError):
            best_threshold(torch.rand(4), torch.zeros(4), objective="auroc")


class TestBinaryMetrics(TestCase):
    def test_reports_positive_rate(self):
        # A constant "never positive" model: the degenerate solution the
        # consistency metrics would otherwise score as perfect.
        scores = torch.full((10,), 0.1)
        labels = torch.tensor([1.0] * 4 + [0.0] * 6)
        metrics = binary_metrics(scores, labels, threshold=0.5)

        self.assertEqual(metrics["positive_rate"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["accuracy"], 0.6)

    def test_perfect_separation(self):
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        metrics = binary_metrics(scores, labels, threshold=0.5)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["positive_rate"], 0.5)


if __name__ == "__main__":
    run_tests()
