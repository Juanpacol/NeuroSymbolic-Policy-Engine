"""Tests for nspe.eval.metrics: AUROC, threshold fitting, binary metrics."""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.eval.metrics import (
    auroc,
    best_threshold,
    binary_metrics,
    calibration_report,
)


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


class TestCalibrationReport(TestCase):
    def test_perfectly_calibrated_scores_have_zero_ece(self):
        # 100 cases at 0.3 with exactly 30 positives, 100 at 0.7 with 70.
        scores = torch.cat([torch.full((100,), 0.3), torch.full((100,), 0.7)])
        labels = torch.cat(
            [torch.ones(30), torch.zeros(70), torch.ones(70), torch.zeros(30)]
        )
        report = calibration_report(scores, labels)

        # Not exactly 0: 0.3 has no exact float32 representation, so the
        # binned score sum misses the label sum by ~1e-8.
        self.assertAlmostEqual(report["ece"], 0.0, places=6)
        self.assertEqual(len(report["bins"]), 2)

    def test_maximally_miscalibrated(self):
        report = calibration_report(torch.ones(50), torch.zeros(50))
        self.assertAlmostEqual(report["ece"], 1.0)
        self.assertAlmostEqual(report["brier"], 1.0)

    def test_brier_matches_hand_computation(self):
        scores = torch.tensor([0.9, 0.1])
        labels = torch.tensor([1.0, 1.0])
        # ((0.9-1)^2 + (0.1-1)^2) / 2 = (0.01 + 0.81) / 2
        self.assertAlmostEqual(calibration_report(scores, labels)["brier"], 0.41)

    def test_bin_counts_sum_to_n(self):
        torch.manual_seed(0)
        report = calibration_report(torch.rand(257), (torch.rand(257) > 0.5).float())
        self.assertEqual(sum(b["count"] for b in report["bins"]), 257)

    def test_score_of_exactly_one_lands_in_the_last_bin(self):
        report = calibration_report(torch.ones(4), torch.ones(4), num_bins=10)
        self.assertEqual(len(report["bins"]), 1)
        self.assertAlmostEqual(report["bins"][0]["lower"], 0.9)

    def test_monotone_recalibration_changes_ece_but_not_auroc(self):
        """Why AUROC was never evidence the calibrator does anything.

        VerdictCalibrator is strictly monotone, so it cannot reorder any
        pair and cannot move AUROC. ECE is what actually sees it.
        """
        from nspe.calibration import VerdictCalibrator

        torch.manual_seed(0)
        raw = 0.1387 + 0.004 * torch.randn(500)
        labels = (torch.rand(500) > 0.6).float()
        calibrator = VerdictCalibrator()
        calibrator.fit_bias_to_base_rate(raw, 0.4)
        calibrated = calibrator(raw).detach()

        self.assertAlmostEqual(auroc(raw, labels), auroc(calibrated, labels), places=9)
        self.assertLess(
            calibration_report(calibrated, labels)["ece"],
            calibration_report(raw, labels)["ece"],
        )


if __name__ == "__main__":
    run_tests()
