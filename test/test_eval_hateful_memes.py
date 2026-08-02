"""Tests for nspe.eval.hateful_memes's core H1/H3 metric functions.

Hand-built tensors, no network/CLIP/checkpoints -- these are what catch
the "grouped by different mu0" bug class if it's ever reintroduced.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.eval.hateful_memes import compute_h1, compute_h3


class TestComputeH1(TestCase):
    def test_uses_same_mu0_for_both_models(self):
        # Two predicate-equivalence classes of 2 cases each (by mu0
        # thresholded at 0.5): class A = [1,0], class B = [0,1].
        mu0 = torch.tensor(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.2, 0.8],
            ]
        )
        # Reasoner agrees within each class (perfectly consistent).
        reasoner_verdict = torch.tensor([0.9, 0.9, 0.1, 0.1])
        # Baseline disagrees within each class (perfectly inconsistent).
        baseline_verdict = torch.tensor([0.9, 0.1, 0.9, 0.1])

        result = compute_h1(mu0, reasoner_verdict, baseline_verdict)

        self.assertEqual(result["reasoner"]["num_classes"], 2)
        self.assertEqual(result["baseline"]["num_classes"], 2)
        self.assertAlmostEqual(result["reasoner"]["inconsistency_rate"], 0.0)
        self.assertAlmostEqual(result["baseline"]["inconsistency_rate"], 1.0)
        self.assertTrue(len(result["worst_classes_baseline"]) > 0)

    def test_reasoner_more_consistent_than_baseline_reflected_in_purity(self):
        mu0 = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        reasoner_verdict = torch.tensor([0.9, 0.9, 0.9, 0.9])
        baseline_verdict = torch.tensor([0.9, 0.1, 0.9, 0.1])

        result = compute_h1(mu0, reasoner_verdict, baseline_verdict)

        self.assertGreater(result["reasoner"]["purity"], result["baseline"]["purity"])


class TestComputeH3(TestCase):
    def test_perfect_reasoner_vs_imperfect_baseline(self):
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        reasoner_verdict = torch.tensor([0.9, 0.9, 0.1, 0.1])  # all correct
        baseline_verdict = torch.tensor([0.9, 0.1, 0.9, 0.1])  # half correct

        result = compute_h3(reasoner_verdict, baseline_verdict, labels)

        self.assertAlmostEqual(result["reasoner"]["accuracy"], 1.0)
        self.assertAlmostEqual(result["baseline"]["accuracy"], 0.5)
        self.assertAlmostEqual(result["accuracy_gap"], 0.5)

    def test_no_significance_threshold_invented(self):
        labels = torch.tensor([1.0, 0.0])
        verdict = torch.tensor([0.9, 0.1])
        result = compute_h3(verdict, verdict, labels)
        self.assertNotIn("significant", result)
        self.assertAlmostEqual(result["accuracy_gap"], 0.0)

    def test_reports_auroc_and_majority_class_reference(self):
        labels = torch.tensor([1.0] * 3 + [0.0] * 7)
        result = compute_h3(torch.rand(10), torch.rand(10), labels)

        self.assertIn("auroc", result["reasoner"])
        self.assertIn("auroc_gap", result)
        self.assertAlmostEqual(result["majority_class_accuracy"], 0.7)

    def test_fitted_threshold_recovers_a_shifted_operating_point(self):
        """Why the threshold is no longer hardcoded to 0.5.

        The calibrator can relocate a model's operating point without
        changing its ranking, so a fixed 0.5 measures calibration rather
        than discrimination.
        """
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        # Ranks perfectly, but every score sits below 0.5.
        compressed = torch.tensor([0.20, 0.19, 0.10, 0.09])

        at_half = compute_h3(compressed, compressed, labels, threshold=0.5)
        fitted = compute_h3(compressed, compressed, labels, threshold=None)

        self.assertEqual(at_half["reasoner"]["accuracy"], 0.5)
        self.assertEqual(at_half["threshold_source"], "provided")
        self.assertEqual(fitted["reasoner"]["accuracy"], 1.0)
        self.assertEqual(fitted["threshold_source"], "fitted")
        # AUROC saw through the operating point either way.
        self.assertEqual(at_half["reasoner"]["auroc"], 1.0)

    def test_each_model_gets_its_own_fitted_threshold(self):
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        result = compute_h3(
            torch.tensor([0.9, 0.8, 0.2, 0.1]),
            torch.tensor([0.09, 0.08, 0.02, 0.01]),
            labels,
        )

        self.assertNotEqual(
            result["reasoner"]["threshold"], result["baseline"]["threshold"]
        )
        self.assertEqual(result["reasoner"]["accuracy"], 1.0)
        self.assertEqual(result["baseline"]["accuracy"], 1.0)


class TestPerArmThresholds(TestCase):
    """A test run must apply each arm the point it fitted on validation.

    The two arms genuinely differ -- seed 0 fitted reasoner 0.0658 and
    baseline 0.1510 -- so a single shared threshold puts at least one of
    them at the wrong operating point.
    """

    def test_pair_applies_a_different_threshold_to_each_arm(self):
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        reasoner = torch.tensor([0.9, 0.8, 0.2, 0.1])
        baseline = torch.tensor([0.09, 0.08, 0.02, 0.01])

        result = compute_h3(reasoner, baseline, labels, threshold=(0.5, 0.05))

        self.assertEqual(result["threshold_source"], "provided_per_arm")
        self.assertEqual(result["reasoner"]["threshold"], 0.5)
        self.assertEqual(result["baseline"]["threshold"], 0.05)
        # Each arm separates its own scale; a shared threshold could not.
        self.assertEqual(result["reasoner"]["accuracy"], 1.0)
        self.assertEqual(result["baseline"]["accuracy"], 1.0)

    def test_a_shared_threshold_would_fail_the_same_case(self):
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        reasoner = torch.tensor([0.9, 0.8, 0.2, 0.1])
        baseline = torch.tensor([0.09, 0.08, 0.02, 0.01])

        shared = compute_h3(reasoner, baseline, labels, threshold=0.5)

        self.assertEqual(shared["threshold_source"], "provided")
        self.assertEqual(shared["baseline"]["accuracy"], 0.5)

    def test_rejects_a_wrong_length_pair(self):
        labels = torch.tensor([1.0, 0.0])
        verdict = torch.tensor([0.9, 0.1])
        with self.assertRaises(ValueError):
            compute_h3(verdict, verdict, labels, threshold=(0.1, 0.2, 0.3))


class TestSingleClassLabelsRefused(TestCase):
    """Silently answering 0.5 on a degenerate split is the worst case.

    This dataset's rows are ordered by label, so a truncated split is
    single-class -- and `auroc` returns exactly 0.5 there, which reads
    as a real chance-level result with an auroc_gap of exactly 0.0.
    """

    def test_all_negative_labels_raise(self):
        verdict = torch.rand(8)
        with self.assertRaisesRegex(ValueError, "single class"):
            compute_h3(verdict, verdict, torch.zeros(8))

    def test_all_positive_labels_raise(self):
        verdict = torch.rand(8)
        with self.assertRaisesRegex(ValueError, "single class"):
            compute_h3(verdict, verdict, torch.ones(8))

    def test_one_of_each_class_is_enough(self):
        verdict = torch.tensor([0.9, 0.1])
        compute_h3(verdict, verdict, torch.tensor([1.0, 0.0]))


class TestDegenerateModelsAreNotRewarded(TestCase):
    def test_constant_reasoner_is_flagged_rather_than_winning_h1(self):
        torch.manual_seed(0)
        mu0 = (torch.rand(200, 3) > 0.5).float()
        constant = torch.full((200,), 0.1)
        discriminating = torch.rand(200)

        h1 = compute_h1(mu0, constant, discriminating)

        # The constant model still wins the raw rate...
        self.assertLess(
            h1["reasoner"]["inconsistency_rate"],
            h1["baseline"]["inconsistency_rate"],
        )
        # ...but is disqualified rather than credited for it.
        self.assertTrue(h1["reasoner"]["degenerate"])
        self.assertFalse(h1["baseline"]["degenerate"])


if __name__ == "__main__":
    run_tests()
