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

        self.assertGreater(
            result["reasoner"]["purity"], result["baseline"]["purity"]
        )


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


if __name__ == "__main__":
    run_tests()
