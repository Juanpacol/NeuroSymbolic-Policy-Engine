"""Tests for nspe.consistency."""

import math

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.consistency import (
    ConsistencyChecker,
    consistency_loss,
    permutation_null_inconsistency,
)


class TestConsistencyChecker(TestCase):
    def test_zero_inconsistency_when_deterministic(self):
        # Two identical signatures, both cases agree on the verdict.
        mu0 = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.2, 0.8]])
        verdict = torch.tensor([1.0, 1.0, 0.0, 0.0])
        report = ConsistencyChecker(tau=0.5)(mu0, verdict)
        self.assertEqual(report.inconsistency_rate, 0.0)
        self.assertEqual(report.purity, 1.0)
        self.assertEqual(report.num_classes, 2)

    def test_positive_inconsistency_when_noisy(self):
        # Same signature, disagreeing verdicts within the class.
        mu0 = torch.tensor([[0.9, 0.1]] * 4)
        verdict = torch.tensor([1.0, 0.0, 1.0, 0.0])
        report = ConsistencyChecker(tau=0.5)(mu0, verdict)
        self.assertGreater(report.inconsistency_rate, 0.0)
        self.assertLess(report.purity, 1.0)
        self.assertEqual(report.num_classes, 1)

    def test_singleton_classes_contribute_no_pairs(self):
        mu0 = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
        verdict = torch.tensor([1.0, 0.0, 1.0])
        report = ConsistencyChecker(tau=0.5)(mu0, verdict)
        self.assertEqual(report.inconsistency_rate, 0.0)
        self.assertEqual(report.num_classes, 3)

    def test_worst_classes_reports_example_indices(self):
        mu0 = torch.cat(
            [torch.tensor([[0.9, 0.1]] * 4), torch.tensor([[0.1, 0.9]] * 2)]
        )
        verdict = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        report = ConsistencyChecker(tau=0.5)(mu0, verdict)
        worst = report.worst_classes(n=1)
        self.assertEqual(len(worst), 1)
        self.assertEqual(worst[0]["size"], 4)
        self.assertGreater(worst[0]["disagreeing_pairs"], 0)
        self.assertLessEqual(len(worst[0]["example_indices"]), 5)


class TestChanceCorrection(TestCase):
    """The de-confounding: a constant model must not win on consistency."""

    def test_constant_model_is_flagged_degenerate(self):
        torch.manual_seed(0)
        mu0 = (torch.rand(200, 3) > 0.5).float()
        checker = ConsistencyChecker()

        report = checker(mu0, torch.full((200,), 0.1))

        # Perfect on the raw metrics while carrying no information.
        self.assertEqual(report.inconsistency_rate, 0.0)
        self.assertEqual(report.purity, 1.0)
        self.assertEqual(report.positive_rate, 0.0)
        self.assertTrue(report.degenerate)
        self.assertTrue(math.isnan(report.adjusted_consistency))

    def test_a_near_constant_model_beats_a_discriminating_one_on_raw_rate(self):
        """Why the raw rate cannot be compared across models.

        The measured run had the baseline at F1 0.21 -- close enough to
        constant that its better raw consistency was largely an artifact
        of not discriminating.
        """
        torch.manual_seed(0)
        mu0 = (torch.rand(200, 3) > 0.5).float()
        checker = ConsistencyChecker()

        near_constant = torch.full((200,), 0.1)
        near_constant[:5] = 0.9
        discriminating = torch.rand(200)

        self.assertLess(
            checker(mu0, near_constant).inconsistency_rate,
            checker(mu0, discriminating).inconsistency_rate,
        )

    def test_analytic_null_matches_a_hand_computed_case(self):
        # 4 cases, 2 positive: k * (n - k) = 4 disagreeing pairs out of
        # C(4, 2) = 6, so 2/3.
        mu0 = torch.tensor([[0.9, 0.1]] * 4)
        report = ConsistencyChecker()(mu0, torch.tensor([0.9, 0.9, 0.1, 0.1]))

        self.assertAlmostEqual(report.null_inconsistency, 2.0 / 3.0, places=9)
        self.assertEqual(report.positive_rate, 0.5)

    def test_analytic_null_matches_the_permutation_estimate(self):
        torch.manual_seed(0)
        mu0 = (torch.rand(120, 3) > 0.5).float()
        verdict = (torch.rand(120) > 0.65).float()
        checker = ConsistencyChecker()
        report = checker(mu0, verdict)

        empirical = permutation_null_inconsistency(
            report.class_id,
            report.num_classes,
            (verdict >= 0.5).float(),
            num_permutations=2000,
        )
        self.assertLess(abs(report.null_inconsistency - empirical), 0.02)

    def test_perfectly_consistent_model_scores_one(self):
        # Verdict is a function of the signature, so every class is
        # unanimous while both classes are still predicted.
        mu0 = torch.tensor([[0.9, 0.1]] * 4 + [[0.1, 0.9]] * 4)
        verdict = torch.tensor([0.9] * 4 + [0.1] * 4)
        report = ConsistencyChecker()(mu0, verdict)

        self.assertEqual(report.inconsistency_rate, 0.0)
        self.assertFalse(report.degenerate)
        self.assertEqual(report.adjusted_consistency, 1.0)

    def test_worse_than_chance_scores_negative(self):
        # One class, verdicts split exactly -- more disagreement than a
        # marginal-preserving shuffle would produce.
        mu0 = torch.tensor([[0.9, 0.1]] * 4)
        report = ConsistencyChecker()(mu0, torch.tensor([0.9, 0.1, 0.9, 0.1]))

        self.assertLess(report.adjusted_consistency, 0.0)

    def test_signature_entropy_detects_collapse(self):
        collapsed = ConsistencyChecker()(torch.full((64, 3), 0.9), torch.rand(64))
        torch.manual_seed(0)
        diverse = ConsistencyChecker()(
            (torch.rand(64, 3) > 0.5).float(), torch.rand(64)
        )

        self.assertEqual(collapsed.signature_entropy, 0.0)
        self.assertGreater(diverse.signature_entropy, 2.0)

    def test_class_size_histogram_sums_to_class_count(self):
        torch.manual_seed(0)
        report = ConsistencyChecker()((torch.rand(64, 3) > 0.5).float(), torch.rand(64))
        histogram = report.class_size_histogram()

        self.assertEqual(sum(histogram.values()), report.num_classes)
        self.assertEqual(sum(size * count for size, count in histogram.items()), 64)


class TestConsistencyLoss(TestCase):
    def test_zero_when_verdict_constant_within_class(self):
        mu0 = torch.tensor([[0.9, 0.1]] * 4)
        verdict = torch.tensor([0.7, 0.7, 0.7, 0.7])
        loss = consistency_loss(mu0, verdict)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_positive_when_verdict_varies_within_class(self):
        mu0 = torch.tensor([[0.9, 0.1]] * 4)
        verdict = torch.tensor([0.9, 0.1, 0.9, 0.1])
        loss = consistency_loss(mu0, verdict)
        self.assertGreater(loss.item(), 0.0)

    def test_gradient_flows_to_verdict(self):
        mu0 = torch.tensor([[0.9, 0.1]] * 4)
        verdict = torch.tensor([0.9, 0.1, 0.9, 0.1], requires_grad=True)
        loss = consistency_loss(mu0, verdict)
        loss.backward()
        self.assertIsNotNone(verdict.grad)
        self.assertTrue((verdict.grad.abs() > 0).any())


if __name__ == "__main__":
    run_tests()
