"""Tests for nspe.consistency."""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.consistency import ConsistencyChecker, consistency_loss


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
        mu0 = torch.cat([torch.tensor([[0.9, 0.1]] * 4), torch.tensor([[0.1, 0.9]] * 2)])
        verdict = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        report = ConsistencyChecker(tau=0.5)(mu0, verdict)
        worst = report.worst_classes(n=1)
        self.assertEqual(len(worst), 1)
        self.assertEqual(worst[0]["size"], 4)
        self.assertGreater(worst[0]["disagreeing_pairs"], 0)
        self.assertLessEqual(len(worst[0]["example_indices"]), 5)


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
