"""Tests for nspe.eval.diagnostics."""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.eval.diagnostics import predicate_stats, signature_distribution

_NAMES = ("slur", "target", "benign")


class TestPredicateStats(TestCase):
    def test_activation_rate_and_mean(self):
        mu0 = torch.tensor(
            [
                [0.9, 0.1, 0.5],
                [0.9, 0.1, 0.5],
                [0.1, 0.1, 0.5],
                [0.1, 0.1, 0.5],
            ]
        )
        stats = predicate_stats(mu0, _NAMES)

        self.assertEqual(stats["slur"]["activation_rate"], 0.5)
        self.assertEqual(stats["target"]["activation_rate"], 0.0)
        self.assertEqual(stats["benign"]["activation_rate"], 1.0)
        self.assertLess(abs(stats["slur"]["mean"] - 0.5), 1e-6)

    def test_flags_duplicated_predicates(self):
        # The collapse signature: two heads that are the same detector.
        column = torch.rand(64, 1)
        mu0 = torch.cat([column, column, torch.rand(64, 1)], dim=1)
        stats = predicate_stats(mu0, _NAMES)

        self.assertGreater(stats["slur"]["max_abs_correlation"], 0.99)
        self.assertGreater(stats["target"]["max_abs_correlation"], 0.99)

    def test_independent_predicates_show_low_correlation(self):
        torch.manual_seed(0)
        stats = predicate_stats(torch.rand(4096, 3), _NAMES)
        for name in _NAMES:
            self.assertLess(stats[name]["max_abs_correlation"], 0.1)

    def test_single_row_is_safe(self):
        stats = predicate_stats(torch.rand(1, 3), _NAMES)
        self.assertEqual(set(stats), set(_NAMES))


class TestSignatureDistribution(TestCase):
    def test_counts_and_ordering(self):
        mu0 = torch.tensor(
            [
                [0.9, 0.9, 0.1],
                [0.9, 0.9, 0.1],
                [0.9, 0.9, 0.1],
                [0.1, 0.1, 0.1],
            ]
        )
        rows = signature_distribution(mu0, _NAMES)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["count"], 3)
        self.assertEqual(rows[0]["signature"], [1, 1, 0])
        self.assertEqual(rows[0]["active"], ["slur", "target"])
        self.assertEqual(rows[0]["share"], 0.75)
        self.assertEqual(rows[1]["count"], 1)

    def test_top_k_truncates(self):
        torch.manual_seed(0)
        rows = signature_distribution(torch.rand(256, 3), _NAMES, top_k=2)
        self.assertEqual(len(rows), 2)

    def test_collapsed_layer_yields_a_single_signature(self):
        rows = signature_distribution(torch.full((128, 3), 0.9), _NAMES)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["share"], 1.0)


if __name__ == "__main__":
    run_tests()
