"""Tests for nspe.eval.significance.

The exact test has a property worth pinning: at small n its p-value is
bounded away from zero, so an all-positive result can look unimpressive
purely because of the sample size. Several cases below assert that the
reported p equals the reported floor, which is what tells a reader the
design was saturated rather than the effect marginal.
"""

import json
from pathlib import Path

from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.eval.significance import min_achievable_p, sign_permutation_test

_COMMITTED = "docs/results/h1_h3"


class TestMinAchievableP(TestCase):
    def test_two_sided_floor(self):
        self.assertEqual(min_achievable_p(5), 2 / 32)
        self.assertEqual(min_achievable_p(10), 2 / 1024)

    def test_one_sided_floor_is_half(self):
        self.assertEqual(min_achievable_p(5, "greater"), 1 / 32)
        self.assertEqual(min_achievable_p(10, "less"), 1 / 1024)

    def test_five_seeds_cannot_reach_significance_two_sided(self):
        # The reason the project ran ten seeds rather than five.
        self.assertGreater(min_achievable_p(5), 0.05)
        self.assertLess(min_achievable_p(10), 0.05)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            min_achievable_p(0)
        with self.assertRaises(ValueError):
            min_achievable_p(5, "sideways")


class TestSignPermutationTest(TestCase):
    def test_all_positive_hits_the_floor_exactly(self):
        result = sign_permutation_test([0.1, 0.2, 0.3, 0.4, 0.5])

        self.assertEqual(result["num_positive"], 5)
        self.assertEqual(result["p_value"], 2 / 32)
        self.assertEqual(result["p_value"], result["min_achievable_p"])

    def test_symmetric_input_is_not_significant(self):
        result = sign_permutation_test([1.0, -1.0, 2.0, -2.0])
        self.assertEqual(result["p_value"], 1.0)

    def test_matches_hand_enumeration_at_n3(self):
        # Gaps [1, 2, 3] -> observed sum 6, the maximum over all 8 sign
        # assignments. Only +++ reaches |6|, and only --- reaches |-6|,
        # so two-sided p = 2/8.
        result = sign_permutation_test([1.0, 2.0, 3.0])

        self.assertEqual(result["num_permutations"], 8)
        self.assertEqual(result["p_value"], 2 / 8)
        self.assertEqual(
            sign_permutation_test([1.0, 2.0, 3.0], "greater")["p_value"], 1 / 8
        )

    def test_invariant_to_scaling(self):
        small = sign_permutation_test([0.001, 0.002, 0.003, 0.004])
        large = sign_permutation_test([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(small["p_value"], large["p_value"])

    def test_one_negative_gap_weakens_the_result(self):
        all_positive = sign_permutation_test([0.3, 0.3, 0.3, 0.3, 0.3])
        one_negative = sign_permutation_test([0.3, 0.3, 0.3, 0.3, -0.3])
        self.assertLess(all_positive["p_value"], one_negative["p_value"])

    def test_reports_the_mean_and_statistic(self):
        result = sign_permutation_test([0.1, 0.3])
        self.assertAlmostEqual(result["statistic"], 0.4)
        self.assertAlmostEqual(result["mean_gap"], 0.2)

    def test_rejects_empty_and_oversized_input(self):
        with self.assertRaises(ValueError):
            sign_permutation_test([])
        with self.assertRaisesRegex(ValueError, "Monte-Carlo"):
            sign_permutation_test([0.1] * 21)


class TestCommittedGaps(TestCase):
    def test_five_seed_test_gaps_are_saturated_not_marginal(self):
        """The committed 5-seed result, and why 10 seeds were needed.

        All five per-seed AUROC gaps are positive, which is the
        strongest outcome the design admits -- and it still only reaches
        p = 0.0625, because that is the floor at n=5.
        """
        if not Path(_COMMITTED).is_dir():
            self.skipTest("committed artifacts not present")

        gaps = []
        for seed in range(5):
            path = Path(_COMMITTED) / f"results_test_s{seed}.json"
            gaps.append(json.loads(path.read_text())["h3_explainability"]["auroc_gap"])

        result = sign_permutation_test(gaps)

        self.assertEqual(result["num_positive"], 5)
        self.assertEqual(result["p_value"], 2 / 32)
        self.assertEqual(result["p_value"], result["min_achievable_p"])


if __name__ == "__main__":
    run_tests()
