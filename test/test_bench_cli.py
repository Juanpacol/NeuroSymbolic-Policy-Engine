"""Tests for nspe.bench.cli.

The load-bearing test here is
``test_crisp_arm_agrees_with_clingo_batched``: the benchmark's headline
speedup is only meaningful if the arm it compares against Clingo
actually computes the same thing, and that arm's inputs are built by
this module rather than by hand.
"""

import argparse
import json
import random
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.baselines.clingo_engine import ClingoEngine
from nspe.bench.cli import (
    _build_envelope,
    _clingo_reps,
    _crisp_mu0,
    _policy_from_args,
    _random_facts,
    build_parser,
    run_sweep,
)
from nspe.data.synthetic import make_layered_policy
from nspe.reasoner import PolicyKGReasoner

_NUM_BASE = 6


def _policy():
    return make_layered_policy(_NUM_BASE, 8, num_layers=2, seed=0)


class TestCrispMu0(TestCase):
    def test_encodes_fact_sets_in_column_order(self):
        base_names = ("a", "b", "c")
        fact_sets = [{"a", "c"}, set(), {"a", "b", "c"}]

        mu0 = _crisp_mu0(fact_sets, base_names, "cpu")

        self.assertEqual(mu0.shape, (3, 3))
        for row, facts in enumerate(fact_sets):
            for col, name in enumerate(base_names):
                expected = 1.0 if name in facts else 0.0
                self.assertEqual(mu0[row, col].item(), expected)

    def test_values_are_exactly_zero_or_one(self):
        base_names = ("a", "b")
        mu0 = _crisp_mu0([{"a"}, {"b"}], base_names, "cpu")
        self.assertTrue(bool(((mu0 == 0.0) | (mu0 == 1.0)).all()))


class TestCrispArmParity(TestCase):
    def test_crisp_arm_agrees_with_clingo_batched(self):
        """The benchmark's own input path must preserve Clingo parity.

        test_clingo_agreement.py proves the crisp t-norm matches Clingo,
        but on hand-built single-case inputs. This goes through
        ``_crisp_mu0`` and runs the whole batch at once, which is what
        the benchmark actually times -- a transposed or mis-ordered
        tensor would leave that test passing and this one failing.
        """
        policy = _policy()
        base_names = policy.predicate_names("base")
        rng = random.Random(0)
        fact_sets = [_random_facts(base_names, rng) for _ in range(16)]

        reasoner = PolicyKGReasoner(policy, tnorm="crisp", store_trace=False)
        out = reasoner(_crisp_mu0(fact_sets, base_names, "cpu"))
        engine = ClingoEngine(policy)

        for row, facts in enumerate(fact_sets):
            derived = engine.infer(facts)
            for name in policy.predicate_names("verdict"):
                self.assertEqual(
                    out.verdicts[name][row].item() >= 0.5,
                    name in derived,
                    f"case {row}, verdict {name}",
                )


class TestClingoReps(TestCase):
    def test_zero_budget_falls_back_to_the_floor(self):
        self.assertEqual(_clingo_reps(elapsed_ms=100.0, reps=200, budget_s=0.0), 5)

    def test_never_exceeds_the_requested_reps(self):
        self.assertEqual(_clingo_reps(elapsed_ms=0.001, reps=3, budget_s=1000.0), 3)

    def test_scales_with_the_budget(self):
        # 10ms per rep, 1s budget -> 100 reps.
        self.assertEqual(_clingo_reps(elapsed_ms=10.0, reps=200, budget_s=1.0), 100)

    def test_survives_a_zero_measurement(self):
        self.assertGreaterEqual(_clingo_reps(elapsed_ms=0.0, reps=200, budget_s=1.0), 5)

    def test_floor_does_not_override_a_smaller_explicit_request(self):
        # The floor guards against budget starvation, not against a
        # caller who deliberately asked for a short run.
        self.assertEqual(_clingo_reps(elapsed_ms=100.0, reps=2, budget_s=0.0), 2)


class TestRunSweep(TestCase):
    def test_rows_carry_three_arms_and_both_speedups(self):
        rows = run_sweep(
            _policy(), "cpu", (1, 4), warmup=1, reps=2, clingo_budget_s=1.0
        )

        self.assertEqual([r["batch_size"] for r in rows], [1, 4])
        for row in rows:
            for arm in ("reasoner_product", "reasoner_crisp", "clingo"):
                self.assertIn(arm, row)
                self.assertEqual(row[arm]["batch_size"], row["batch_size"])
                self.assertIn("per_item_median_ms", row[arm])
            self.assertGreater(row["speedup_median_crisp"], 0.0)
            self.assertGreater(row["speedup_median_product"], 0.0)

    def test_clingo_reps_respect_the_budget(self):
        rows = run_sweep(
            _policy(), "cpu", (4,), warmup=1, reps=200, clingo_budget_s=0.0
        )
        self.assertEqual(rows[0]["clingo"]["reps"], 5)


class TestParserAndEnvelope(TestCase):
    def test_synthetic_flag_builds_a_policy_of_that_size(self):
        args = build_parser().parse_args(["--synthetic", "12", "20"])
        policy = _policy_from_args(args)

        self.assertEqual(len(policy.predicate_names("base")), 12)
        self.assertTrue(policy.name.startswith("synthetic_b12"))

    def test_policy_and_synthetic_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--policy", "x.yaml", "--synthetic", "1", "2"])

    def test_default_policy_resolves_when_neither_is_given(self):
        policy = _policy_from_args(build_parser().parse_args([]))
        self.assertTrue(len(policy.rules) > 0)

    def test_envelope_is_versioned_and_json_round_trips(self):
        policy = _policy()
        rows = run_sweep(policy, "cpu", (1,), warmup=1, reps=2, clingo_budget_s=1.0)

        envelope = _build_envelope(policy, "cpu", rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.json"
            path.write_text(json.dumps(envelope, indent=2))
            reloaded = json.loads(path.read_text())

        self.assertEqual(reloaded["schema_version"], 2)
        self.assertEqual(reloaded["policy_name"], policy.name)
        self.assertIn("policy_fingerprint", reloaded)
        self.assertIn("speedup_median_crisp", reloaded["results"][0])
        self.assertIn("reasoner_product", reloaded["results"][0])


if __name__ == "__main__":
    run_tests()
