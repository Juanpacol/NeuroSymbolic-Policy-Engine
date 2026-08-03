"""Tests for the exhaustive wiring sweep.

The load-bearing test is
``test_column_permutation_equals_policy_rewiring``: the sweep permutes
mu0's columns instead of rebuilding 720 policies, and that shortcut is
only valid in one direction of the permutation. Getting it backwards
produces a plausible-looking sweep of the wrong wirings, which no other
check would catch.

Everything here is synthetic -- no CLIP, no dataset, no checkpoint.
"""

import json
import tempfile
from pathlib import Path

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.eval.metrics import auroc
from nspe.eval.wiring_sweep import (
    load_mu0_dump,
    main,
    summarize_sweep,
    sweep_wirings,
)
from nspe.policy.compiler import compile_policy
from nspe.policy.loader import load_policy
from nspe.policy.schema import Literal, Policy, Predicate, Rule
from nspe.policy.scramble import apply_permutation
from nspe.reasoner import PolicyKGReasoner

_REAL_POLICY = "nspe/policies/hateful_memes.yaml"


def _policy(num_base: int = 4) -> Policy:
    """A small policy with negation, an exception, and a `hateful` verdict."""
    predicates = [
        Predicate(f"b{i}", "base", description=f"base {i}") for i in range(num_base)
    ]
    predicates += [Predicate("signal", "derived"), Predicate("hateful", "verdict")]
    rules = (
        Rule("R1", "signal", (Literal("b0"), Literal("b1")), (Literal("b2"),), 0.9),
        Rule("R2", "hateful", (Literal("signal"),), (Literal("b3"),), 0.8),
    )
    return Policy(name="toy", predicates=tuple(predicates), rules=rules)


def _data(num_base: int = 4, n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    mu0 = torch.rand(n, num_base)
    # Labels track one column, so AUROC is not degenerate.
    labels = (mu0[:, 0] > 0.5).float()
    return mu0, labels


class TestColumnPermutationEquivalence(TestCase):
    def test_column_permutation_equals_policy_rewiring(self):
        """The identity the sweep's whole design rests on.

        Rewiring rule bodies by a bijection s over base names is the
        same computation as feeding mu0's columns permuted by
        perm[i] = column_of(s(names[i])). If the direction were
        inverted, this fails.
        """
        policy = _policy()
        mu0, _ = _data()
        names = policy.predicate_names("base")
        column = {name: i for i, name in enumerate(names)}
        reasoner = PolicyKGReasoner(policy, store_trace=False)

        for row in sweep_wirings(policy, mu0, _data()[1]):
            mapping = row["mapping"]
            rewired = PolicyKGReasoner(
                apply_permutation(policy, mapping), store_trace=False
            )
            expected = rewired(mu0).verdicts["hateful"]
            actual = reasoner(mu0[:, row["permutation"]]).verdicts["hateful"]

            torch.testing.assert_close(actual, expected)
            self.assertEqual(row["permutation"], [column[mapping[n]] for n in names])

    def test_the_real_policy_too(self):
        real = load_policy(_REAL_POLICY)
        mu0, labels = _data(num_base=6)
        rows = sweep_wirings(real, mu0, labels)
        reasoner = PolicyKGReasoner(real, store_trace=False)

        for row in rows[::37]:
            rewired = PolicyKGReasoner(
                apply_permutation(real, row["mapping"]), store_trace=False
            )
            torch.testing.assert_close(
                reasoner(mu0[:, row["permutation"]]).verdicts["hateful"],
                rewired(mu0).verdicts["hateful"],
            )


class TestSweepShape(TestCase):
    def test_row_count_and_derangement_count(self):
        mu0, labels = _data()
        rows = sweep_wirings(_policy(), mu0, labels)

        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(r["is_derangement"] for r in rows), 9)

    def test_six_predicates_gives_the_known_counts(self):
        mu0, labels = _data(num_base=6)
        rows = sweep_wirings(load_policy(_REAL_POLICY), mu0, labels)

        self.assertEqual(len(rows), 720)
        self.assertEqual(sum(r["is_derangement"] for r in rows), 265)

    def test_identity_appears_once_and_matches_a_direct_call(self):
        policy = _policy()
        mu0, labels = _data()
        rows = sweep_wirings(policy, mu0, labels)

        identity = [r for r in rows if r["permutation"] == [0, 1, 2, 3]]
        self.assertEqual(len(identity), 1)
        self.assertFalse(identity[0]["is_derangement"])

        direct = PolicyKGReasoner(policy, store_trace=False)(mu0).verdicts["hateful"]
        self.assertAlmostEqual(identity[0]["auroc"], auroc(direct, labels), places=9)

    def test_rejects_a_width_mismatch(self):
        mu0, labels = _data(num_base=3)
        with self.assertRaisesRegex(ValueError, "columns"):
            sweep_wirings(_policy(4), mu0, labels)

    def test_rejects_an_unknown_verdict(self):
        mu0, labels = _data()
        with self.assertRaisesRegex(ValueError, "not a verdict"):
            sweep_wirings(_policy(), mu0, labels, verdict="signal")


class TestSummarize(TestCase):
    def _rows(self):
        return [
            {
                "permutation": [0, 1],
                "mapping": {},
                "is_derangement": False,
                "auroc": 0.8,
            },
            {
                "permutation": [1, 0],
                "mapping": {},
                "is_derangement": True,
                "auroc": 0.6,
            },
            {
                "permutation": [0, 1],
                "mapping": {},
                "is_derangement": True,
                "auroc": 0.7,
            },
        ]

    def test_reports_intact_worst_and_spread(self):
        summary = summarize_sweep(self._rows())

        self.assertEqual(summary["intact_auroc"], 0.8)
        self.assertEqual(summary["worst"]["auroc"], 0.6)
        self.assertEqual(summary["best"]["auroc"], 0.8)
        self.assertAlmostEqual(summary["all_permutations"]["spread"], 0.2)
        self.assertEqual(summary["derangements"]["count"], 2)

    def test_rank_distinguishes_ties_from_strict_wins(self):
        rows = self._rows()
        rows[1]["auroc"] = 0.8
        summary = summarize_sweep(rows)

        self.assertEqual(summary["intact_rank"]["strictly_below"], 1)
        self.assertEqual(summary["intact_rank"]["at_or_below"], 3)
        self.assertEqual(summary["intact_rank"]["total"], 3)


class TestMu0Dump(TestCase):
    def _dump(self, tmp: str, **overrides) -> str:
        policy = load_policy(_REAL_POLICY)
        mu0, labels = _data(num_base=6, n=16)
        payload = {
            "schema": 1,
            "split": "validation",
            "policy_fingerprint": compile_policy(policy).fingerprint,
            "predicate_names": list(policy.predicate_names("base")),
            "mu0": mu0.tolist(),
            "labels": labels.tolist(),
        }
        payload.update(overrides)
        path = Path(tmp) / "mu0.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def test_round_trips(self):
        policy = load_policy(_REAL_POLICY)
        with tempfile.TemporaryDirectory() as tmp:
            mu0, labels = load_mu0_dump(self._dump(tmp), policy)
        self.assertEqual(tuple(mu0.shape), (16, 6))
        self.assertEqual(tuple(labels.shape), (16,))

    def test_rejects_mismatched_predicate_names(self):
        policy = load_policy(_REAL_POLICY)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._dump(tmp, predicate_names=["a", "b", "c", "d", "e", "f"])
            with self.assertRaisesRegex(ValueError, "predicate_names"):
                load_mu0_dump(path, policy)

    def test_rejects_a_fingerprint_mismatch(self):
        policy = load_policy(_REAL_POLICY)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._dump(tmp, policy_fingerprint="0" * 64)
            with self.assertRaisesRegex(ValueError, "policy_fingerprint"):
                load_mu0_dump(path, policy)

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = self._dump(tmp)
            out = Path(tmp) / "sweep.json"
            result = main(["--policy", _REAL_POLICY, "--mu0", dump, "--out", str(out)])
            written = json.loads(out.read_text())

        self.assertEqual(len(result["per_seed"]), 1)
        self.assertEqual(result["per_seed"][0]["num_examples"], 16)
        self.assertIn("intact_minus_worst", result["across_seeds"])
        self.assertEqual(written["schema"], 1)


if __name__ == "__main__":
    run_tests()
