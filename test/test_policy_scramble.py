"""Tests for the scrambled-policy control.

The load-bearing claim is that scrambling can never produce a policy
that fails to compile: base predicates are absent from the compiler's
dependency graph, so permuting them cannot introduce a cycle or an
illegal negation. That claim is what lets the control be generated
mechanically rather than hand-checked, so it is tested first and by
brute force.
"""

import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.policy.compiler import compile_policy
from nspe.policy.loader import dump_policy, load_policy, policy_to_dict
from nspe.policy.scramble import base_derangement, scramble_policy
from nspe.policy.schema import Literal, Policy, Predicate, Rule

_POLICY = "nspe/policies/hateful_memes.yaml"
_SEEDS = range(50)


def _layered_policy(num_base: int = 5) -> Policy:
    """A policy with negation and two derived strata, to stress the check."""
    predicates = [
        Predicate(f"b{i}", "base", description=f"base {i}") for i in range(num_base)
    ]
    predicates += [
        Predicate("d1", "derived"),
        Predicate("d2", "derived"),
        Predicate("v", "verdict"),
    ]
    rules = (
        Rule("R1", "d1", (Literal("b0"), Literal("b1")), (Literal("b2"),), 0.9),
        Rule("R2", "d2", (Literal("d1"), Literal("b3")), (Literal("b4"),), 0.8),
        Rule("R3", "v", (Literal("d2"), Literal("b0", negated=True)), (Literal("b1"),)),
    )
    return Policy(name="layered", predicates=tuple(predicates), rules=rules)


class TestScramblingNeverBreaksCompilation(TestCase):
    def test_real_policy_compiles_under_every_seed(self):
        policy = load_policy(_POLICY)
        for seed in _SEEDS:
            scrambled, _ = scramble_policy(policy, seed)
            compile_policy(scrambled)

    def test_layered_policy_with_negation_compiles_under_every_seed(self):
        policy = _layered_policy()
        for seed in _SEEDS:
            scrambled, _ = scramble_policy(policy, seed)
            compile_policy(scrambled)

    def test_strata_are_unchanged(self):
        # The concrete reason compilation is safe: stratification runs
        # over derived/verdict predicates only, which are untouched.
        policy = _layered_policy()
        intact = compile_policy(policy)
        for seed in _SEEDS:
            scrambled, _ = scramble_policy(policy, seed)
            self.assertEqual(
                compile_policy(scrambled).rule_stratum.tolist(),
                intact.rule_stratum.tolist(),
            )


class TestDerangement(TestCase):
    def test_is_a_bijection_with_no_fixed_point(self):
        names = tuple(f"p{i}" for i in range(6))
        for seed in _SEEDS:
            mapping = base_derangement(names, seed)
            self.assertEqual(set(mapping), set(names))
            self.assertEqual(set(mapping.values()), set(names))
            for src, dst in mapping.items():
                self.assertNotEqual(src, dst)

    def test_is_deterministic_in_the_seed(self):
        names = tuple(f"p{i}" for i in range(6))
        self.assertEqual(base_derangement(names, 3), base_derangement(names, 3))

    def test_two_names_swap(self):
        self.assertEqual(base_derangement(("a", "b"), 0), {"a": "b", "b": "a"})

    def test_a_single_name_has_no_derangement(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            base_derangement(("a",), 0)


class TestStructureIsPreserved(TestCase):
    def setUp(self):
        self.policy = load_policy(_POLICY)

    def test_rule_graph_and_confidences_survive(self):
        for seed in _SEEDS:
            scrambled, _ = scramble_policy(self.policy, seed)
            for intact, rule in zip(self.policy.rules, scrambled.rules, strict=True):
                self.assertEqual(rule.id, intact.id)
                self.assertEqual(rule.head, intact.head)
                self.assertEqual(rule.confidence, intact.confidence)
                self.assertEqual(len(rule.body), len(intact.body))
                self.assertEqual(len(rule.unless), len(intact.unless))

    def test_descriptions_stay_with_their_own_names(self):
        # The grounding signal must be identical; only the wiring moves.
        scrambled, _ = scramble_policy(self.policy, 0)
        self.assertEqual(
            scrambled.predicate_descriptions(), self.policy.predicate_descriptions()
        )
        self.assertEqual(scrambled.predicates, self.policy.predicates)

    def test_bodies_stay_duplicate_free_and_disjoint_from_unless(self):
        # A per-slot shuffle could repeat a predicate in one body, which
        # under the product t-norm is mu^2 -- a semantic change, not a
        # rewiring. A bijection cannot.
        for seed in _SEEDS:
            scrambled, _ = scramble_policy(self.policy, seed)
            for rule in scrambled.rules:
                body = [lit.predicate for lit in rule.body]
                unless = [lit.predicate for lit in rule.unless]
                self.assertEqual(len(set(body)), len(body), rule.id)
                self.assertEqual(len(set(unless)), len(unless), rule.id)
                self.assertEqual(set(body) & set(unless), set(), rule.id)

    def test_negation_flags_ride_along_with_their_literal(self):
        policy = _layered_policy()
        scrambled, _ = scramble_policy(policy, 0)
        negated = [lit.negated for rule in scrambled.rules for lit in rule.body]
        self.assertEqual(
            negated, [lit.negated for rule in policy.rules for lit in rule.body]
        )

    def test_derived_literals_are_left_alone(self):
        policy = _layered_policy()
        scrambled, _ = scramble_policy(policy, 0)
        # R2's body reads d1; only b3 may move.
        self.assertEqual(scrambled.rules[1].body[0].predicate, "d1")

    def test_the_wiring_actually_changes(self):
        for seed in _SEEDS:
            scrambled, _ = scramble_policy(self.policy, seed)
            self.assertNotEqual(scrambled.rules, self.policy.rules)
            self.assertNotEqual(
                compile_policy(scrambled).fingerprint,
                compile_policy(self.policy).fingerprint,
            )

    def test_name_records_the_seed(self):
        # nspe.eval.aggregate.group_key reads policy_name, so this is
        # what keeps control runs out of the intact result's mean.
        scrambled, _ = scramble_policy(self.policy, 7)
        self.assertEqual(scrambled.name, "hateful_memes_policy_scrambled_s7")


class TestYamlRoundTrip(TestCase):
    def _round_trip(self, policy: Policy) -> Policy:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.yaml"
            dump_policy(policy, path, header="generated\nby a test")
            self.assertTrue(path.read_text().startswith("# generated\n# by a test\n"))
            return load_policy(path)

    def test_intact_policy_survives_a_round_trip(self):
        policy = load_policy(_POLICY)
        reloaded = self._round_trip(policy)
        self.assertEqual(
            compile_policy(reloaded).fingerprint, compile_policy(policy).fingerprint
        )
        self.assertEqual(reloaded, policy)

    def test_scrambled_policy_survives_a_round_trip(self):
        for seed in _SEEDS:
            scrambled, _ = scramble_policy(load_policy(_POLICY), seed)
            self.assertEqual(self._round_trip(scrambled), scrambled)

    def test_negated_literals_serialize_back_to_a_mapping(self):
        policy = _layered_policy()
        rendered = policy_to_dict(policy)
        self.assertIn({"not": "b0"}, rendered["rules"][2]["body"])
        self.assertEqual(self._round_trip(policy), policy)

    def test_empty_optional_fields_are_omitted(self):
        rendered = policy_to_dict(_layered_policy())
        self.assertNotIn("cite", rendered["rules"][0])
        self.assertNotIn("unless", rendered["rules"][2]["body"])
        self.assertNotIn("description", rendered["predicates"][-1])


if __name__ == "__main__":
    run_tests()
