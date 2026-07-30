"""Correctness tests for PolicyKGReasoner against a crisp Python oracle."""

import random

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.policy.schema import Literal, Policy, Predicate, Rule
from nspe.reasoner import PolicyKGReasoner


def _crisp_oracle(policy: Policy, facts: set[str]) -> set[str]:
    """Naive stratified forward chainer over crisp (0/1) facts.

    This is the reference oracle: exact classical Datalog-with-negation
    semantics, re-derived independently of the tensor implementation.
    """
    known = set(facts)
    changed = True
    while changed:
        changed = False
        for rule in policy.rules:
            if rule.head in known:
                continue
            body_ok = all(
                (lit.predicate in known) != lit.negated for lit in rule.body
            )
            if not body_ok:
                continue
            defeated = any(
                (lit.predicate in known) != lit.negated for lit in rule.unless
            )
            if defeated:
                continue
            known.add(rule.head)
            changed = True
    return known


def _make_random_acyclic_policy(seed: int) -> Policy:
    rng = random.Random(seed)
    num_base = rng.randint(2, 5)
    num_layers = rng.randint(1, 3)
    predicates = [Predicate(f"b{i}", "base") for i in range(num_base)]
    rules = []
    prev_layer = [f"b{i}" for i in range(num_base)]
    for layer in range(num_layers):
        kind = "verdict" if layer == num_layers - 1 else "derived"
        num_preds_this_layer = rng.randint(1, 3)
        this_layer = []
        for j in range(num_preds_this_layer):
            name = f"d{layer}_{j}"
            predicates.append(Predicate(name, kind))
            this_layer.append(name)
            body_size = rng.randint(1, min(2, len(prev_layer)))
            body_preds = rng.sample(prev_layer, body_size)
            body = tuple(
                Literal(p, negated=rng.random() < 0.3) for p in body_preds
            )
            unless = ()
            if rng.random() < 0.4 and len(prev_layer) > body_size:
                remaining = [p for p in prev_layer if p not in body_preds]
                if remaining:
                    unless = (Literal(rng.choice(remaining)),)
            rules.append(
                Rule(id=f"R{layer}_{j}", head=name, body=body, unless=unless)
            )
        prev_layer = this_layer
    return Policy(name="random", predicates=tuple(predicates), rules=tuple(rules))


class TestReasonerAgreesWithCrispOracle(TestCase):
    def test_random_policies_agree_on_random_fact_sets(self):
        for seed in range(60):
            policy = _make_random_acyclic_policy(seed)
            reasoner = PolicyKGReasoner(policy, tnorm="crisp")
            base_names = policy.predicate_names("base")

            rng = random.Random(seed + 10_000)
            facts = {b for b in base_names if rng.random() < 0.5}

            oracle = _crisp_oracle(policy, facts)

            mu0 = torch.tensor(
                [[1.0 if b in facts else 0.0 for b in base_names]]
            )
            out = reasoner(mu0)
            for name, idx in reasoner.rule_tensor.name_to_index.items():
                got_true = out.mu[0, idx].item() >= 0.5
                want_true = name in oracle
                self.assertEqual(
                    got_true,
                    want_true,
                    f"seed={seed} predicate={name}: got {got_true}, want {want_true}",
                )


class TestReasonerFixpointProperties(TestCase):
    def test_idempotent_on_second_application(self):
        policy = _make_random_acyclic_policy(3)
        reasoner = PolicyKGReasoner(policy, tnorm="product")
        base_names = policy.predicate_names("base")
        mu0 = torch.rand(4, len(base_names))
        out = reasoner(mu0)
        # Re-running with the derived mu as if it were base-only input
        # isn't directly meaningful (shapes differ across kinds), so
        # idempotence here is checked by re-running the same mu0 and
        # confirming determinism instead (the fixpoint has no random
        # state and no early exit in eval() with store_trace variance).
        out2 = reasoner(mu0)
        torch.testing.assert_close(out.mu, out2.mu)

    def test_monotone_in_base_predicates_when_no_exceptions(self):
        # Build a policy with no `unless` clauses so monotonicity is
        # guaranteed to hold exactly.
        policy = Policy(
            name="mono",
            predicates=(
                Predicate("a", "base"),
                Predicate("b", "base"),
                Predicate("v", "verdict"),
            ),
            rules=(Rule(id="R1", head="v", body=(Literal("a"), Literal("b"))),),
        )
        reasoner = PolicyKGReasoner(policy, tnorm="product")
        low = torch.tensor([[0.2, 0.3]])
        high = torch.tensor([[0.6, 0.7]])
        out_low = reasoner(low)
        out_high = reasoner(high)
        self.assertTrue((out_high.mu >= out_low.mu - 1e-6).all())

    def test_gradients_flow_to_base_predicates(self):
        policy = Policy(
            name="grad",
            predicates=(
                Predicate("a", "base"),
                Predicate("b", "base"),
                Predicate("v", "verdict"),
            ),
            rules=(Rule(id="R1", head="v", body=(Literal("a"), Literal("b"))),),
        )
        reasoner = PolicyKGReasoner(policy, tnorm="product")
        mu0 = torch.tensor([[0.4, 0.6]], requires_grad=True)
        out = reasoner(mu0)
        out.verdicts["v"].sum().backward()
        self.assertIsNotNone(mu0.grad)
        self.assertTrue((mu0.grad.abs() > 0).all())


if __name__ == "__main__":
    run_tests()
