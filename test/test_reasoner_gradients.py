"""Gradient-correctness tests for PolicyKGReasoner.

MPS has no float64 support, so gradcheck (which requires float64 for a
tight numerical tolerance) runs CPU-only. That is a hard platform
limitation, not a choice: torch.rand(..., device='mps', dtype=torch.float64)
raises directly.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.policy.schema import Literal, Policy, Predicate, Rule
from nspe.reasoner import PolicyKGReasoner


def _chain_policy() -> Policy:
    return Policy(
        name="chain",
        predicates=(
            Predicate("a", "base"),
            Predicate("b", "base"),
            Predicate("c", "base"),
            Predicate("mid", "derived"),
            Predicate("v", "verdict"),
        ),
        rules=(
            Rule(id="R1", head="mid", body=(Literal("a"), Literal("b"))),
            Rule(id="R2", head="v", body=(Literal("mid"), Literal("c"))),
        ),
    )


class TestGradcheckProduct(TestCase):
    def test_gradcheck_cpu_float64(self):
        policy = _chain_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product").double()
        mu0 = torch.tensor(
            [[0.3, 0.6, 0.8], [0.2, 0.9, 0.4]], dtype=torch.float64, requires_grad=True
        )

        def fn(x):
            return reasoner(x).verdicts["v"]

        self.assertTrue(torch.autograd.gradcheck(fn, (mu0,), eps=1e-6, atol=1e-4))

    def test_gradgradcheck_cpu_float64(self):
        policy = _chain_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product").double()
        mu0 = torch.tensor([[0.4, 0.5, 0.7]], dtype=torch.float64, requires_grad=True)

        def fn(x):
            return reasoner(x).verdicts["v"]

        self.assertTrue(torch.autograd.gradgradcheck(fn, (mu0,), eps=1e-6, atol=1e-3))


class TestGradientFlowEveryLiteral(TestCase):
    def test_product_every_base_predicate_gets_gradient(self):
        policy = _chain_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product")
        mu0 = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
        out = reasoner(mu0)
        out.verdicts["v"].sum().backward()
        self.assertTrue((mu0.grad.abs() > 1e-6).all())

    def test_godel_gradient_count_matches_argmin_literals(self):
        # A single rule, single stratum: mid <- a AND b AND c, with a
        # strict min among the three -- exactly one should get gradient.
        policy = Policy(
            name="godel_chain",
            predicates=(
                Predicate("a", "base"),
                Predicate("b", "base"),
                Predicate("c", "base"),
                Predicate("v", "verdict"),
            ),
            rules=(
                Rule(
                    id="R1", head="v", body=(Literal("a"), Literal("b"), Literal("c"))
                ),
            ),
        )
        reasoner = PolicyKGReasoner(policy, tnorm="godel")
        mu0 = torch.tensor([[0.9, 0.2, 0.6]], requires_grad=True)
        out = reasoner(mu0)
        out.verdicts["v"].sum().backward()
        nonzero = (mu0.grad.abs() > 1e-6).sum().item()
        self.assertEqual(nonzero, 1)


class TestNoNanAtExtremes(TestCase):
    def test_exact_zero_and_one_inputs_give_finite_gradients(self):
        policy = _chain_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product")
        mu0 = torch.tensor([[0.0, 1.0, 1.0]], requires_grad=True)
        out = reasoner(mu0)
        out.verdicts["v"].sum().backward()
        self.assertTrue(torch.isfinite(out.mu).all())
        self.assertTrue(torch.isfinite(mu0.grad).all())


def _exception_only_policy() -> Policy:
    """A policy where ``guard`` appears only under ``unless:``."""
    return Policy(
        name="exception_only",
        predicates=(
            Predicate("a", "base"),
            Predicate("guard", "base"),
            Predicate("v", "verdict"),
        ),
        rules=(
            Rule(
                id="R1",
                head="v",
                body=(Literal("a"),),
                unless=(Literal("guard"),),
            ),
        ),
    )


class TestExceptionOnlyPredicateGradients(TestCase):
    """Predicates reachable only through negation must stay trainable.

    Negation goes through ``log1mexp``, whose two-branch ``torch.where``
    used to return a finite value with a NaN gradient near ``log_x = 0``,
    and whose gradient diverges as a predicate saturates toward 1. For a
    predicate appearing only under ``unless:``, that path is its only
    source of gradient, so either failure silently freezes it for the
    whole run.
    """

    def test_gradient_is_finite_and_nonzero_across_the_range(self):
        reasoner = PolicyKGReasoner(_exception_only_policy(), tnorm="product")
        guard_index = reasoner.rule_tensor.name_to_index["guard"]

        for guard in (1e-4, 1e-3, 0.5, 1 - 1e-3, 1 - 1e-4):
            mu0 = torch.tensor([[0.7, guard]], requires_grad=True)
            reasoner(mu0).verdicts["v"].sum().backward()

            grad = mu0.grad[0, guard_index]
            self.assertTrue(torch.isfinite(grad), f"non-finite grad at {guard}")
            self.assertNotEqual(grad.item(), 0.0, f"dead grad at {guard}")

    def test_saturated_guard_does_not_poison_other_gradients(self):
        reasoner = PolicyKGReasoner(_exception_only_policy(), tnorm="product")
        mu0 = torch.tensor([[0.7, 1.0 - 1e-7]], requires_grad=True)
        reasoner(mu0).verdicts["v"].sum().backward()

        self.assertTrue(torch.isfinite(mu0.grad).all())


class TestLearnableConfidence(TestCase):
    def test_rule_confidence_receives_gradient(self):
        policy = _chain_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product", learnable_confidence=True)
        mu0 = torch.tensor([[0.6, 0.7, 0.8]])
        out = reasoner(mu0)
        out.verdicts["v"].sum().backward()
        self.assertIsNotNone(reasoner.log_rule_conf.grad)
        self.assertTrue((reasoner.log_rule_conf.grad.abs() > 0).any())


if __name__ == "__main__":
    run_tests()
