"""Regression guard against predicate-layer collapse.

A full run of the unregularized model produced only 5 distinct
thresholded predicate signatures across 831 validation cases, out of 64
possible for 6 binary predicates. That makes the equivalence classes
underpinning H1 so coarse that the metric measures grouping artifacts
rather than reasoning, so signature diversity is a property worth
asserting rather than discovering after a training run.

Uses PredicateHead directly rather than NeuroSymbolicLayer so no CLIP
download is required.
"""

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.policy.schema import Literal, Policy, Predicate, Rule
from nspe.reasoner import PolicyKGReasoner
from nspe.train.regularizers import (
    activation_entropy_loss,
    decorrelation_loss,
)
from nspe.trunk import PredicateHead

_NUM_PREDICATES = 6
_IN_DIM = 32


def _policy() -> Policy:
    """A policy shaped like hateful_memes.yaml: conjunctions plus an unless."""
    names = [f"p{i}" for i in range(_NUM_PREDICATES)]
    return Policy(
        name="diversity",
        predicates=tuple(
            [Predicate(n, "base") for n in names]
            + [Predicate("signal", "derived"), Predicate("verdict", "verdict")]
        ),
        rules=(
            Rule(
                id="R1",
                head="signal",
                body=(Literal("p0"), Literal("p1")),
                unless=(Literal("p3"),),
                confidence=0.95,
            ),
            Rule(
                id="R2",
                head="verdict",
                body=(Literal("signal"),),
                unless=(Literal("p5"),),
            ),
        ),
    )


def _signature_count(mu0: torch.Tensor) -> int:
    return int(torch.unique((mu0 >= 0.5).float(), dim=0).shape[0])


def _train(head: nn.Module, features, labels, regularize: bool, steps: int = 200):
    reasoner = PolicyKGReasoner(_policy(), store_trace=False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        mu0 = head(features)
        verdict = reasoner(mu0).verdicts["verdict"]
        loss = nn.functional.binary_cross_entropy(verdict, labels)
        if regularize:
            loss = loss + 0.5 * decorrelation_loss(mu0)
            loss = loss + 0.2 * activation_entropy_loss(mu0)
        loss.backward()
        # Mirrors train_model: without this, a predicate saturating
        # toward 1 explodes the gradient through its negated occurrences.
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
    return head(features).detach()


class TestPredicateDiversity(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = torch.randn(256, _IN_DIM)
        # Label depends on several independent directions, so a diverse
        # predicate layer is genuinely learnable from these features.
        projection = torch.randn(_IN_DIM, 3)
        self.labels = (self.features @ projection).sum(dim=1).sigmoid().round()

    def test_regularized_layer_produces_diverse_signatures(self):
        head = PredicateHead(_IN_DIM, _NUM_PREDICATES, hidden_dim=32, dropout=0.0)
        mu0 = _train(head, self.features, self.labels, regularize=True)

        self.assertGreaterEqual(_signature_count(mu0), 8)

    def test_predicates_do_not_collapse_into_copies(self):
        head = PredicateHead(_IN_DIM, _NUM_PREDICATES, hidden_dim=32, dropout=0.0)
        mu0 = _train(head, self.features, self.labels, regularize=True)

        self.assertLess(decorrelation_loss(mu0).item(), 0.5)

    def test_every_predicate_stays_informative(self):
        # A predicate pinned at always-on or always-off contributes no
        # bit to the signature no matter what the verdict loss says.
        head = PredicateHead(_IN_DIM, _NUM_PREDICATES, hidden_dim=32, dropout=0.0)
        mu0 = _train(head, self.features, self.labels, regularize=True)

        activation_rate = (mu0 >= 0.5).float().mean(dim=0)
        self.assertTrue(
            bool(((activation_rate > 0.02) & (activation_rate < 0.98)).all())
        )

    def test_head_emits_a_strictly_interior_range(self):
        # Phase 3: saturating to exactly 0 or 1 kills the gradient through
        # every negated occurrence of a predicate.
        head = PredicateHead(_IN_DIM, _NUM_PREDICATES, hidden_dim=0, mu_eps=1e-4)
        with torch.no_grad():
            head.logit_scale.fill_(1e6)
        mu0 = head(torch.randn(64, _IN_DIM) * 100)

        self.assertTrue(bool((mu0 > 0.0).all()))
        self.assertTrue(bool((mu0 < 1.0).all()))


if __name__ == "__main__":
    run_tests()
