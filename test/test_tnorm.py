"""Tests for nspe.logic.tnorm operator families."""

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from nspe.logic.ops import safe_log
from nspe.logic.tnorm import get_tnorm


def _log(*values: float) -> torch.Tensor:
    return safe_log(torch.tensor([list(values)]))


@instantiate_parametrized_tests
class TestTNormAlgebraicLaws(TestCase):
    @parametrize("name", ["product", "godel", "lukasiewicz"])
    def test_conj_with_identity_is_noop(self, name):
        tnorm = get_tnorm(name)
        log_x = _log(0.3, 1.0)
        index = torch.tensor([0, 0])
        out = tnorm.conj_segment(log_x, index, num_segments=1)
        torch.testing.assert_close(
            out.exp(), torch.tensor([[0.3]]), atol=1e-4, rtol=1e-4
        )

    @parametrize("name", ["product", "godel", "lukasiewicz"])
    def test_conj_with_zero_is_zero(self, name):
        tnorm = get_tnorm(name)
        log_x = _log(0.7, 1e-7)
        index = torch.tensor([0, 0])
        out = tnorm.conj_segment(log_x, index, num_segments=1)
        self.assertLess(out.exp().item(), 1e-4)

    @parametrize("name", ["product", "godel", "lukasiewicz"])
    def test_conj_is_monotonic(self, name):
        tnorm = get_tnorm(name)
        index = torch.tensor([0, 0])
        low = tnorm.conj_segment(_log(0.3, 0.5), index, 1)
        high = tnorm.conj_segment(_log(0.6, 0.5), index, 1)
        self.assertGreaterEqual(high.exp().item(), low.exp().item() - 1e-6)

    @parametrize("name", ["product", "godel", "lukasiewicz"])
    def test_conj_never_exceeds_min_literal(self, name):
        tnorm = get_tnorm(name)
        index = torch.tensor([0, 0, 0])
        out = tnorm.conj_segment(_log(0.9, 0.4, 0.7), index, 1)
        self.assertLessEqual(out.exp().item(), 0.4 + 1e-4)

    @parametrize("name", ["product", "godel", "lukasiewicz"])
    def test_disj_never_below_max_literal(self, name):
        tnorm = get_tnorm(name)
        index = torch.tensor([0, 0, 0])
        out = tnorm.disj_segment(_log(0.2, 0.6, 0.1), index, 1)
        self.assertGreaterEqual(out.exp().item(), 0.6 - 1e-4)

    @parametrize("name", ["product", "godel", "lukasiewicz"])
    def test_negation_is_involution(self, name):
        tnorm = get_tnorm(name)
        log_x = _log(0.37)
        twice = tnorm.neg(tnorm.neg(log_x))
        torch.testing.assert_close(
            twice.exp(), torch.tensor([[0.37]]), atol=1e-4, rtol=1e-4
        )

    @parametrize("name", ["product", "godel", "lukasiewicz"])
    def test_disj_pair_matches_disj_segment(self, name):
        tnorm = get_tnorm(name)
        a, b = _log(0.3), _log(0.8)
        pair = tnorm.disj_pair(a, b)
        combined = torch.cat([a, b], dim=-1)
        index = torch.tensor([0, 0])
        segment = tnorm.disj_segment(combined, index, 1)
        torch.testing.assert_close(pair.exp(), segment.exp(), atol=1e-4, rtol=1e-4)


class TestGodelGradientSparsity(TestCase):
    def test_only_argmin_literal_gets_gradient(self):
        tnorm = get_tnorm("godel")
        log_x = _log(0.9, 0.2, 0.6).requires_grad_(True)
        index = torch.tensor([0, 0, 0])
        out = tnorm.conj_segment(log_x, index, num_segments=1)
        out.sum().backward()
        grad = log_x.grad.squeeze(0)
        nonzero = (grad.abs() > 1e-6).sum().item()
        self.assertEqual(nonzero, 1)
        self.assertGreater(grad[1].abs().item(), 0.0)


class TestProductGradientIsDense(TestCase):
    def test_every_literal_gets_gradient(self):
        tnorm = get_tnorm("product")
        log_x = _log(0.9, 0.2, 0.6).requires_grad_(True)
        index = torch.tensor([0, 0, 0])
        out = tnorm.conj_segment(log_x, index, num_segments=1)
        out.sum().backward()
        grad = log_x.grad.squeeze(0)
        self.assertTrue((grad.abs() > 1e-6).all())


class TestLukasiewiczCanSaturate(TestCase):
    def test_long_conjunction_can_be_zero_gradient(self):
        tnorm = get_tnorm("lukasiewicz")
        log_x = safe_log(torch.full((1, 6), 0.5)).requires_grad_(True)
        index = torch.zeros(6, dtype=torch.long)
        out = tnorm.conj_segment(log_x, index, num_segments=1)
        self.assertLess(out.exp().item(), 1e-4)
        out.sum().backward()
        grad = log_x.grad.squeeze(0)
        self.assertTrue((grad.abs() < 1e-6).all())


if __name__ == "__main__":
    run_tests()
