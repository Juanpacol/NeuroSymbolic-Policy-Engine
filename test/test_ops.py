"""Tests for nspe.logic.ops numerical primitives."""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.logic.ops import (
    gather_literal_log,
    log1mexp,
    safe_log,
    segment_amax,
    segment_amin,
    segment_sum,
)


class TestLog1mexp(TestCase):
    def test_matches_float64_reference(self):
        log_x = torch.linspace(-40, -1e-8, steps=2000, dtype=torch.float64)
        got = log1mexp(log_x)
        want = torch.log(1 - torch.exp(log_x))
        self.assertTrue(torch.isfinite(got).all())
        torch.testing.assert_close(got, want, atol=1e-8, rtol=1e-6)

    def test_near_zero_no_nan_or_inf(self):
        log_x = torch.tensor([-1e-12, -1e-6, -1e-3], dtype=torch.float64)
        got = log1mexp(log_x)
        self.assertTrue(torch.isfinite(got).all())

    def test_gradient_is_finite_near_zero(self):
        """A finite forward value is not enough.

        ``torch.where`` evaluates both branches and its backward
        multiplies the unselected one by zero, but ``0 * inf`` is NaN.
        Near ``log_x = 0`` the large branch's ``log1p(-exp(log_x))``
        rounds its input to exactly -1, so an unguarded implementation
        returns a correct value with a NaN gradient -- which silently
        destroys any training run where a predicate saturates.
        """
        for dtype in (torch.float32, torch.float64):
            log_x = torch.tensor(
                [-1e-12, -1e-8, -1e-4, -0.5, -30.0],
                dtype=dtype,
                requires_grad=True,
            )
            log1mexp(log_x).sum().backward()
            self.assertTrue(torch.isfinite(log_x.grad).all())

    def test_is_involution_with_safe_log(self):
        # log(1 - (1 - x)) == log(x)
        x = torch.tensor([0.1, 0.5, 0.9, 0.999])
        log_x = safe_log(x)
        twice = log1mexp(log1mexp(log_x))
        torch.testing.assert_close(twice, log_x, atol=1e-4, rtol=1e-4)


class TestSegmentReductions(TestCase):
    def test_segment_sum_matches_manual(self):
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        index = torch.tensor([0, 0, 1, 2])
        out = segment_sum(values, index, num_segments=3)
        torch.testing.assert_close(out, torch.tensor([[3.0, 3.0, 4.0]]))

    def test_segment_amax_matches_manual(self):
        values = torch.tensor([[1.0, 5.0, 3.0, 4.0]])
        index = torch.tensor([0, 0, 1, 2])
        out = segment_amax(values, index, num_segments=3)
        torch.testing.assert_close(out, torch.tensor([[5.0, 3.0, 4.0]]))

    def test_segment_amin_matches_manual(self):
        values = torch.tensor([[1.0, 5.0, 3.0, 4.0]])
        index = torch.tensor([0, 0, 1, 2])
        out = segment_amin(values, index, num_segments=3)
        torch.testing.assert_close(out, torch.tensor([[1.0, 3.0, 4.0]]))

    def test_empty_segment_uses_fill(self):
        values = torch.tensor([[1.0, 2.0]])
        index = torch.tensor([0, 0])
        out = segment_amax(values, index, num_segments=2, fill=-99.0)
        torch.testing.assert_close(out, torch.tensor([[2.0, -99.0]]))


class TestGatherLiteralLog(TestCase):
    def test_positive_and_negative_literals(self):
        log_mu = safe_log(torch.tensor([[0.9, 0.2]]))
        index = torch.tensor([0, 1, 0])
        sign = torch.tensor([1.0, 1.0, -1.0])
        out = gather_literal_log(log_mu, index, sign)
        expected = torch.tensor([[0.9, 0.2, 0.1]]).log()
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
