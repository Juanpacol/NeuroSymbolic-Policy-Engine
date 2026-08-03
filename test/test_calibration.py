"""Tests for nspe.calibration.VerdictCalibrator.

The load-bearing property is AUROC invariance: calibration relocates the
operating point of a compressed verdict distribution, and must be unable
to manufacture ranking quality that was not already there.
"""

import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.calibration import VerdictCalibrator
from nspe.eval.metrics import auroc


class TestVerdictCalibrator(TestCase):
    def test_strictly_monotone(self):
        calibrator = VerdictCalibrator(init_scale=2.0, init_bias=-1.0)
        verdict = torch.linspace(1e-4, 1 - 1e-4, 256)
        out = calibrator(verdict)
        self.assertTrue(bool((out[1:] > out[:-1]).all()))

    def test_auroc_is_invariant(self):
        torch.manual_seed(0)
        # A compressed verdict band, as the product t-norm produces.
        verdict = 0.1387 + 0.004 * torch.randn(512)
        labels = (torch.rand(512) > 0.6).float()
        calibrator = VerdictCalibrator(init_scale=3.0, init_bias=2.5)

        self.assertEqual(
            round(auroc(verdict, labels), 9),
            round(auroc(calibrator(verdict), labels), 9),
        )

    def test_scale_stays_positive_after_updates(self):
        calibrator = VerdictCalibrator(init_scale=0.5)
        self.assertEqual(round(calibrator.scale().item(), 5), 0.5)
        with torch.no_grad():
            calibrator.raw_scale.fill_(-50.0)
        self.assertGreater(calibrator.scale().item(), 0.0)

    def test_rejects_nonpositive_init_scale(self):
        with self.assertRaises(ValueError):
            VerdictCalibrator(init_scale=0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a second device")
    def test_fit_bias_to_base_rate_accepts_an_off_device_verdict(self):
        """Regression: the verdict may live on a different device.

        Callers accumulate verdicts across batches and commonly move
        them to CPU to do so (nspe.train.cli._warm_start does exactly
        this), while the calibrator itself may be on GPU.
        """
        calibrator = VerdictCalibrator().to("cuda")
        verdict = (0.1387 + 0.004 * torch.randn(64)).cpu()

        calibrator.fit_bias_to_base_rate(verdict, base_rate=0.35)

        self.assertLess(
            abs(calibrator(verdict.to("cuda")).mean().item() - 0.35), 0.01
        )

    def test_fit_bias_to_base_rate(self):
        torch.manual_seed(0)
        verdict = 0.1387 + 0.004 * torch.randn(1024)
        calibrator = VerdictCalibrator()
        calibrator.fit_bias_to_base_rate(verdict, base_rate=0.35)

        self.assertLess(abs(calibrator(verdict).mean().item() - 0.35), 0.01)

    def test_disabled_is_identity(self):
        calibrator = VerdictCalibrator(init_scale=4.0, init_bias=2.0, enabled=False)
        verdict = torch.rand(16)
        self.assertEqual(calibrator(verdict), verdict)

    def test_gradient_flows_to_verdict_and_params(self):
        calibrator = VerdictCalibrator()
        verdict = torch.rand(8, requires_grad=True)
        calibrator(verdict).sum().backward()

        self.assertIsNotNone(verdict.grad)
        self.assertTrue(bool((verdict.grad != 0).all()))
        self.assertIsNotNone(calibrator.raw_scale.grad)
        self.assertIsNotNone(calibrator.bias.grad)


if __name__ == "__main__":
    run_tests()
