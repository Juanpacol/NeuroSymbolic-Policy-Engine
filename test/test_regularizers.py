"""Tests for nspe.train.regularizers, the anti-collapse auxiliary losses."""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.train.regularizers import (
    activation_entropy_loss,
    anchor_loss,
    decorrelation_loss,
    zero_shot_targets,
)


class TestDecorrelationLoss(TestCase):
    def test_identical_predicates_score_one(self):
        column = torch.rand(64, 1)
        mu0 = column.repeat(1, 4)
        self.assertLess(abs(decorrelation_loss(mu0).item() - 1.0), 1e-4)

    def test_uncorrelated_predicates_score_near_zero(self):
        torch.manual_seed(0)
        mu0 = torch.rand(4096, 4)
        self.assertLess(decorrelation_loss(mu0).item(), 0.01)

    def test_anticorrelated_predicates_also_penalized(self):
        # Squared correlation: a perfect inverse copy carries no more
        # information than a perfect copy.
        column = torch.rand(64, 1)
        mu0 = torch.cat([column, 1.0 - column], dim=1)
        self.assertLess(abs(decorrelation_loss(mu0).item() - 1.0), 1e-4)

    def test_degenerate_shapes_are_safe(self):
        self.assertEqual(decorrelation_loss(torch.rand(8, 1)).item(), 0.0)
        self.assertEqual(decorrelation_loss(torch.rand(1, 4)).item(), 0.0)

    def test_differentiable(self):
        mu0 = torch.rand(32, 3, requires_grad=True)
        decorrelation_loss(mu0).backward()
        self.assertTrue(bool((mu0.grad != 0).any()))


class TestActivationEntropyLoss(TestCase):
    def test_minimized_at_target_rate(self):
        at_target = activation_entropy_loss(torch.full((32, 4), 0.5), 0.5)
        always_on = activation_entropy_loss(torch.full((32, 4), 0.99), 0.5)
        always_off = activation_entropy_loss(torch.full((32, 4), 0.01), 0.5)

        self.assertLess(at_target.item(), always_on.item())
        self.assertLess(at_target.item(), always_off.item())

    def test_respects_a_nondefault_target(self):
        loss = activation_entropy_loss(torch.full((32, 4), 0.2), target_rate=0.2)
        self.assertLess(
            loss.item(),
            activation_entropy_loss(torch.full((32, 4), 0.8), target_rate=0.2).item(),
        )

    def test_differentiable(self):
        mu0 = torch.rand(16, 3, requires_grad=True)
        activation_entropy_loss(mu0).backward()
        self.assertTrue(bool((mu0.grad != 0).any()))


class TestAnchorLoss(TestCase):
    def test_zero_when_matching_targets(self):
        mu0 = torch.rand(16, 5)
        self.assertEqual(anchor_loss(mu0, mu0.clone()).item(), 0.0)

    def test_grows_with_deviation(self):
        targets = torch.full((16, 3), 0.5)
        near = anchor_loss(torch.full((16, 3), 0.6), targets)
        far = anchor_loss(torch.full((16, 3), 0.9), targets)
        self.assertLess(near.item(), far.item())

    def test_differentiable_in_mu0_only(self):
        mu0 = torch.rand(8, 3, requires_grad=True)
        targets = torch.rand(8, 3)
        anchor_loss(mu0, targets).backward()
        self.assertTrue(bool((mu0.grad != 0).any()))


class TestZeroShotTargets(TestCase):
    def test_shape_range_and_detached(self):
        torch.manual_seed(0)
        half = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        fused = torch.cat([half, half], dim=-1)
        description = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)
        weight = torch.cat([description, description], dim=-1)

        targets = zero_shot_targets(fused, weight)

        self.assertEqual(targets.shape, (8, 3))
        self.assertTrue(bool(((targets > 0) & (targets < 1)).all()))
        self.assertFalse(targets.requires_grad)

    def test_higher_similarity_gives_higher_target(self):
        description = torch.nn.functional.normalize(torch.randn(1, 16), dim=-1)
        weight = torch.cat([description, description], dim=-1)
        aligned = torch.cat([description, description], dim=-1)
        opposed = -aligned

        targets = zero_shot_targets(torch.cat([aligned, opposed]), weight)
        self.assertGreater(targets[0, 0].item(), targets[1, 0].item())

    def test_unset_description_yields_a_flat_target(self):
        # A predicate with no description in the policy keeps an all-zero
        # row; it must not turn into amplified numerical noise.
        torch.manual_seed(0)
        fused = torch.randn(16, 8)
        weight = torch.zeros(2, 8)
        weight[0] = torch.nn.functional.normalize(torch.randn(8), dim=-1)

        targets = zero_shot_targets(fused, weight)
        self.assertEqual(targets[:, 1], torch.full((16,), 0.5))

    def test_targets_are_well_spread_regardless_of_scale(self):
        # The failure mode a fixed temperature had: similarities living
        # in a narrow, backbone-specific band saturating to hard 0/1.
        torch.manual_seed(0)
        description = torch.nn.functional.normalize(torch.randn(3, 32), dim=-1)
        weight = torch.cat([description, description], dim=-1)
        half = torch.nn.functional.normalize(torch.randn(128, 32), dim=-1)
        fused = torch.cat([half, half], dim=-1)

        for scale in (0.01, 1.0, 100.0):
            targets = zero_shot_targets(fused * scale, weight)
            self.assertTrue(bool(((targets > 0) & (targets < 1)).all()))
            self.assertGreater(targets.std().item(), 0.1)


if __name__ == "__main__":
    run_tests()
