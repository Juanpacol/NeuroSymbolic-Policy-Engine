"""Tests for nspe.trunk.PredicateTrunk."""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.trunk import PredicateTrunk


class TestPredicateTrunk(TestCase):
    def test_output_shape(self):
        trunk = PredicateTrunk(in_dim=64, hidden_dim=16)
        self.assertEqual(trunk(torch.randn(8, 64)).shape, (8, 16))
        self.assertEqual(trunk.out_dim, 16)

    def test_zero_hidden_dim_is_identity(self):
        trunk = PredicateTrunk(in_dim=32, hidden_dim=0)
        fused = torch.randn(4, 32)

        self.assertEqual(trunk(fused), fused)
        self.assertEqual(trunk.out_dim, 32)
        self.assertEqual(list(trunk.parameters()), [])

    def test_layernorm_restores_scale_on_unit_norm_features(self):
        """The reason a LayerNorm sits in the trunk at all.

        A fused CLIP embedding is two L2-normalized vectors, so
        per-element magnitude is ~0.03 and a plain linear layer on top
        emits logits with std ~0.025 -- every predicate lands at
        0.5 +/- 0.01 and the model is a near-constant function at init.
        """
        torch.manual_seed(0)
        half = torch.nn.functional.normalize(torch.randn(64, 512), dim=-1)
        fused = torch.cat([half, half], dim=-1)

        self.assertLess(fused.std().item(), 0.06)
        trunk = PredicateTrunk(in_dim=1024, hidden_dim=256, dropout=0.0)
        self.assertGreater(trunk(fused).std().item(), 0.5)

    def test_gradients_flow(self):
        trunk = PredicateTrunk(in_dim=16, hidden_dim=8, dropout=0.0)
        fused = torch.randn(4, 16, requires_grad=True)
        trunk(fused).sum().backward()

        self.assertIsNotNone(fused.grad)
        self.assertTrue(bool((fused.grad != 0).any()))


if __name__ == "__main__":
    run_tests()
