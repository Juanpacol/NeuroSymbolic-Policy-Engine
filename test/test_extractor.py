"""Tests for nspe.extractor and nspe.engine.

These tests download CLIP weights (~350MB for ViT-B-32) on first run
and are skipped if open_clip_torch is not installed (the `clip` extra).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.policy.schema import Literal, Policy, Predicate, Rule

try:
    import open_clip  # noqa: F401

    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False


def _tiny_policy() -> Policy:
    return Policy(
        name="tiny",
        predicates=(
            Predicate("slur", "base"),
            Predicate("target", "base"),
            Predicate("remove", "verdict"),
        ),
        rules=(Rule(id="R1", head="remove", body=(Literal("slur"), Literal("target"))),),
    )


class TestNeuroSymbolicLayer(TestCase):
    def setUp(self):
        if not _HAS_OPEN_CLIP:
            self.skipTest("open_clip_torch not installed (clip extra)")

    def test_output_shape_and_range(self):
        from nspe.extractor import NeuroSymbolicLayer

        policy = _tiny_policy()
        extractor = NeuroSymbolicLayer.from_policy(policy)
        images = torch.rand(3, 3, 224, 224)
        texts = ["a photo of a cat", "hello world", "some text"]

        mu0 = extractor(images, texts)
        self.assertEqual(mu0.shape, (3, 2))
        self.assertTrue((mu0 > 0).all())
        self.assertTrue((mu0 < 1).all())

    def test_clip_backbone_is_frozen(self):
        from nspe.extractor import NeuroSymbolicLayer

        policy = _tiny_policy()
        extractor = NeuroSymbolicLayer.from_policy(policy)
        self.assertTrue(all(not p.requires_grad for p in extractor.clip.parameters()))

    def test_gradient_reaches_heads_but_not_clip(self):
        from nspe.extractor import NeuroSymbolicLayer

        policy = _tiny_policy()
        extractor = NeuroSymbolicLayer.from_policy(policy)
        images = torch.rand(2, 3, 224, 224)
        texts = ["a", "b"]

        mu0 = extractor(images, texts)
        mu0.sum().backward()

        self.assertIsNotNone(extractor.heads.weight.grad)
        self.assertTrue((extractor.heads.weight.grad.abs() > 0).any())
        self.assertTrue(all(p.grad is None for p in extractor.clip.parameters()))


class TestPolicyEngine(TestCase):
    def setUp(self):
        if not _HAS_OPEN_CLIP:
            self.skipTest("open_clip_torch not installed (clip extra)")

    def test_end_to_end_forward_and_backward(self):
        from nspe.engine import PolicyEngine
        from nspe.extractor import NeuroSymbolicLayer
        from nspe.reasoner import PolicyKGReasoner

        policy = _tiny_policy()
        extractor = NeuroSymbolicLayer.from_policy(policy)
        reasoner = PolicyKGReasoner(policy)
        engine = PolicyEngine(extractor, reasoner)

        images = torch.rand(2, 3, 224, 224)
        texts = ["some caption", "another caption"]

        out = engine(images, texts)
        self.assertIn("remove", out.verdicts)
        self.assertEqual(out.verdicts["remove"].shape, (2,))

        out.verdicts["remove"].sum().backward()
        self.assertIsNotNone(extractor.heads.weight.grad)
        self.assertTrue(all(p.grad is None for p in extractor.clip.parameters()))


if __name__ == "__main__":
    run_tests()
