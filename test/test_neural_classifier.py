"""Tests for nspe.baselines.neural_classifier.NeuralBaselineClassifier."""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

try:
    import open_clip  # noqa: F401

    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False


class TestNeuralBaselineClassifier(TestCase):
    def setUp(self):
        if not _HAS_OPEN_CLIP:
            self.skipTest("open_clip_torch not installed (clip extra)")

    def test_output_shape_and_range(self):
        from nspe.baselines.neural_classifier import NeuralBaselineClassifier

        model = NeuralBaselineClassifier()
        images = torch.rand(3, 3, 224, 224)
        texts = ["a photo of a cat", "hello world", "some text"]

        verdict = model(images, texts)
        self.assertEqual(verdict.shape, (3,))
        self.assertTrue((verdict > 0).all())
        self.assertTrue((verdict < 1).all())

    def test_clip_backbone_is_frozen(self):
        from nspe.baselines.neural_classifier import NeuralBaselineClassifier

        model = NeuralBaselineClassifier()
        self.assertTrue(all(not p.requires_grad for p in model.clip.parameters()))

    def test_gradient_reaches_head_but_not_clip(self):
        from nspe.baselines.neural_classifier import NeuralBaselineClassifier

        model = NeuralBaselineClassifier()
        images = torch.rand(2, 3, 224, 224)
        texts = ["a", "b"]

        verdict = model(images, texts)
        verdict.sum().backward()

        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue((model.head.weight.grad.abs() > 0).any())
        self.assertTrue(all(p.grad is None for p in model.clip.parameters()))


if __name__ == "__main__":
    run_tests()
