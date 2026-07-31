"""Tests for nspe.train.loop.train_model, using a toy stub model.

No CLIP, network, or real dataset -- exercises only the loop's
mechanics (loss decreases, checkpoint is written, best-val-loss
selection) on a trivially learnable synthetic problem.
"""

import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.train.loop import train_model


class _ToyModel(nn.Module):
    """Linear-in-images stub: verdict = sigmoid(w . mean(images) + b)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        del texts
        features = images.mean(dim=(1, 2, 3), keepdim=True)
        return torch.sigmoid(self.linear(features)).flatten()


def _toy_batches(n_batches: int, batch_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        labels = (torch.rand(batch_size, generator=g) > 0.5).float()
        # images whose mean tracks the label, so the problem is learnable.
        base = labels.view(-1, 1, 1, 1) * 2 - 1
        images = base.expand(batch_size, 1, 2, 2) + 0.01 * torch.randn(
            batch_size, 1, 2, 2, generator=g
        )
        texts = ["x"] * batch_size
        yield images, texts, labels


class TestTrainModel(TestCase):
    def test_loss_decreases_and_checkpoint_written(self):
        model = _ToyModel()
        train_batches = list(_toy_batches(n_batches=8, batch_size=16, seed=0))
        val_batches = list(_toy_batches(n_batches=2, batch_size=16, seed=1))

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            result = train_model(
                model,
                forward_fn=lambda m, images, texts: m(images, texts),
                train_loader=train_batches,
                val_loader=val_batches,
                epochs=15,
                lr=0.1,
                device="cpu",
                checkpoint_path=checkpoint_path,
            )

            self.assertTrue(checkpoint_path.exists())
            self.assertLess(result["val_losses"][-1], result["val_losses"][0])
            self.assertEqual(result["best_val_loss"], min(result["val_losses"]))

            state = torch.load(checkpoint_path, weights_only=True)
            self.assertIn("linear.weight", state)


if __name__ == "__main__":
    run_tests()
