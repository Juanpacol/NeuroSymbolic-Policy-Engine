"""Tests for nspe.train.loop.train_model, using a toy stub model.

No CLIP, network, or real dataset -- exercises only the loop's
mechanics (loss decreases, checkpoint is written, metric-based
selection, early stopping, resume) on a trivially learnable synthetic
problem.
"""

import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.train.loop import train_model
from nspe.train.seed import set_seed


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


def _forward(model, images, texts):
    return model(images, texts)


class TestTrainModel(TestCase):
    def test_loss_decreases_and_checkpoint_written(self):
        model = _ToyModel()
        train_batches = list(_toy_batches(n_batches=8, batch_size=16, seed=0))
        val_batches = list(_toy_batches(n_batches=2, batch_size=16, seed=1))

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            result = train_model(
                model,
                forward_fn=_forward,
                train_loader=train_batches,
                val_loader=val_batches,
                epochs=15,
                lr=0.1,
                patience=None,
                scheduler=None,
                device="cpu",
                checkpoint_path=checkpoint_path,
            )

            self.assertTrue(checkpoint_path.exists())
            self.assertLess(result["val_losses"][-1], result["val_losses"][0])
            self.assertEqual(result["select_metric"], "auroc")

            state = torch.load(checkpoint_path, weights_only=True)
            self.assertIn("linear.weight", state)

    def test_bce_selection_prefers_a_constant_model_that_auroc_rejects(self):
        """The reason checkpoint selection moved off validation BCE.

        A constant predictor sitting at the base rate has a finite BCE
        but carries no ranking information, so BCE alone cannot tell it
        apart from a model that actually discriminates.
        """
        from nspe.eval.metrics import auroc

        labels = torch.tensor([1.0] * 3 + [0.0] * 7)
        # Sits exactly at the base rate; ranks nothing.
        constant = torch.full((10,), 0.3)
        # Ranks every positive above every negative, but is badly
        # calibrated -- the compressed-verdict regime this pipeline was
        # actually in.
        discriminating = torch.tensor([0.05, 0.04, 0.03] + [0.02] * 7)

        constant_bce = nn.functional.binary_cross_entropy(constant, labels)
        discriminating_bce = nn.functional.binary_cross_entropy(discriminating, labels)
        self.assertLess(constant_bce.item(), discriminating_bce.item())

        self.assertEqual(auroc(constant, labels), 0.5)
        self.assertEqual(auroc(discriminating, labels), 1.0)

    def test_early_stopping_halts_before_max_epochs(self):
        model = _ToyModel()
        # Random labels: nothing to learn, so val AUROC never improves.
        val_batches = list(_toy_batches(n_batches=2, batch_size=16, seed=3))
        flat = [
            (images, texts, torch.zeros_like(labels).bernoulli_(0.5))
            for images, texts, labels in _toy_batches(4, 16, seed=2)
        ]

        result = train_model(
            model,
            forward_fn=_forward,
            train_loader=flat,
            val_loader=val_batches,
            epochs=50,
            lr=1e-4,
            patience=2,
            scheduler=None,
            device="cpu",
        )

        self.assertLess(len(result["train_losses"]), 50)

    def test_same_seed_reproduces_losses(self):
        train_batches = list(_toy_batches(n_batches=4, batch_size=16, seed=0))
        val_batches = list(_toy_batches(n_batches=2, batch_size=16, seed=1))

        runs = []
        for _ in range(2):
            set_seed(7)
            result = train_model(
                _ToyModel(),
                forward_fn=_forward,
                train_loader=train_batches,
                val_loader=val_batches,
                epochs=3,
                lr=0.05,
                patience=None,
                scheduler=None,
                seed=7,
                device="cpu",
            )
            runs.append(result["train_losses"])

        self.assertEqual(runs[0], runs[1])

    def test_pos_weight_changes_the_loss(self):
        batches = list(_toy_batches(n_batches=2, batch_size=16, seed=5))
        kwargs = dict(
            forward_fn=_forward,
            train_loader=batches,
            val_loader=batches,
            epochs=1,
            lr=0.0,
            patience=None,
            scheduler=None,
            seed=0,
            device="cpu",
        )
        unweighted = train_model(_ToyModel(), pos_weight=None, **kwargs)
        weighted = train_model(_ToyModel(), pos_weight=3.0, **kwargs)

        self.assertNotEqual(unweighted["train_losses"][0], weighted["train_losses"][0])

    def test_resume_restores_checkpointed_weights(self):
        train_batches = list(_toy_batches(n_batches=8, batch_size=16, seed=0))
        val_batches = list(_toy_batches(n_batches=2, batch_size=16, seed=1))

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            train_model(
                _ToyModel(),
                forward_fn=_forward,
                train_loader=train_batches,
                val_loader=val_batches,
                epochs=10,
                lr=0.1,
                patience=None,
                scheduler=None,
                device="cpu",
                checkpoint_path=checkpoint_path,
            )
            saved = torch.load(checkpoint_path, weights_only=True)

            resumed = _ToyModel()
            result = train_model(
                resumed,
                forward_fn=_forward,
                train_loader=train_batches,
                val_loader=val_batches,
                epochs=0,
                device="cpu",
                resume_from=checkpoint_path,
            )

            self.assertEqual(result["train_losses"], [])
            self.assertEqual(resumed.linear.weight, saved["linear.weight"])

    def test_rejects_unknown_select_metric(self):
        with self.assertRaises(ValueError):
            train_model(
                _ToyModel(),
                forward_fn=_forward,
                train_loader=[],
                val_loader=[],
                epochs=1,
                select_metric="nonsense",
            )


if __name__ == "__main__":
    run_tests()
