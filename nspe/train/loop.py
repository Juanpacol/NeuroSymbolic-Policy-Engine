"""Model-agnostic training loop: BCE on a verdict against a real label.

Shared by both the symbolic PolicyEngine (extractor -> reasoner) and the
NeuralBaselineClassifier -- the two differ only in how a batch maps to a
verdict tensor, which callers supply via ``forward_fn``. Everything else
(loss, optimizer scope, checkpointing, device handling) is identical, so
the loop itself never branches on model type.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def train_model(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, Tensor, list[str]], Tensor],
    train_loader: Iterable[tuple[Tensor, list[str], Tensor]],
    val_loader: Iterable[tuple[Tensor, list[str], Tensor]],
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Trains ``model``'s trainable parameters against a binary label.

    Args:
        model: the module to train. Only parameters with
            ``requires_grad=True`` are optimized (e.g. a frozen CLIP
            backbone is skipped automatically).
        forward_fn: maps ``(model, images, texts)`` to a verdict tensor
            of shape ``(batch,)`` in ``(0, 1)`` -- e.g.
            ``lambda m, i, t: m(i, t).verdicts["hateful"]`` for a
            ``PolicyEngine``, or ``lambda m, i, t: m(i, t)`` for
            ``NeuralBaselineClassifier``.
        train_loader: yields ``(images, texts, labels)`` batches.
        val_loader: yields ``(images, texts, labels)`` batches, used for
            checkpoint selection only (no gradient updates).
        epochs: number of passes over ``train_loader``.
        lr: Adam learning rate.
        device: ``"cpu"``, ``"mps"``, or ``"cuda"``.
        checkpoint_path: if given, the best-val-loss ``model.state_dict()``
            is saved here via ``torch.save``.

    Returns:
        A dict with ``best_val_loss`` and per-epoch ``train_losses`` and
        ``val_losses`` lists.
    """
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state: dict[str, Tensor] | None = None

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for images, texts, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            verdict = forward_fn(model, images, texts)
            loss = F.binary_cross_entropy(verdict, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        train_losses.append(running_loss / max(num_batches, 1))

        model.eval()
        running_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for images, texts, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                verdict = forward_fn(model, images, texts)
                loss = F.binary_cross_entropy(verdict, labels)
                running_val_loss += loss.item()
                num_val_batches += 1
        val_loss = running_val_loss / max(num_val_batches, 1)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if checkpoint_path is not None and best_state is not None:
        torch.save(best_state, checkpoint_path)

    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
