"""Model-agnostic training loop: BCE on a verdict against a real label.

Shared by both the symbolic PolicyEngine (extractor -> reasoner) and the
NeuralBaselineClassifier -- the two differ only in how a batch maps to a
verdict tensor, which callers supply via ``forward_fn``. Everything else
(loss, optimizer scope, checkpointing, device handling) is identical, so
the loop itself never branches on model type.

Checkpoint selection defaults to validation AUROC rather than validation
BCE. A model that collapses to predicting the base rate for every case
attains a respectable cross-entropy while carrying no information, so
selecting on BCE actively rewards the degenerate solution this pipeline
was previously falling into. AUROC is invariant to monotone rescaling
and only measures the ranking the model learned.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nspe.eval.metrics import binary_metrics
from nspe.train.seed import set_seed

_MAXIMIZE = {"auroc", "f1", "accuracy"}


def _weighted_bce(verdict: Tensor, labels: Tensor, pos_weight: float | None) -> Tensor:
    """BCE with an optional up-weight on the positive class.

    Stays in probability space rather than using
    ``binary_cross_entropy_with_logits``: the reasoner emits a truth
    degree, not a logit, and there is no logit to recover once the
    t-norm has run.

    Args:
        verdict: predicted probabilities, shape ``(batch,)``.
        labels: binary targets, shape ``(batch,)``.
        pos_weight: weight applied to positive-class terms, or ``None``
            for unweighted.

    Returns:
        Scalar loss tensor.
    """
    if pos_weight is None:
        return F.binary_cross_entropy(verdict, labels)
    weight = torch.where(
        labels > 0.5,
        torch.full_like(labels, pos_weight),
        torch.ones_like(labels),
    )
    return F.binary_cross_entropy(verdict, labels, weight=weight)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, Tensor, Any], Tensor],
    loader: Iterable[tuple[Tensor, Any, Tensor]],
    device: str,
    pos_weight: float | None,
) -> dict[str, float]:
    """Runs one validation pass and returns loss plus binary metrics."""
    model.eval()
    scores, targets = [], []
    total_loss, num_batches = 0.0, 0
    for inputs, aux, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        verdict = forward_fn(model, inputs, aux)
        total_loss += _weighted_bce(verdict, labels, pos_weight).item()
        num_batches += 1
        scores.append(verdict.detach().cpu())
        targets.append(labels.detach().cpu())

    if not scores:
        return {"loss": float("inf"), "auroc": 0.5, "f1": 0.0}
    metrics = binary_metrics(torch.cat(scores), torch.cat(targets))
    metrics["loss"] = total_loss / num_batches
    return metrics


def train_model(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, Tensor, Any], Tensor],
    train_loader: Iterable[tuple[Tensor, Any, Tensor]],
    val_loader: Iterable[tuple[Tensor, Any, Tensor]],
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    pos_weight: float | None = None,
    aux_loss_fn: Callable[[nn.Module, Tensor, Any], Tensor] | None = None,
    select_metric: str = "auroc",
    patience: int | None = 5,
    scheduler: str | None = "cosine",
    seed: int | None = 0,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Trains ``model``'s trainable parameters against a binary label.

    Args:
        model: the module to train. Only parameters with
            ``requires_grad=True`` are optimized (e.g. a frozen CLIP
            backbone is skipped automatically).
        forward_fn: maps ``(model, inputs, aux)`` to a verdict tensor
            of shape ``(batch,)`` in ``(0, 1)``.
        train_loader: yields ``(inputs, aux, labels)`` batches, where
            ``inputs`` is whatever ``forward_fn`` consumes (preprocessed
            images, or cached embeddings) and ``aux`` is the matching
            per-batch side input (raw texts, or ``None`` when the
            embedding already carries them).
        val_loader: yields batches in the same shape, used for
            checkpoint selection only (no gradient updates).
        epochs: maximum passes over ``train_loader``.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        pos_weight: up-weight applied to positive-class BCE terms.
            Pass ``n_negative / n_positive`` to offset class imbalance.
        aux_loss_fn: optional ``(model, inputs, aux) -> scalar`` added to
            the BCE term, for regularizers that need model internals
            rather than just the verdict.
        select_metric: validation metric driving checkpointing and early
            stopping. One of ``"auroc"``, ``"f1"``, ``"accuracy"``
            (maximized) or ``"bce"`` (minimized).
        patience: stop after this many epochs without improvement, or
            ``None`` to always run all ``epochs``.
        scheduler: ``"cosine"`` for ``CosineAnnealingLR``, or ``None``.
        seed: seed applied before training, or ``None`` to leave global
            RNG state alone.
        device: ``"cpu"``, ``"mps"``, or ``"cuda"``.
        checkpoint_path: if given, ``model.state_dict()`` is written here
            after every epoch that improves ``select_metric``, so an
            interrupted session keeps its progress.
        resume_from: if given, a ``state_dict`` loaded into ``model``
            before training starts. Optimizer state is not restored --
            AdamW re-warms in a few steps on heads this small.

    Returns:
        A dict with ``best_val_loss``, ``best_metric``, ``best_epoch``,
        ``select_metric``, per-epoch ``train_losses``, and a ``history``
        list of per-epoch validation metric dicts.
    """
    if select_metric not in _MAXIMIZE and select_metric != "bce":
        raise ValueError(f"unknown select_metric: {select_metric}")
    if seed is not None:
        set_seed(seed)
    if resume_from is not None:
        model.load_state_dict(torch.load(resume_from, weights_only=True))

    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    lr_schedule = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if scheduler == "cosine" and epochs > 0
        else None
    )

    maximize = select_metric in _MAXIMIZE
    best_score = -float("inf") if maximize else float("inf")
    best_epoch, best_val_loss = -1, float("inf")
    train_losses: list[float] = []
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        running_loss, num_batches = 0.0, 0
        for inputs, aux, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            verdict = forward_fn(model, inputs, aux)
            loss = _weighted_bce(verdict, labels, pos_weight)
            if aux_loss_fn is not None:
                loss = loss + aux_loss_fn(model, inputs, aux)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        train_losses.append(running_loss / max(num_batches, 1))
        if lr_schedule is not None:
            lr_schedule.step()

        metrics = _evaluate(model, forward_fn, val_loader, device, pos_weight)
        history.append(metrics)

        score = metrics[select_metric] if maximize else metrics["loss"]
        improved = score > best_score if maximize else score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_val_loss = metrics["loss"]
            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path)
        elif patience is not None and epoch - best_epoch >= patience:
            break

    return {
        "best_val_loss": best_val_loss,
        "best_metric": best_score,
        "best_epoch": best_epoch,
        "select_metric": select_metric,
        "train_losses": train_losses,
        "history": history,
        "val_losses": [m["loss"] for m in history],
    }
