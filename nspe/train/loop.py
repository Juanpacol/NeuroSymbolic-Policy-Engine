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

Checkpoints exclude the frozen CLIP backbone. Both arms hold CLIP as a
registered submodule, so an unfiltered ``state_dict`` is over 99.9%
frozen weights that are already reproducible from the ``open_clip``
pretrained tag -- roughly 1.7GB per checkpoint against 1.7MB of trained
parameters, which exhausts a free-tier disk quota within a couple of
dozen runs.
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
_BACKBONE_ATTR = "clip"


def frozen_backbone_keys(model: nn.Module) -> set[str]:
    """Returns the state-dict keys owned by frozen CLIP submodules.

    Derived from the module tree rather than by matching key strings: a
    ``k.startswith("clip.")`` test would also swallow any future
    parameter that merely happens to be named that way, and the keys it
    must *not* touch -- ``extractor.head.zero_shot_weight`` above all,
    which the evaluation path relies on the checkpoint to carry -- are
    only distinguishable by which module owns them.

    Args:
        model: the module whose backbone keys to identify.

    Returns:
        The set of ``model.state_dict()`` keys belonging to a submodule
        named ``clip``, at any depth.
    """
    keys: set[str] = set()
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] == _BACKBONE_ATTR:
            keys.update(f"{name}.{key}" for key in module.state_dict())
    return keys


def trainable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Returns ``model.state_dict()`` without the frozen CLIP backbone.

    Args:
        model: the module to serialize.

    Returns:
        A state dict carrying every trained tensor and no backbone
        weights.
    """
    excluded = frozen_backbone_keys(model)
    return {k: v for k, v in model.state_dict().items() if k not in excluded}


def load_trainable_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    """Loads a checkpoint that may omit the frozen CLIP backbone.

    ``strict=False`` alone would also mask a genuinely truncated
    checkpoint, so every missing key is checked against the backbone key
    set and anything else is an error. Backbone weights are rebuilt from
    the ``open_clip`` pretrained tag during construction, so omitting
    them leaves the backbone correct rather than randomly initialized.
    Checkpoints written before filtering existed still load.

    Args:
        model: the module to load into, already constructed.
        state: a state dict from :func:`trainable_state_dict` or a full
            ``model.state_dict()``.

    Raises:
        RuntimeError: if the checkpoint carries unexpected keys, or is
            missing a key that is not part of the frozen backbone.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has unexpected keys: {sorted(unexpected)[:10]}")
    absent = sorted(set(missing) - frozen_backbone_keys(model))
    if absent:
        raise RuntimeError(f"checkpoint is missing trained keys: {absent[:10]}")


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
    max_grad_norm: float | None = 1.0,
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
        max_grad_norm: gradient-norm clip, or ``None`` to disable.
            Needed, not optional hygiene: a predicate saturating toward
            1 makes ``d log(1 - mu) / d mu`` diverge, so every negated
            occurrence of it produces an exploding gradient that drives
            the heads to NaN within a few steps.
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
        load_trainable_state_dict(
            model, torch.load(resume_from, weights_only=True)
        )

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
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
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
                torch.save(trainable_state_dict(model), path)
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
