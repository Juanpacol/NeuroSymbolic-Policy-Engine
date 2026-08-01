"""Threshold-free and threshold-fitted binary classification metrics.

Pure tensor implementations, no sklearn dependency. AUROC matters here
beyond convention: selecting checkpoints on BCE rewards a model that
collapses to the base rate, since a constant predictor at the base rate
has a respectable cross-entropy while carrying no information. AUROC is
invariant to any monotone rescaling of the scores, so it measures only
the ranking the model actually learned.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nspe.logic.ops import segment_sum


def _average_ranks(x: Tensor) -> Tensor:
    """Returns 1-based ranks of ``x``, averaging over tied values.

    Args:
        x: scores, shape ``(n,)``.

    Returns:
        Tensor of shape ``(n,)`` with the rank of each element.
    """
    order = torch.argsort(x)
    sorted_x = x[order]
    positions = torch.arange(1, x.numel() + 1, dtype=torch.float64, device=x.device)
    _, inverse, counts = torch.unique(sorted_x, return_inverse=True, return_counts=True)
    group_sums = segment_sum(positions, inverse, counts.numel())
    ranks = torch.empty_like(positions)
    ranks[order] = (group_sums / counts.to(positions.dtype))[inverse]
    return ranks


def auroc(scores: Tensor, labels: Tensor) -> float:
    """Computes AUROC via the Mann-Whitney U statistic, tie-corrected.

    Args:
        scores: predicted scores, shape ``(n,)``. Any monotone rescaling
            leaves the result unchanged.
        labels: binary ground truth, shape ``(n,)``.

    Returns:
        AUROC in ``[0, 1]``, or ``0.5`` if either class is absent.
    """
    labels = labels.flatten()
    positive = labels > 0.5
    num_pos = int(positive.sum().item())
    num_neg = int(labels.numel() - num_pos)
    if num_pos == 0 or num_neg == 0:
        return 0.5

    ranks = _average_ranks(scores.flatten().to(torch.float64))
    rank_sum = ranks[positive].sum().item()
    return (rank_sum - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)


def _sweep(scores: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Returns candidate thresholds and their F1 and accuracy.

    Sweeps every distinct score as a threshold, using cumulative counts
    over the descending score order rather than re-evaluating the
    confusion matrix per candidate.

    Args:
        scores: predicted scores, shape ``(n,)``.
        labels: binary ground truth, shape ``(n,)``.

    Returns:
        A tuple ``(thresholds, f1, accuracy)``, each of shape
        ``(num_candidate,)``.
    """
    scores = scores.flatten().to(torch.float64)
    labels = (labels.flatten() > 0.5).to(torch.float64)
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    num_pos = sorted_labels.sum()
    num_total = float(scores.numel())
    true_pos = torch.cumsum(sorted_labels, dim=0)
    predicted_pos = torch.arange(
        1, scores.numel() + 1, dtype=torch.float64, device=scores.device
    )
    false_pos = predicted_pos - true_pos
    true_neg = (num_total - num_pos) - false_pos

    precision = true_pos / predicted_pos
    recall = true_pos / num_pos.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    accuracy = (true_pos + true_neg) / num_total

    # Only the last index of a run of tied scores is a realizable
    # threshold; earlier ones would split cases that compare equal.
    keep = torch.ones_like(sorted_scores, dtype=torch.bool)
    keep[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    return sorted_scores[keep], f1[keep], accuracy[keep]


def best_threshold(
    scores: Tensor, labels: Tensor, objective: str = "f1"
) -> tuple[float, float]:
    """Finds the threshold maximizing ``objective`` on these data.

    Fitting a threshold on the same split it is reported for would
    inflate the result; callers are expected to fit on validation and
    apply the returned value elsewhere.

    Args:
        scores: predicted scores, shape ``(n,)``.
        labels: binary ground truth, shape ``(n,)``.
        objective: ``"f1"`` or ``"accuracy"``.

    Returns:
        A tuple ``(threshold, score)``: predictions are ``scores >=
        threshold``, and ``score`` is the achieved value of
        ``objective``.
    """
    if objective not in ("f1", "accuracy"):
        raise ValueError(f"unknown objective: {objective}")
    if scores.numel() == 0:
        return 0.5, 0.0

    thresholds, f1, accuracy = _sweep(scores, labels)
    values = f1 if objective == "f1" else accuracy
    best = int(torch.argmax(values).item())
    return float(thresholds[best].item()), float(values[best].item())


def binary_metrics(
    scores: Tensor, labels: Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Computes accuracy, precision, recall, F1, AUROC, positive rate.

    ``positive_rate`` is reported alongside the rest because every
    consistency metric in :mod:`nspe.consistency` is trivially maximized
    by a model that predicts one class; without it, a degenerate model
    looks like a consistent one.

    Args:
        scores: predicted scores, shape ``(n,)``.
        labels: binary ground truth, shape ``(n,)``.
        threshold: predictions are ``scores >= threshold``.

    Returns:
        A dict with ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``auroc``, and ``positive_rate``.
    """
    predicted = (scores >= threshold).to(labels.dtype)
    true_pos = ((predicted == 1) & (labels == 1)).sum().item()
    false_pos = ((predicted == 1) & (labels == 0)).sum().item()
    false_neg = ((predicted == 0) & (labels == 1)).sum().item()

    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (predicted == labels).float().mean().item(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc(scores, labels),
        "positive_rate": predicted.float().mean().item(),
    }
