"""Consistency detection: the H1 mechanism.

Two cases are "predicate-equivalent" if they share the same thresholded
base-predicate activation signature. Within each equivalence class, a
disagreement in verdict is an inconsistency: two near-identical pieces
of content that a deterministic policy should treat the same way.

Note: equivalence classes are defined on the *thresholded* signature,
but the metrics below are meant to be computed against a reasoner (or
baseline) fed the *continuous* ``mu0`` -- not the thresholded signature
itself. A reasoner that is a deterministic function of the thresholded
signature would trivially score a perfect (zero) inconsistency rate,
which is a theorem, not a finding. Feeding continuous inputs while
grouping on the discrete signature keeps the comparison meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nspe.logic.ops import segment_sum


@dataclass
class ConsistencyReport:
    """Consistency metrics for one batch of predicate-equivalent classes.

    Attributes:
        inconsistency_rate: fraction of predicate-equivalent pairs that
            disagree on the (binarized) verdict.
        purity: mean class purity, ``1 - normalized entropy``, weighted
            by class size (``1.0`` = every class is unanimous).
        num_classes: number of distinct equivalence classes found.
        class_id: equivalence class id per case, shape ``(batch,)``.
        class_sizes: number of cases per class, shape ``(num_classes,)``.
        disagree_per_class: disagreeing-pair count per class, shape
            ``(num_classes,)``.
    """

    inconsistency_rate: float
    purity: float
    num_classes: int
    class_id: Tensor
    class_sizes: Tensor
    disagree_per_class: Tensor

    def worst_classes(self, n: int = 5) -> list[dict[str, object]]:
        """Returns the ``n`` most-inconsistent equivalence classes.

        Args:
            n: maximum number of classes to return.

        Returns:
            A list of dicts with ``class_id``, ``size``,
            ``disagreeing_pairs``, and up to 5 ``example_indices`` (row
            indices into the original batch belonging to that class),
            ordered by disagreeing-pair count, descending.
        """
        k = min(n, self.disagree_per_class.numel())
        if k == 0:
            return []
        top = torch.topk(self.disagree_per_class, k=k).indices.tolist()
        results = []
        for c in top:
            members = (self.class_id == c).nonzero(as_tuple=True)[0].tolist()
            results.append(
                {
                    "class_id": c,
                    "size": int(self.class_sizes[c].item()),
                    "disagreeing_pairs": float(self.disagree_per_class[c].item()),
                    "example_indices": members[:5],
                }
            )
        return results


def _signature_classes(mu0: Tensor, tau: float) -> tuple[Tensor, int]:
    signature = (mu0 >= tau).to(mu0.dtype)
    _, class_id = torch.unique(signature, dim=0, return_inverse=True)
    num_classes = int(class_id.max().item()) + 1 if class_id.numel() else 0
    return class_id, num_classes


class ConsistencyChecker(nn.Module):
    """Measures verdict inconsistency across predicate-equivalent cases.

    Args:
        tau: threshold used to binarize ``mu0`` into an activation
            signature that defines equivalence classes.
        verdict_threshold: threshold used to binarize the (continuous)
            verdict truth into a disagree/agree label.
    """

    def __init__(self, tau: float = 0.5, verdict_threshold: float = 0.5) -> None:
        super().__init__()
        self.tau = tau
        self.verdict_threshold = verdict_threshold

    @torch.no_grad()
    def forward(self, mu0: Tensor, verdict: Tensor) -> ConsistencyReport:
        """Computes a :class:`ConsistencyReport` for one batch.

        Args:
            mu0: base predicate truth degrees, shape
                ``(batch, num_base_predicates)``.
            verdict: verdict truth degrees, shape ``(batch,)``.

        Returns:
            The computed :class:`ConsistencyReport`.
        """
        class_id, num_classes = _signature_classes(mu0, self.tau)
        if num_classes == 0:
            empty = torch.zeros(0)
            return ConsistencyReport(0.0, 1.0, 0, class_id, empty, empty)

        label = (verdict >= self.verdict_threshold).to(mu0.dtype)
        n_c = segment_sum(torch.ones_like(label), class_id, num_classes)
        count1 = segment_sum(label, class_id, num_classes)
        count0 = n_c - count1

        disagree_per_class = count0 * count1
        total_pairs = (n_c * (n_c - 1) / 2).sum()
        disagreeing_pairs = disagree_per_class.sum()
        inconsistency_rate = (
            (disagreeing_pairs / total_pairs).item() if total_pairs > 0 else 0.0
        )

        p1 = torch.where(n_c > 0, count1 / n_c.clamp(min=1), torch.zeros_like(n_c))
        p0 = 1.0 - p1
        entropy = -(
            p1 * torch.log2(p1.clamp(min=1e-12))
            + p0 * torch.log2(p0.clamp(min=1e-12))
        )
        class_purity = 1.0 - entropy
        purity = (
            (class_purity * n_c).sum() / n_c.sum()
        ).item() if n_c.sum() > 0 else 1.0

        return ConsistencyReport(
            inconsistency_rate=inconsistency_rate,
            purity=purity,
            num_classes=num_classes,
            class_id=class_id,
            class_sizes=n_c,
            disagree_per_class=disagree_per_class,
        )


def consistency_loss(mu0: Tensor, verdict: Tensor, tau: float = 0.5) -> Tensor:
    """Differentiable training signal: within-class verdict variance.

    Equivalence classes are computed with no gradient (they are a
    discrete grouping), but the variance itself is computed over the
    continuous ``verdict`` tensor, so gradients flow back through
    ``verdict`` to whatever produced it.

    Args:
        mu0: base predicate truth degrees, shape
            ``(batch, num_base_predicates)``.
        verdict: verdict truth degrees, shape ``(batch,)``, with
            ``requires_grad`` if this is used as a training loss.
        tau: threshold used to binarize ``mu0`` into equivalence classes.

    Returns:
        Scalar tensor: the size-weighted mean within-class variance of
        ``verdict``.
    """
    with torch.no_grad():
        class_id, num_classes = _signature_classes(mu0, tau)
    if num_classes == 0:
        return verdict.sum() * 0.0

    n_c = segment_sum(torch.ones_like(verdict), class_id, num_classes)
    sum_c = segment_sum(verdict, class_id, num_classes)
    sumsq_c = segment_sum(verdict * verdict, class_id, num_classes)
    mean_c = sum_c / n_c.clamp(min=1)
    var_c = (sumsq_c / n_c.clamp(min=1)) - mean_c * mean_c
    return (var_c * n_c).sum() / n_c.sum().clamp(min=1)
