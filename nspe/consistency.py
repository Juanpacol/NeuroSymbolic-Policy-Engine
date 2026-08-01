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

A second, sharper degeneracy needs handling explicitly: **a model that
predicts one class for every case is perfectly consistent**. It has
``inconsistency_rate = 0`` and ``purity = 1`` while carrying no
information whatsoever. This is not hypothetical -- a measured run had
the neural baseline at F1 0.21, close enough to constant that its
better raw consistency score was substantially an artifact of not
discriminating rather than evidence of better reasoning.

So the raw rate is never reported alone. ``positive_rate`` exposes how
close a model is to constant; ``null_inconsistency`` gives the rate
expected from a random model with the same marginal, and
``adjusted_consistency`` is the chance-corrected score built from it. A
model that predicts a single class is flagged ``degenerate`` and is
disqualified from the comparison rather than winning it.
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
            disagree on the (binarized) verdict. Never interpret this
            without ``positive_rate``: a constant model scores ``0.0``.
        purity: mean class purity, ``1 - normalized entropy``, weighted
            by class size (``1.0`` = every class is unanimous). Subject
            to the same degeneracy as ``inconsistency_rate``.
        num_classes: number of distinct equivalence classes found.
        positive_rate: fraction of cases the model called positive. At
            ``0.0`` or ``1.0`` the model is constant and its consistency
            scores are vacuous.
        null_inconsistency: inconsistency expected from a random model
            with this same ``positive_rate`` and these class sizes.
        adjusted_consistency: ``1 - inconsistency_rate /
            null_inconsistency``. ``1.0`` is perfectly consistent given
            the marginal, ``0.0`` is chance, negative is worse than
            chance. ``nan`` when the model is degenerate.
        degenerate: ``True`` when the model predicted a single class, in
            which case the consistency scores carry no information.
        signature_entropy: Shannon entropy in bits of the equivalence
            class distribution. Low values mean the predicate layer has
            collapsed and the classes are too coarse to be meaningful.
        class_id: equivalence class id per case, shape ``(batch,)``.
        class_sizes: number of cases per class, shape ``(num_classes,)``.
        disagree_per_class: disagreeing-pair count per class, shape
            ``(num_classes,)``.
    """

    inconsistency_rate: float
    purity: float
    num_classes: int
    positive_rate: float
    null_inconsistency: float
    adjusted_consistency: float
    degenerate: bool
    signature_entropy: float
    class_id: Tensor
    class_sizes: Tensor
    disagree_per_class: Tensor

    def class_size_histogram(self) -> dict[int, int]:
        """Returns a mapping from class size to how many classes have it.

        Makes collapse visible directly: a handful of huge classes is
        the signature of a predicate layer that stopped discriminating.
        """
        histogram: dict[int, int] = {}
        for size in self.class_sizes.tolist():
            histogram[int(size)] = histogram.get(int(size), 0) + 1
        return dict(sorted(histogram.items()))

    def as_dict(self) -> dict[str, object]:
        """Returns the scalar metrics, for JSON reporting."""
        return {
            "inconsistency_rate": self.inconsistency_rate,
            "purity": self.purity,
            "num_classes": self.num_classes,
            "positive_rate": self.positive_rate,
            "null_inconsistency": self.null_inconsistency,
            "adjusted_consistency": self.adjusted_consistency,
            "degenerate": self.degenerate,
            "signature_entropy": self.signature_entropy,
            "class_size_histogram": self.class_size_histogram(),
        }

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


def _null_inconsistency(
    class_sizes: Tensor, num_cases: int, num_positive: float
) -> float:
    """Inconsistency expected from a random model with the same marginal.

    Under a null that shuffles the verdict labels across cases while
    holding both the number of positives and the class sizes fixed, any
    particular pair disagrees with probability
    ``2 * k * (n - k) / (n * (n - 1))`` for ``k`` positives among ``n``
    cases -- sampling two labels without replacement. That value does
    not depend on which class the pair is in, so it is also the expected
    overall rate.

    This is what makes the raw inconsistency rate interpretable. A model
    predicting one class has ``k`` of ``0`` or ``n``, hence an expected
    rate of zero, which is exactly why its perfect observed score means
    nothing.

    Args:
        class_sizes: cases per equivalence class, shape
            ``(num_classes,)``. Unused in the closed form, kept in the
            signature because the quantity is only well-defined
            alongside a fixed grouping.
        num_cases: total number of cases.
        num_positive: number of cases predicted positive.

    Returns:
        The expected disagreeing-pair fraction under the null.
    """
    del class_sizes
    if num_cases < 2:
        return 0.0
    k, n = float(num_positive), float(num_cases)
    return 2.0 * k * (n - k) / (n * (n - 1.0))


def permutation_null_inconsistency(
    class_id: Tensor,
    num_classes: int,
    verdict_label: Tensor,
    num_permutations: int = 1000,
    seed: int = 0,
) -> float:
    """Estimates the null inconsistency by shuffling labels directly.

    Validates the closed form in :func:`_null_inconsistency` rather than
    replacing it -- the analytic value is exact, and the two agreeing is
    a useful check that the grouping code does what the derivation
    assumes.

    Args:
        class_id: equivalence class id per case, shape ``(batch,)``.
        num_classes: number of distinct classes.
        verdict_label: binarized verdicts, shape ``(batch,)``.
        num_permutations: how many shuffles to average over.
        seed: RNG seed, so the estimate is reproducible.

    Returns:
        The mean disagreeing-pair fraction across permutations.
    """
    generator = torch.Generator(device=verdict_label.device).manual_seed(seed)
    n_c = segment_sum(torch.ones_like(verdict_label), class_id, num_classes)
    total_pairs = (n_c * (n_c - 1) / 2).sum()
    if total_pairs <= 0:
        return 0.0

    total = 0.0
    for _ in range(num_permutations):
        order = torch.randperm(
            verdict_label.numel(), generator=generator, device=verdict_label.device
        )
        shuffled = verdict_label[order]
        count1 = segment_sum(shuffled, class_id, num_classes)
        total += ((n_c - count1) * count1).sum().item()
    return total / num_permutations / total_pairs.item()


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
            return ConsistencyReport(
                inconsistency_rate=0.0,
                purity=1.0,
                num_classes=0,
                positive_rate=0.0,
                null_inconsistency=0.0,
                adjusted_consistency=float("nan"),
                degenerate=True,
                signature_entropy=0.0,
                class_id=class_id,
                class_sizes=empty,
                disagree_per_class=empty,
            )

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
            p1 * torch.log2(p1.clamp(min=1e-12)) + p0 * torch.log2(p0.clamp(min=1e-12))
        )
        class_purity = 1.0 - entropy
        purity = (
            ((class_purity * n_c).sum() / n_c.sum()).item() if n_c.sum() > 0 else 1.0
        )

        positive_rate = label.mean().item()
        null_inconsistency = _null_inconsistency(
            n_c, int(label.numel()), float(label.sum().item())
        )
        degenerate = positive_rate in (0.0, 1.0)
        adjusted_consistency = (
            float("nan")
            if degenerate or null_inconsistency <= 0.0
            else 1.0 - inconsistency_rate / null_inconsistency
        )

        share = n_c / n_c.sum().clamp(min=1)
        signature_entropy = -(share * torch.log2(share.clamp(min=1e-12))).sum().item()

        return ConsistencyReport(
            inconsistency_rate=inconsistency_rate,
            purity=purity,
            num_classes=num_classes,
            positive_rate=positive_rate,
            null_inconsistency=null_inconsistency,
            adjusted_consistency=adjusted_consistency,
            degenerate=degenerate,
            signature_entropy=signature_entropy,
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
