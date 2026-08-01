"""Auxiliary losses that keep the predicate layer from collapsing.

The predicate layer receives gradient from a single scalar verdict and
has no per-predicate labels to learn from -- Hateful Memes provides only
a binary hateful/not-hateful annotation. Nothing in the primary
objective distinguishes "six predicates that each detect something" from
"six copies of one label predictor", and a run of the unregularized
model produced only 5 distinct thresholded signatures across 831 cases
out of 64 possible.

These three terms attack that from different directions: ``anchor_loss``
supplies the missing per-predicate signal from the policy's own text,
``decorrelation_loss`` penalizes heads that duplicate each other, and
``activation_entropy_loss`` penalizes heads that carry no information
because they are always on or always off.
"""

from __future__ import annotations

import torch
from torch import Tensor


def decorrelation_loss(mu0: Tensor) -> Tensor:
    """Penalizes predicates that behave as copies of one another.

    A predicate perfectly correlated with another contributes no
    additional bit to the activation signature that defines
    predicate-equivalence classes, so a collapsed layer produces
    artificially few classes and makes the H1 metric meaningless.

    Args:
        mu0: base predicate truth degrees, shape ``(batch, P)``.

    Returns:
        Scalar tensor: mean squared off-diagonal correlation, in
        ``[0, 1]``. Zero when predicates are uncorrelated across the
        batch.
    """
    num_predicates = mu0.shape[-1]
    if num_predicates < 2 or mu0.shape[0] < 2:
        return mu0.sum() * 0.0

    centered = mu0 - mu0.mean(dim=0, keepdim=True)
    std = centered.pow(2).mean(dim=0).sqrt()
    normalized = centered / std.clamp(min=1e-6)
    correlation = (normalized.T @ normalized) / mu0.shape[0]

    off_diagonal = ~torch.eye(num_predicates, dtype=torch.bool, device=mu0.device)
    return correlation[off_diagonal].pow(2).mean()


def activation_entropy_loss(mu0: Tensor, target_rate: float = 0.5) -> Tensor:
    """Pushes each predicate's mean activation toward ``target_rate``.

    A predicate that is always on or always off never changes an
    equivalence signature, so it is dead weight in the symbolic layer no
    matter what the verdict loss says. Penalizing the batch mean rather
    than per-case values leaves individual predictions free.

    Args:
        mu0: base predicate truth degrees, shape ``(batch, P)``.
        target_rate: activation rate each predicate is pulled toward.

    Returns:
        Scalar tensor, minimized when every predicate's batch mean
        equals ``target_rate``.
    """
    mean_activation = mu0.mean(dim=0).clamp(1e-6, 1 - 1e-6)
    target = torch.full_like(mean_activation, target_rate)
    return torch.nn.functional.binary_cross_entropy(mean_activation, target)


def anchor_loss(mu0: Tensor, zero_shot: Tensor) -> Tensor:
    """Keeps each predicate near its CLIP zero-shot description score.

    ``zero_shot`` is the frozen-CLIP similarity between each case and
    each predicate's natural-language definition from the policy,
    rescaled to ``(0, 1)``. This is the only per-predicate supervision
    available, and it grounds the predicate layer in what the policy
    actually says rather than in whatever direction the verdict gradient
    happens to find. Weight it moderately: pushed too hard, the layer
    degenerates into CLIP zero-shot classification and learns nothing.

    Args:
        mu0: base predicate truth degrees, shape ``(batch, P)``.
        zero_shot: target truth degrees in ``(0, 1)``, same shape.

    Returns:
        Scalar tensor: mean squared deviation from the zero-shot target.
    """
    return (mu0 - zero_shot).pow(2).mean()


def zero_shot_targets(
    fused: Tensor, zero_shot_weight: Tensor, gain: float = 1.0
) -> Tensor:
    """Turns CLIP description similarities into anchor targets.

    Similarities are standardized per predicate across the batch before
    the sigmoid, so the targets say "this case scores high or low for
    this predicate relative to its peers" rather than depending on
    CLIP's absolute similarity scale. Raw cosine similarities occupy a
    narrow, backbone-specific band -- a fixed temperature tuned for one
    backbone saturates to hard 0/1 on another, which would turn a soft
    anchor into a hard pseudo-label.

    Args:
        fused: fused CLIP embeddings, shape ``(batch, 2 * embed_dim)``.
        zero_shot_weight: per-predicate description embeddings, shape
            ``(P, 2 * embed_dim)``, as held by
            :class:`~nspe.extractor.NeuroSymbolicLayer`.
        gain: multiplier on the standardized score. Larger values make
            the targets more confident.

    Returns:
        Tensor of shape ``(batch, P)`` in ``(0, 1)``, detached -- these
        are targets, not a path for gradient. Predicates with an unset
        (all-zero) description row get a flat ``0.5``.
    """
    similarity = torch.nn.functional.linear(fused, zero_shot_weight)
    centered = similarity - similarity.mean(dim=0, keepdim=True)
    std = centered.pow(2).mean(dim=0, keepdim=True).sqrt()
    # An unset description row yields a constant column; std ~ 0 there,
    # and the clamp collapses it to a flat 0.5 target rather than noise.
    return torch.sigmoid(gain * centered / std.clamp(min=1e-6)).detach()
