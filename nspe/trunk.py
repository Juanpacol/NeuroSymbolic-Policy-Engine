"""Trainable feature stack shared by the symbolic and neural arms.

The H1/H3 comparison only isolates the aggregation step if everything
below it is identical. It previously was not: the baseline was a single
``Linear(1024, 1)`` (1025 parameters) while the reasoner path ran
``Linear(1024, 6)`` (6150 parameters) into a fixed logic circuit, so any
measured difference was confounded by capacity. Mounting both arms on
one trunk of the same shape makes the two differ only in what sits on
top -- a fixed policy versus a learned aggregator over the same latent
predicate vector.

The LayerNorm is load-bearing beyond regularization. A fused CLIP
embedding is two L2-normalized vectors concatenated, so per-element
magnitude is around 0.03; a default-initialized linear layer on top
produces logits with std ~0.025, which puts every predicate at
0.5 +/- 0.01 and makes the whole model a near-constant function at init.
Normalizing restores an O(1) scale at the point where the problem
originates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PredicateTrunk(nn.Module):
    """Maps a fused CLIP embedding to a shared hidden representation.

    Args:
        in_dim: fused CLIP embedding width (``2 * embed_dim``).
        hidden_dim: trunk width. ``0`` makes the trunk an identity,
            reproducing the linear-probe configuration for ablations.
        dropout: dropout probability applied after the nonlinearity.
    """

    def __init__(
        self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = hidden_dim if hidden_dim > 0 else in_dim
        if hidden_dim > 0:
            self.body: nn.Module = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
        else:
            self.body = nn.Identity()

    def forward(self, fused: Tensor) -> Tensor:
        """Runs the trunk.

        Args:
            fused: fused CLIP embeddings, shape ``(batch, in_dim)``.

        Returns:
            Tensor of shape ``(batch, out_dim)``.
        """
        return self.body(fused)


class PredicateHead(nn.Module):
    """Turns a fused embedding into base predicate truth degrees.

    Held separately from :class:`~nspe.extractor.NeuroSymbolicLayer` so
    the predicate computation -- the part where collapse happens and the
    part the anti-collapse machinery targets -- can be constructed and
    tested without downloading a CLIP backbone.

    Args:
        in_dim: fused embedding width (``2 * embed_dim``).
        num_predicates: number of base predicates to emit.
        hidden_dim: shared trunk width; ``0`` for a linear probe.
        dropout: trunk dropout probability.
        mu_eps: half-width of the margin kept away from 0 and 1. The
            reasoner floors truth degrees and ``log1mexp`` clamps its
            input, so a saturating predicate contributes exactly zero
            gradient through every negated occurrence -- which, for
            predicates appearing only under ``unless:``, is their only
            occurrence. A strictly interior range keeps both clamps off
            the live path.
    """

    zero_shot_weight: Tensor

    def __init__(
        self,
        in_dim: int,
        num_predicates: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        mu_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.mu_eps = mu_eps
        self.trunk = PredicateTrunk(in_dim, hidden_dim, dropout)
        self.heads = nn.Linear(self.trunk.out_dim, num_predicates)
        self.logit_scale = nn.Parameter(torch.ones(num_predicates))
        self.logit_bias = nn.Parameter(torch.zeros(num_predicates))
        # Zero-shot residual, filled in from policy descriptions.
        # Registered up front so a checkpoint round-trips either way.
        self.register_buffer("zero_shot_weight", torch.zeros(num_predicates, in_dim))
        self.zs_scale = nn.Parameter(torch.ones(num_predicates))

    def forward(self, fused: Tensor) -> Tensor:
        """Produces predicate truth degrees.

        Args:
            fused: fused embeddings, shape ``(batch, in_dim)``.

        Returns:
            Tensor of shape ``(batch, num_predicates)``, values in
            ``[mu_eps, 1 - mu_eps]``.
        """
        logits = self.heads(self.trunk(fused))
        logits = logits + self.zs_scale * F.linear(fused, self.zero_shot_weight)
        mu = torch.sigmoid(self.logit_scale * logits + self.logit_bias)
        return self.mu_eps + (1.0 - 2.0 * self.mu_eps) * mu
