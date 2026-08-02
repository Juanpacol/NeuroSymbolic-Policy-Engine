"""Monotone calibration from a fuzzy truth degree to a probability.

A fuzzy verdict and a probability are different objects, and training
conflated them. The product t-norm compresses a verdict standing behind
a conjunction of several literals into a narrow band -- for the Hateful
Memes policy, uniform ``mu0 = 0.5`` inputs yield a verdict of 0.139, and
the reachable range is roughly ``[3e-7, 0.9985]``. Feeding that directly
into binary cross-entropy against 0/1 labels makes "raise the constant
toward the base rate" the cheapest descent direction, so the model
learns to be a constant before it learns to discriminate.

This module maps the truth degree into log-odds space, applies a
strictly increasing affine transform, and maps back. Because the
transform is strictly monotone, it cannot change the ranking of any two
cases, so **AUROC is mathematically invariant to calibration** -- it
cannot manufacture ranking quality, only relocate the operating point.
It adds two parameters and applies identically to the symbolic and
neural arms. The raw ``mu``/``verdicts`` an audit chain explains are
left untouched; only the quantity compared against a binary label is
calibrated.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VerdictCalibrator(nn.Module):
    """Maps a truth degree in ``(0, 1)`` to a calibrated probability.

    Args:
        init_scale: initial gain applied in log-odds space. Must be
            positive.
        init_bias: initial log-odds offset. Prefer setting this from the
            training base rate via :meth:`fit_bias_to_base_rate` rather
            than guessing.
        eps: floor applied before the logit, keeping the map finite at
            the boundaries of the reachable verdict range.
        enabled: when ``False``, :meth:`forward` is the identity. Used to
            observe the uncalibrated verdict distribution before fitting
            the bias to it.
    """

    def __init__(
        self,
        init_scale: float = 1.0,
        init_bias: float = 0.0,
        eps: float = 1e-6,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}")
        # softplus inverse, so scale() starts exactly at init_scale.
        self.raw_scale = nn.Parameter(torch.tensor(math.log(math.expm1(init_scale))))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))
        self.eps = eps
        self.enabled = enabled

    def scale(self) -> Tensor:
        """Returns the strictly positive log-odds gain."""
        return F.softplus(self.raw_scale)

    def forward(self, verdict: Tensor) -> Tensor:
        """Calibrates a batch of truth degrees.

        Args:
            verdict: truth degrees in ``(0, 1)``, shape ``(batch,)``.

        Returns:
            Calibrated probabilities, shape ``(batch,)``. Strictly
            increasing in ``verdict``, or ``verdict`` unchanged when
            ``self.enabled`` is ``False``.
        """
        if not self.enabled:
            return verdict
        clamped = verdict.clamp(self.eps, 1.0 - self.eps)
        return torch.sigmoid(self.scale() * torch.logit(clamped) + self.bias)

    @torch.no_grad()
    def fit_bias_to_base_rate(self, verdict: Tensor, base_rate: float) -> None:
        """Sets ``bias`` so the mean calibrated probability is ``base_rate``.

        Starting the model already at the label base rate removes the
        degenerate descent direction that otherwise dominates the first
        epochs.

        Args:
            verdict: truth degrees from an untrained forward pass over
                the training split, shape ``(n,)``. Accepted on any
                device: callers accumulate this across batches and
                often move it to CPU to do so, while the calibrator
                itself may be on GPU.
            base_rate: fraction of positive labels in that split.
        """
        verdict = verdict.to(self.bias.device)
        clamped = verdict.clamp(self.eps, 1.0 - self.eps)
        logits = self.scale() * torch.logit(clamped)
        target = math.log(base_rate / (1.0 - base_rate))
        self.bias.fill_(target - logits.mean().item())
