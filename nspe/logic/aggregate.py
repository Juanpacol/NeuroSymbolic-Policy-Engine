"""Generalized-mean aggregation, as an alternative to t-conorm disjunction.

Following LTN's "stable product configuration", aggregating many rules
that conclude the same head with a plain t-conorm can saturate to 1 as
the number of contributing rules grows. A tunable p-mean gives a softer
aggregation: ``p=1`` recovers a (non-t-conorm) arithmetic mean, and
``p`` large approaches ``max``. Exposed as an opt-in alternative to
:meth:`nspe.logic.tnorm.TNorm.disj_segment`, not the default reasoner
path.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nspe.logic.ops import segment_sum


def p_mean_segment(
    log_x: Tensor, index: Tensor, num_segments: int, p: float = 2.0, eps: float = 1e-7
) -> Tensor:
    """Segment-wise generalized (power) mean, returned in log space.

    Args:
        log_x: literal/rule log-truths, shape ``(batch, nnz)``.
        index: segment id per element, shape ``(nnz,)``.
        num_segments: number of output segments.
        p: the power of the generalized mean. ``p=1`` is the arithmetic
            mean; larger ``p`` approaches ``max``.
        eps: numerical floor before taking logs/powers.

    Returns:
        Tensor of shape ``(batch, num_segments)``, in log space.
    """
    x = torch.exp(log_x).clamp(min=eps)
    powered = x.pow(p)
    count = segment_sum(torch.ones_like(x), index, num_segments).clamp(min=1.0)
    total = segment_sum(powered, index, num_segments)
    mean = (total / count).clamp(min=eps, max=1.0)
    return mean.pow(1.0 / p).log()
