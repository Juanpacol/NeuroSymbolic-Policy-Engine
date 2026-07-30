"""Numerically stable primitives for log-space fuzzy logic.

All truth degrees are represented as ``log_x = log(x)`` for ``x`` in
``(0, 1]``, so ``log_x`` is in ``(-inf, 0]``. Working in log space turns
the product t-norm's conjunction into a plain sum and avoids underflow on
long rule bodies (see logLTN, arXiv:2306.14546).
"""

from __future__ import annotations

import torch
from torch import Tensor

_LOG_2 = 0.6931471805599453


def safe_log(x: Tensor, eps: float = 1e-7) -> Tensor:
    """Returns ``log(x)`` with ``x`` clamped to ``[eps, 1]``.

    Args:
        x: tensor of truth degrees in ``[0, 1]``.
        eps: lower clamp bound, avoids ``log(0) = -inf``.

    Returns:
        Tensor of the same shape as ``x``, in ``(-inf, 0]``.
    """
    return torch.log(x.clamp(min=eps, max=1.0))


def log1mexp(log_x: Tensor) -> Tensor:
    """Returns ``log(1 - exp(log_x))`` for ``log_x <= 0``, stably.

    This is exactly negation in log space: if ``log_x = log(x)`` then
    ``log1mexp(log_x) = log(1 - x)``. Uses Maechler's two-branch formula
    to avoid catastrophic cancellation near both ``log_x = 0`` and
    ``log_x = -inf``.

    Args:
        log_x: tensor of log-truths, values must be `<= 0`.

    Returns:
        Tensor of the same shape as ``log_x``, in ``(-inf, 0]``.
    """
    log_x = log_x.clamp(max=-1e-12)
    small = log_x > -_LOG_2
    out = torch.empty_like(log_x)
    out = torch.where(
        small,
        torch.log(-torch.expm1(log_x)),
        torch.log1p(-torch.exp(log_x)),
    )
    return out


def segment_sum(
    values: Tensor, index: Tensor, num_segments: int, dim: int = -1
) -> Tensor:
    """Sums ``values`` along ``dim`` into ``num_segments`` buckets.

    Args:
        values: tensor whose size along ``dim`` equals ``index.shape[0]``.
        index: 1D tensor of segment ids in ``[0, num_segments)``.
        num_segments: total number of output segments.
        dim: dimension along which to segment-reduce.

    Returns:
        Tensor equal to ``values`` except size ``num_segments`` along
        ``dim``, holding the per-segment sum. Empty segments are ``0``.
    """
    shape = list(values.shape)
    shape[dim] = num_segments
    out = values.new_zeros(shape)
    out.index_add_(dim, index, values)
    return out


def segment_amax(
    values: Tensor, index: Tensor, num_segments: int, fill: float = float("-inf")
) -> Tensor:
    """Segment-wise max over the last dimension of ``values``.

    Args:
        values: tensor of shape ``(..., nnz)``.
        index: 1D tensor of shape ``(nnz,)`` with segment ids.
        num_segments: total number of output segments.
        fill: value used for segments with no contributing elements.

    Returns:
        Tensor of shape ``(..., num_segments)``.
    """
    shape = values.shape[:-1] + (num_segments,)
    out = values.new_full(shape, fill)
    idx = index.expand(values.shape)
    out.scatter_reduce_(-1, idx, values, reduce="amax", include_self=True)
    return out


def segment_amin(
    values: Tensor, index: Tensor, num_segments: int, fill: float = float("inf")
) -> Tensor:
    """Segment-wise min over the last dimension of ``values``.

    Implemented as ``-segment_amax(-values, ...)`` since PyTorch's MPS
    backend does not implement an ``amin`` scatter_reduce as reliably as
    ``amax``.

    Args:
        values: tensor of shape ``(..., nnz)``.
        index: 1D tensor of shape ``(nnz,)`` with segment ids.
        num_segments: total number of output segments.
        fill: value used for segments with no contributing elements.

    Returns:
        Tensor of shape ``(..., num_segments)``.
    """
    return -segment_amax(-values, index, num_segments, fill=-fill)


def gather_literal_log(log_mu: Tensor, index: Tensor, sign: Tensor) -> Tensor:
    """Gathers per-literal log-truths, applying negation for ``sign < 0``.

    Args:
        log_mu: log-truths, shape ``(batch, num_predicates)``.
        index: predicate index per literal, shape ``(nnz,)``.
        sign: `+1` for a positive literal, `-1` for a negated one, shape
            ``(nnz,)``.

    Returns:
        Tensor of shape ``(batch, nnz)`` with the (possibly negated)
        log-truth of each literal.
    """
    gathered = log_mu.index_select(-1, index)
    negated = log1mexp(gathered)
    return torch.where(sign.unsqueeze(0) > 0, gathered, negated)
