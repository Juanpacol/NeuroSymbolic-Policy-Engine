"""Per-predicate and signature-structure diagnostics for an eval run.

The ``num_classes = 5`` collapse that invalidated a full evaluation was
only noticed because the number happened to be printed. These functions
put the underlying structure -- how often each predicate fires, how
correlated the predicates are, which signatures actually occur -- into
every result file, so the same failure is visible rather than inferred.

They are also the instruments for interpreting a negative H1 result. A
reasoner that looks less consistent than the baseline means something
quite different depending on whether its predicates have collapsed into
copies of one another or are genuinely firing independently, and these
are what tell those apart.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def predicate_stats(
    mu0: Tensor, names: tuple[str, ...], tau: float = 0.5
) -> dict[str, dict[str, float]]:
    """Summarizes each predicate's behaviour over a split.

    Args:
        mu0: base predicate truth degrees, shape ``(batch, P)``.
        names: predicate names, length ``P``.
        tau: threshold defining "fired".

    Returns:
        A dict keyed by predicate name, each holding ``mean``, ``std``,
        ``activation_rate``, and ``max_abs_correlation`` against any
        other predicate. An activation rate near 0 or 1 means the
        predicate never changes a signature; a high correlation means it
        duplicates another predicate.
    """
    num_predicates = mu0.shape[-1]
    activation = (mu0 >= tau).to(mu0.dtype)

    if num_predicates > 1 and mu0.shape[0] > 1:
        centered = mu0 - mu0.mean(dim=0, keepdim=True)
        std = centered.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
        correlation = (centered.T @ centered) / mu0.shape[0] / torch.outer(std, std)
        correlation = correlation - torch.diag(torch.diagonal(correlation))
        max_correlation = correlation.abs().max(dim=1).values
    else:
        max_correlation = torch.zeros(num_predicates)

    return {
        name: {
            "mean": mu0[:, i].mean().item(),
            "std": mu0[:, i].std(unbiased=False).item(),
            "activation_rate": activation[:, i].mean().item(),
            "max_abs_correlation": max_correlation[i].item(),
        }
        for i, name in enumerate(names)
    }


def signature_distribution(
    mu0: Tensor, names: tuple[str, ...], tau: float = 0.5, top_k: int | None = None
) -> list[dict[str, Any]]:
    """Lists the distinct activation signatures and how often each occurs.

    Args:
        mu0: base predicate truth degrees, shape ``(batch, P)``.
        names: predicate names, length ``P``.
        tau: threshold defining "fired".
        top_k: return only the ``top_k`` most frequent signatures, or
            ``None`` for all.

    Returns:
        A list of dicts with ``signature`` (the thresholded bit vector),
        ``active`` (names of the predicates that fired), ``count``, and
        ``share``, ordered by count descending.
    """
    signature = (mu0 >= tau).to(torch.int64)
    unique, counts = torch.unique(signature, dim=0, return_counts=True)
    order = torch.argsort(counts, descending=True)
    total = float(signature.shape[0])

    rows = []
    for index in order.tolist():
        bits = unique[index].tolist()
        rows.append(
            {
                "signature": bits,
                # strict: a name list that does not match the signature
                # width would silently mislabel or truncate the active
                # predicates, which is the whole content of this row.
                "active": [n for n, b in zip(names, bits, strict=True) if b],
                "count": int(counts[index].item()),
                "share": counts[index].item() / total,
            }
        )
    return rows if top_k is None else rows[:top_k]
