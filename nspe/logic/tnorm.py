"""Pluggable t-norm families for the fuzzy reasoner.

Default is the product t-norm, computed in log space. Van Krieken et al.
("Analyzing Differentiable Fuzzy Logic Operators", arXiv:2002.06100) show
it is the only one of the three classical families with no vanishing
derivative: Goedel's min/max routes gradient to a single literal per
conjunction, and Lukasiewicz saturates to zero gradient outside its
support. Both are kept here as pluggable alternatives, for ablations and
as a crisp-logic reference used in correctness tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from nspe.logic.ops import log1mexp, segment_amax, segment_amin, segment_sum


class TNorm(ABC):
    """A fuzzy logic operator family, expressed entirely in log space.

    All methods take and return log-truths (``log(x)`` for ``x`` in
    ``(0, 1]``), so callers never need to exponentiate.
    """

    name: str

    @abstractmethod
    def conj_segment(self, log_x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """Segment-wise AND (rule-body conjunction).

        Args:
            log_x: literal log-truths, shape ``(batch, nnz)``.
            index: segment id per literal (e.g. rule id), shape ``(nnz,)``.
            num_segments: number of segments (e.g. number of rules).

        Returns:
            Tensor of shape ``(batch, num_segments)``.
        """

    @abstractmethod
    def disj_segment(self, log_x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """Segment-wise OR (aggregating rules that share a head).

        Args:
            log_x: per-rule log-truths, shape ``(batch, nnz)``.
            index: segment id per rule (e.g. head predicate id).
            num_segments: number of segments (e.g. number of predicates).

        Returns:
            Tensor of shape ``(batch, num_segments)``.
        """

    @abstractmethod
    def disj_pair(self, log_a: Tensor, log_b: Tensor) -> Tensor:
        """Elementwise binary OR, used for the monotone fixpoint update."""

    def neg(self, log_x: Tensor) -> Tensor:
        """Standard negation: ``log(1 - x)``, shared by all families."""
        return log1mexp(log_x)

class ProductTNorm(TNorm):
    """Product t-norm / t-conorm, computed in log space (the default)."""

    name = "product"

    def conj_segment(self, log_x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """See `TNorm.conj_segment`."""
        return segment_sum(log_x, index, num_segments)

    def disj_segment(self, log_x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """See `TNorm.disj_segment`."""
        neg_sum = segment_sum(log1mexp(log_x), index, num_segments)
        return log1mexp(neg_sum)

    def disj_pair(self, log_a: Tensor, log_b: Tensor) -> Tensor:
        """See `TNorm.disj_pair`."""
        return log1mexp(log1mexp(log_a) + log1mexp(log_b))

class GodelTNorm(TNorm):
    """Goedel (min/max) t-norm / t-conorm.

    Non-differentiable at ties and routes gradient to a single literal
    per conjunction/disjunction. Kept for ablation and as a semantics
    option, not recommended as the training default.
    """

    name = "godel"

    def conj_segment(self, log_x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """See `TNorm.conj_segment`."""
        return segment_amin(log_x, index, num_segments, fill=0.0)

    def disj_segment(self, log_x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """See `TNorm.disj_segment`."""
        return segment_amax(log_x, index, num_segments, fill=float("-inf"))

    def disj_pair(self, log_a: Tensor, log_b: Tensor) -> Tensor:
        """See `TNorm.disj_pair`."""
        return torch.maximum(log_a, log_b)

class LukasiewiczTNorm(TNorm):
    """Lukasiewicz t-norm / t-conorm.

    Its derivative is exactly zero outside the operator's support, so a
    conjunction of more than a couple of literals is commonly dead
    (zero-gradient) at initialization. Kept for ablation only.
    """

    name = "lukasiewicz"

    def conj_segment(self, log_x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """See `TNorm.conj_segment`."""
        x = torch.exp(log_x)
        count = segment_sum(torch.ones_like(x), index, num_segments)
        total = segment_sum(x, index, num_segments)
        linear = (total - count + 1.0).clamp(min=1e-7, max=1.0)
        return torch.log(linear)

    def disj_segment(self, log_x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """See `TNorm.disj_segment`."""
        x = torch.exp(log_x)
        total = segment_sum(x, index, num_segments).clamp(min=1e-7, max=1.0)
        return torch.log(total)

    def disj_pair(self, log_a: Tensor, log_b: Tensor) -> Tensor:
        """See `TNorm.disj_pair`."""
        linear = (torch.exp(log_a) + torch.exp(log_b)).clamp(min=1e-7, max=1.0)
        return torch.log(linear)

class CrispTNorm(TNorm):
    """Hard boolean logic (threshold at 0.5), used as a correctness oracle.

    Not differentiable in any meaningful sense; exists so the fuzzy
    reasoner can be checked against classical stratified-Datalog
    semantics on crisp (0/1) inputs.
    """

    name = "crisp"
    threshold: float = 0.5

    def _harden(self, log_x: Tensor) -> Tensor:
        x = torch.exp(log_x)
        hard = (x >= self.threshold).to(log_x.dtype)
        return torch.log(hard.clamp(min=1e-7, max=1.0))

    def conj_segment(self, log_x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """Hardened Goedel-style segment AND; see `TNorm.conj_segment`."""
        return segment_amin(self._harden(log_x), index, num_segments, fill=0.0)

    def disj_segment(self, log_x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """Hardened Goedel-style segment OR; see `TNorm.disj_segment`."""
        return segment_amax(
            self._harden(log_x), index, num_segments, fill=float("-inf")
        )

    def disj_pair(self, log_a: Tensor, log_b: Tensor) -> Tensor:
        """Hardened elementwise OR; see `TNorm.disj_pair`."""
        return torch.maximum(self._harden(log_a), self._harden(log_b))


_REGISTRY: dict[str, type[TNorm]] = {
    "product": ProductTNorm,
    "godel": GodelTNorm,
    "lukasiewicz": LukasiewiczTNorm,
    "crisp": CrispTNorm,
}


def get_tnorm(name: str) -> TNorm:
    """Looks up a t-norm implementation by name.

    Args:
        name: one of ``"product"``, ``"godel"``, ``"lukasiewicz"``, or
            ``"crisp"``.

    Returns:
        A new instance of the requested :class:`TNorm`.

    Raises:
        KeyError: if ``name`` is not a registered t-norm.
    """
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown t-norm {name!r}. Known: {known}") from exc
