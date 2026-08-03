"""Rule -> predicate -> verdict explanation extraction.

Explanations are read off the tensors a :class:`~nspe.reasoner.PolicyKGReasoner`
forward pass already produced (``fire_trace``, ``mu_trace``): no second,
separate symbolic inference pass is needed. This module is pure
``@torch.no_grad()`` post-processing of a completed
:class:`~nspe.reasoner.ReasonerOutput`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from nspe.reasoner import PolicyKGReasoner, ReasonerOutput


@dataclass
class ExplanationNode:
    """One node in a rule -> predicate -> verdict explanation tree.

    Attributes:
        predicate: predicate name at this node.
        truth: this predicate's truth degree for the explained case.
        kind: ``"base"``, ``"derived"``, or ``"verdict"``.
        negated: whether this node is used in negated form by its
            parent rule's body.
        rule_id: id of the rule that concluded this predicate, or
            ``None`` for a leaf (base predicate, or no rule fired).
        confidence: the concluding rule's confidence, or ``None``.
        children: body literals of the concluding rule.
        defeated_by: ``(exception_predicate, effective_truth)`` pairs for
            every exception literal on the concluding rule, regardless
            of whether it actually defeated the rule.
    """

    predicate: str
    truth: float
    kind: str
    negated: bool = False
    rule_id: str | None = None
    confidence: float | None = None
    children: list[ExplanationNode] = field(default_factory=list)
    defeated_by: list[tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Returns this node and its subtree as a JSON-ready dict.

        The rendered text is for a human reading a results file;
        this is for anything that has to *process* the chain -- building
        a figure, counting which rules fire, checking that a cited rule
        exists in the policy. Recovering any of that by re-parsing
        :meth:`render`'s indentation would be needless and fragile.

        Returns:
            A dict mirroring this node's fields, with ``children``
            recursed and ``defeated_by`` pairs expanded into named
            entries.
        """
        return {
            "predicate": self.predicate,
            "truth": self.truth,
            "kind": self.kind,
            "negated": self.negated,
            "rule_id": self.rule_id,
            "confidence": self.confidence,
            "defeated_by": [
                {"predicate": name, "effective_truth": truth}
                for name, truth in self.defeated_by
            ],
            "children": [child.as_dict() for child in self.children],
        }

    def render(self, indent: int = 0) -> str:
        """Renders this node (and its subtree) as indented text."""
        pad = "  " * indent
        label = f"NOT {self.predicate}" if self.negated else self.predicate
        line = f"{pad}{label} = {self.truth:.4f}"
        if self.rule_id is not None:
            line += f"  (via {self.rule_id}, conf={self.confidence:.2f})"
        elif not self.children:
            line += f"  [{self.kind}]"
        lines = [line]
        for child in self.children:
            lines.append(child.render(indent + 1))
        for name, truth in self.defeated_by:
            lines.append(f"{pad}  defeated_by: {name} = {truth:.4f}")
        return "\n".join(lines)


@dataclass
class Explanation:
    """A full explanation for one predicate's truth, for one case.

    Attributes:
        verdict: the explained predicate's name.
        case_index: which row of the batch this explains.
        truth: the explained predicate's truth degree.
        root: the explanation tree.
        policy_fingerprint: fingerprint of the policy this was produced
            from, for reproducible audits.
    """

    verdict: str
    case_index: int
    truth: float
    root: ExplanationNode
    policy_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        """Returns the full explanation as a JSON-ready dict.

        Carries ``policy_fingerprint`` alongside the tree so an audit
        can confirm which compiled rule set produced the chain, rather
        than trusting that the surrounding file was not reassembled.

        Returns:
            A dict with ``verdict``, ``case_index``, ``truth``,
            ``policy_fingerprint`` and the nested ``root``.
        """
        return {
            "verdict": self.verdict,
            "case_index": self.case_index,
            "truth": self.truth,
            "policy_fingerprint": self.policy_fingerprint,
            "root": self.root.as_dict(),
        }

    def render(self) -> str:
        """Renders the full explanation tree as indented text."""
        return self.root.render()


@torch.no_grad()
def explain(
    reasoner: PolicyKGReasoner,
    out: ReasonerOutput,
    targets: Sequence[str] | None = None,
    case_index: int = 0,
) -> list[Explanation]:
    """Builds explanations for one case from a reasoner output.

    Args:
        reasoner: the reasoner that produced ``out`` (needed for its
            compiled rule buffers).
        out: output of a forward pass run with ``store_trace=True``.
        targets: predicate names to explain; defaults to every verdict
            predicate in the policy.
        case_index: which row of the batch to explain.

    Returns:
        One :class:`Explanation` per requested target.

    Raises:
        ValueError: if ``out`` was produced with ``store_trace=False``.
    """
    if out.fire_trace is None or out.mu_trace is None:
        raise ValueError(
            "explain() requires a ReasonerOutput produced with store_trace=True"
        )
    rt = reasoner.rule_tensor
    names = list(targets) if targets is not None else list(rt.verdict_names)
    return [
        _explain_one(reasoner, out, name, case_index, visited=set()) for name in names
    ]


def _explain_one(
    reasoner: PolicyKGReasoner,
    out: ReasonerOutput,
    predicate: str,
    case_index: int,
    visited: set[str],
) -> Explanation:
    root = _build_node(reasoner, out, predicate, case_index, visited)
    return Explanation(
        verdict=predicate,
        case_index=case_index,
        truth=root.truth,
        root=root,
        policy_fingerprint=reasoner.rule_tensor.fingerprint,
    )


def _build_node(
    reasoner: PolicyKGReasoner,
    out: ReasonerOutput,
    predicate: str,
    case_index: int,
    visited: set[str],
) -> ExplanationNode:
    rt = reasoner.rule_tensor
    idx = rt.name_to_index[predicate]
    kind = (
        "base"
        if idx < rt.num_base
        else ("verdict" if predicate in rt.verdict_names else "derived")
    )
    truth = out.mu[case_index, idx].item()

    if kind == "base" or predicate in visited:
        return ExplanationNode(predicate=predicate, truth=truth, kind=kind)

    assert out.fire_trace is not None  # enforced by explain()'s guard
    fire_trace = out.fire_trace
    candidates = [r_i for r_i in range(rt.num_rules) if rt.head_idx[r_i].item() == idx]
    floor = math.log(reasoner.eps)
    best_r, best_val = None, floor
    for r_i in candidates:
        t = int(rt.rule_stratum[r_i].item())
        val = fire_trace[t, case_index, r_i].item()
        if val > best_val:
            best_val, best_r = val, r_i

    if best_r is None:
        return ExplanationNode(predicate=predicate, truth=truth, kind=kind)

    next_visited = visited | {predicate}
    body_mask = rt.body_rule == best_r
    children = []
    for pred_idx, sign in zip(
        rt.body_idx[body_mask].tolist(), rt.body_sign[body_mask].tolist(), strict=True
    ):
        child_name = rt.predicate_names[pred_idx]
        child = _build_node(reasoner, out, child_name, case_index, next_visited)
        child.negated = sign < 0
        children.append(child)

    exc_mask = rt.exc_rule == best_r
    defeated_by = []
    for pred_idx, sign in zip(
        rt.exc_idx[exc_mask].tolist(), rt.exc_sign[exc_mask].tolist(), strict=True
    ):
        name = rt.predicate_names[pred_idx]
        raw_truth = out.mu[case_index, pred_idx].item()
        original_negated = sign > 0
        effective_truth = raw_truth if not original_negated else (1.0 - raw_truth)
        defeated_by.append((name, effective_truth))
    defeated_by.sort(key=lambda pair: pair[1], reverse=True)

    return ExplanationNode(
        predicate=predicate,
        truth=truth,
        kind=kind,
        rule_id=rt.rule_ids[best_r],
        confidence=math.exp(reasoner.log_rule_conf[best_r].item()),
        children=children,
        defeated_by=defeated_by,
    )


def attribution(verdict: Tensor, mu0: Tensor, retain_graph: bool = False) -> Tensor:
    """Gradient-based explanation: sensitivity of a verdict to ``mu0``.

    Only possible because the reasoner is fully differentiable: this is
    a second explanation modality alongside the discrete rule chain from
    :func:`explain`, obtained with a single backward pass.

    Args:
        verdict: a verdict truth tensor from
            :attr:`~nspe.reasoner.ReasonerOutput.verdicts`, shape
            ``(batch,)``.
        mu0: the base predicate tensor that produced ``verdict``, with
            ``requires_grad=True``.
        retain_graph: forwarded to :func:`torch.autograd.grad`; set to
            ``True`` if computing attribution for more than one verdict
            from the same forward pass.

    Returns:
        Tensor of the same shape as ``mu0``: per-case, per-base-predicate
        sensitivity of ``verdict`` to that predicate's truth.
    """
    (grad,) = torch.autograd.grad(verdict.sum(), mu0, retain_graph=retain_graph)
    return grad
