"""The compiled tensor representation of a policy."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class RuleTensor:
    """Flat tensor buffers a :class:`~nspe.reasoner.PolicyKGReasoner` runs on.

    All literal buffers use a COO (rule, predicate) layout so conjunction
    and head-aggregation are plain segment reductions, independent of the
    total predicate count.

    Attributes:
        num_predicates: total predicate count (base + derived + verdict).
        num_base: number of base predicates; they occupy indices
            ``[0, num_base)`` and are supplied by the neural extractor.
        num_rules: number of rules.
        predicate_names: predicate names, ordered by index.
        name_to_index: predicate name to index.
        verdict_names: names of predicates of kind ``"verdict"``.
        rule_ids: original rule id string per rule index, for citing a
            specific rule in an explanation.
        body_idx: predicate index per body literal, shape ``(nnz_b,)``.
        body_rule: rule index per body literal, shape ``(nnz_b,)``.
        body_sign: ``+1`` for a positive literal, ``-1`` for a negated
            one, shape ``(nnz_b,)``.
        exc_idx: predicate index per exception literal, shape ``(nnz_e,)``.
        exc_rule: rule index per exception literal, shape ``(nnz_e,)``.
        exc_sign: sign such that gathering with this sign yields
            ``log(1 - exception_truth)`` directly, shape ``(nnz_e,)``.
        head_idx: predicate index of each rule's head, shape ``(R,)``.
        rule_stratum: stratum of each rule's head, shape ``(R,)``. Rules
            must fire in stratum order so a negated/exception literal
            always reads its referenced predicate's fully-stabilized
            value, never a same-sweep partial one.
        log_rule_conf: log of each rule's confidence, shape ``(R,)``.
        num_iterations: number of fixpoint iterations to unroll, equal to
            the number of strata in the rule dependency graph.
        fingerprint: sha256 hex digest identifying the exact policy this
            was compiled from, for audit trails.
    """

    num_predicates: int
    num_base: int
    num_rules: int
    predicate_names: tuple[str, ...]
    name_to_index: dict[str, int]
    verdict_names: tuple[str, ...]
    rule_ids: tuple[str, ...]
    body_idx: Tensor
    body_rule: Tensor
    body_sign: Tensor
    exc_idx: Tensor
    exc_rule: Tensor
    exc_sign: Tensor
    head_idx: Tensor
    rule_stratum: Tensor
    log_rule_conf: Tensor
    num_iterations: int
    fingerprint: str
