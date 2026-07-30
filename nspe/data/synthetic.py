"""Synthetic policy generation, for benchmarking rule-base scaling."""

from __future__ import annotations

import random

from nspe.policy.schema import Literal, Policy, Predicate, Rule


def make_layered_policy(
    num_base: int,
    num_rules: int,
    num_layers: int = 3,
    exception_rate: float = 0.3,
    negation_rate: float = 0.2,
    seed: int = 0,
) -> Policy:
    """Builds a random acyclic, stratified policy of a target size.

    Rules are spread evenly across ``num_layers`` strata; every rule's
    body literals are drawn only from strictly earlier layers (base
    predicates count as layer 0), so the result always satisfies the
    compiler's acyclicity and stratified-negation requirements.

    Args:
        num_base: number of base predicates.
        num_rules: total number of rules to generate.
        num_layers: number of derived/verdict strata. The final layer's
            predicates are marked as verdicts.
        exception_rate: probability that a given rule gets one ``unless``
            exception literal drawn from an earlier layer.
        negation_rate: probability that a given body literal is negated.
        seed: random seed, for reproducible benchmark policies.

    Returns:
        A :class:`~nspe.policy.schema.Policy` ready for
        :func:`nspe.policy.compiler.compile_policy`.
    """
    rng = random.Random(seed)
    predicates = [Predicate(f"b{i}", "base") for i in range(num_base)]
    rules = []

    layers: list[list[str]] = [[f"b{i}" for i in range(num_base)]]
    rules_per_layer = max(1, num_rules // num_layers)

    for layer_i in range(num_layers):
        kind = "verdict" if layer_i == num_layers - 1 else "derived"
        available = [name for grp in layers for name in grp]
        this_layer: list[str] = []
        n_rules_here = (
            rules_per_layer
            if layer_i < num_layers - 1
            else num_rules - rules_per_layer * (num_layers - 1)
        )
        for j in range(max(1, n_rules_here)):
            name = f"L{layer_i}_{j}"
            if name not in this_layer:
                predicates.append(Predicate(name, kind))
                this_layer.append(name)
            body_size = rng.randint(1, min(3, len(available)))
            body_preds = rng.sample(available, body_size)
            body = tuple(
                Literal(p, negated=rng.random() < negation_rate) for p in body_preds
            )
            unless: tuple[Literal, ...] = ()
            if rng.random() < exception_rate:
                remaining = [p for p in available if p not in body_preds]
                if remaining:
                    unless = (Literal(rng.choice(remaining)),)
            rules.append(
                Rule(id=f"R{layer_i}_{j}", head=name, body=body, unless=unless)
            )
        layers.append(this_layer)

    return Policy(
        name=f"synthetic_b{num_base}_r{len(rules)}",
        predicates=tuple(predicates),
        rules=tuple(rules),
    )
