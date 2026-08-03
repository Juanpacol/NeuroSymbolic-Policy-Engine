"""Generates scrambled policies: the control for whether the rules matter.

Every H1/H3 number comes from one policy, which leaves a reviewer free
to ask whether the reasoner's advantage is a property of the *rules* or
merely of having a fixed nonlinear aggregator in place of a learned
linear one. A scrambled policy answers that: it permutes which base
predicate each rule reads while leaving the rule graph, the confidences
and the predicate descriptions untouched, so the model has the same
capacity, the same grounding signal and the same computation -- wired
to the wrong evidence.

The permutation is a single global derangement over base predicate
names, not a per-slot shuffle. Being a bijection, it cannot place the
same predicate twice in one body (under the product t-norm that would
be ``mu^2``, a semantic change beyond rewiring) and it preserves the
disjointness of ``body`` and ``unless``.

Scrambling cannot break compilation. `nspe.policy.compiler` builds its
dependency graph only over derived and verdict predicates, skipping any
literal outside that set, so base predicates never enter the SCC input
or stratification at all -- permuting them among body slots cannot
change a single edge.

Usage:
    python -m nspe.policy.scramble --policy nspe/policies/hateful_memes.yaml \
        --seed 0 --out policy_scrambled_s0.yaml
"""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path

from nspe.policy.compiler import compile_policy
from nspe.policy.loader import dump_policy, load_policy
from nspe.policy.schema import Literal, Policy, Rule

_MAX_ATTEMPTS = 1000


def base_derangement(names: tuple[str, ...], seed: int) -> dict[str, str]:
    """Returns a fixed-point-free permutation of ``names``.

    Args:
        names: the base predicate names to permute. Order is significant
            only in that it seeds the shuffle reproducibly.
        seed: seed for the permutation.

    Returns:
        A dict mapping each name to a different name, bijectively.

    Raises:
        ValueError: if fewer than two names are given, for which no
            derangement exists.
    """
    if len(names) < 2:
        raise ValueError(
            f"a derangement needs at least 2 names, got {len(names)}: {names}"
        )
    rng = random.Random(seed)
    ordered = list(names)
    for _ in range(_MAX_ATTEMPTS):
        shuffled = ordered[:]
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(ordered, shuffled, strict=True)):
            return dict(zip(ordered, shuffled, strict=True))
    raise RuntimeError(f"no derangement found for {len(names)} names after retries")


def _remap(
    literals: tuple[Literal, ...], mapping: dict[str, str]
) -> tuple[Literal, ...]:
    return tuple(
        replace(lit, predicate=mapping.get(lit.predicate, lit.predicate))
        for lit in literals
    )


def scramble_policy(policy: Policy, seed: int) -> tuple[Policy, dict[str, str]]:
    """Rewires a policy's rules to read the wrong base predicates.

    Predicate declarations are left exactly as they are -- each name
    keeps its own description, so
    :meth:`~nspe.extractor.NeuroSymbolicLayer.init_heads_from_descriptions`
    still grounds every head to the same semantics it would have had.
    Only which rule reads which predicate changes, which is precisely
    the thing under test.

    Args:
        policy: the intact policy.
        seed: seed for the derangement.

    Returns:
        A tuple of the scrambled policy, whose ``name`` gains a
        ``_scrambled_s{seed}`` suffix, and the permutation applied.
    """
    mapping = base_derangement(policy.predicate_names("base"), seed)
    rules = tuple(
        Rule(
            id=rule.id,
            head=rule.head,
            body=_remap(rule.body, mapping),
            unless=_remap(rule.unless, mapping),
            confidence=rule.confidence,
            cite=rule.cite,
        )
        for rule in policy.rules
    )
    scrambled = replace(policy, name=f"{policy.name}_scrambled_s{seed}", rules=rules)
    return scrambled, mapping


def _header(
    policy: Policy, scrambled: Policy, seed: int, mapping: dict[str, str]
) -> str:
    lines = [
        "GENERATED -- do not edit by hand.",
        "",
        f"Scrambled control for {policy.name!r}, seed {seed}.",
        "Reproduce with:",
        f"    python -m nspe.policy.scramble --policy <intact> --seed {seed}",
        "",
        "Rule graph, confidences and predicate descriptions are identical to",
        "the intact policy; only which base predicate each rule reads changes.",
        "",
        f"intact fingerprint:    {compile_policy(policy).fingerprint}",
        f"scrambled fingerprint: {compile_policy(scrambled).fingerprint}",
        "",
        "Permutation:",
    ]
    lines += [f"    {src} -> {dst}" for src, dst in sorted(mapping.items())]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Writes one scrambled policy and reports what changed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    policy = load_policy(args.policy)
    scrambled, mapping = scramble_policy(policy, args.seed)
    # Compiling before writing means a policy that cannot be reasoned
    # over never reaches disk, let alone a training run.
    compile_policy(scrambled)

    dump_policy(
        scrambled, args.out, header=_header(policy, scrambled, args.seed, mapping)
    )
    print(f"wrote {args.out}")
    print(_header(policy, scrambled, args.seed, mapping))

    reloaded = load_policy(args.out)
    if compile_policy(reloaded).fingerprint != compile_policy(scrambled).fingerprint:
        raise RuntimeError(f"{args.out} does not round-trip")
    print(f"round-trip verified: {Path(args.out).name}")


if __name__ == "__main__":
    main()
