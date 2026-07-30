"""Compiles a :class:`~nspe.policy.schema.Policy` into a :class:`RuleTensor`.

Scope (v0.1): the rule dependency graph among derived/verdict predicates
must be acyclic. Content-moderation policies are, in practice, layered
("tier 1 hate speech" -> "remove"), so this is not a real restriction for
the target domain, and it keeps the fixpoint's iteration count exact
(one pass per stratum, computed at compile time) instead of a heuristic
depth for convergent positive cycles. See CLAUDE.md / docs for the
rationale; recursive positive cycles are explicitly out of scope, not a
bug.

Negation (including ``unless`` exceptions) must always reference a
predicate in a strictly lower stratum than the rule's head -- the
classical stratified-Datalog restriction. This is what keeps the
monotone fixpoint update sound and keeps semantics comparable to an ASP
solver such as Clingo.
"""

from __future__ import annotations

import hashlib

import torch

from nspe.policy.rule_tensor import RuleTensor
from nspe.policy.schema import Policy, Rule


class PolicyCompileError(ValueError):
    """Raised when a policy fails static validation at compile time."""


def _tarjan_scc(nodes: list[str], edges: dict[str, list[str]]) -> list[list[str]]:
    """Returns the strongly connected components of ``(nodes, edges)``."""
    index_counter = 0
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    result: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index_counter
        index[v] = index_counter
        lowlink[v] = index_counter
        index_counter += 1
        stack.append(v)
        on_stack[v] = True
        for w in edges.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w):
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            result.append(scc)

    for v in nodes:
        if v not in index:
            strongconnect(v)
    return result


def _validate_rule_predicates(rule: Rule, known: dict[str, str]) -> None:
    if rule.head not in known:
        raise PolicyCompileError(
            f"Rule {rule.id!r} concludes undefined predicate {rule.head!r}"
        )
    if known[rule.head] == "base":
        raise PolicyCompileError(
            f"Rule {rule.id!r} concludes {rule.head!r}, which is declared as "
            "a base predicate; base predicates come only from the neural "
            "extractor and cannot be rule heads."
        )
    for lit in (*rule.body, *rule.unless):
        if lit.predicate not in known:
            raise PolicyCompileError(
                f"Rule {rule.id!r} references undefined predicate "
                f"{lit.predicate!r}"
            )


def _fingerprint(policy: Policy) -> str:
    parts = [policy.name, str(policy.version)]
    for pred in sorted(policy.predicates, key=lambda p: p.name):
        parts.append(f"P|{pred.name}|{pred.kind}")
    for rule in sorted(policy.rules, key=lambda r: r.id):
        body = ",".join(
            sorted(f"{'~' if lit.negated else ''}{lit.predicate}" for lit in rule.body)
        )
        unless = ",".join(sorted(lit.predicate for lit in rule.unless))
        parts.append(f"R|{rule.id}|{rule.head}|{body}|{unless}|{rule.confidence}")
    canonical = "\n".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compile_policy(policy: Policy, max_iterations: int = 8) -> RuleTensor:
    """Compiles a validated :class:`Policy` into flat tensor buffers.

    Args:
        policy: the policy to compile.
        max_iterations: hard cap on the number of unrolled fixpoint
            iterations (also the maximum supported number of strata).

    Returns:
        A :class:`RuleTensor` ready to drive
        :class:`~nspe.reasoner.PolicyKGReasoner`.

    Raises:
        PolicyCompileError: on any undefined predicate, a rule concluding
            a base predicate, a cyclic predicate dependency, or a
            negation/exception that does not reference a strictly lower
            stratum.
    """
    kind_by_name = {p.name: p.kind for p in policy.predicates}
    for rule in policy.rules:
        _validate_rule_predicates(rule, kind_by_name)

    base = [p.name for p in policy.predicates if p.kind == "base"]
    derived = [p.name for p in policy.predicates if p.kind == "derived"]
    verdict = [p.name for p in policy.predicates if p.kind == "verdict"]
    ordered_names = base + derived + verdict
    name_to_index = {name: i for i, name in enumerate(ordered_names)}
    num_base = len(base)

    node_set = set(derived) | set(verdict)
    pos_edges: dict[str, set[str]] = {n: set() for n in node_set}
    neg_edges: dict[str, set[str]] = {n: set() for n in node_set}
    for rule in policy.rules:
        head = rule.head
        for lit in rule.body:
            if lit.predicate not in node_set:
                continue
            (neg_edges if lit.negated else pos_edges)[head].add(lit.predicate)
        for lit in rule.unless:
            if lit.predicate not in node_set:
                continue
            neg_edges[head].add(lit.predicate)

    all_edges = {n: pos_edges[n] | neg_edges[n] for n in node_set}
    sccs = _tarjan_scc(list(node_set), {n: list(all_edges[n]) for n in node_set})
    scc_of = {n: i for i, scc in enumerate(sccs) for n in scc}

    for n in node_set:
        for dep in neg_edges[n]:
            if scc_of[dep] == scc_of[n]:
                raise PolicyCompileError(
                    f"Illegal negation: predicate {n!r} negates {dep!r}, but "
                    f"they are mutually dependent (cycle: {sccs[scc_of[n]]}), "
                    f"so {dep!r} is not in a strictly lower stratum."
                )

    for scc in sccs:
        if len(scc) > 1:
            raise PolicyCompileError(
                f"Cyclic dependency among derived/verdict predicates: "
                f"{scc}. Recursive positive cycles are not supported in "
                "this version; restructure the policy to remove the cycle."
            )
        n = scc[0]
        if n in all_edges[n]:
            raise PolicyCompileError(
                f"Predicate {n!r} depends on itself, which is not supported."
            )

    stratum: dict[str, int] = {}

    def compute_stratum(n: str) -> int:
        if n in stratum:
            return stratum[n]
        deps = all_edges[n]
        stratum[n] = 0 if not deps else 1 + max(compute_stratum(d) for d in deps)
        return stratum[n]

    for n in node_set:
        compute_stratum(n)

    for n in node_set:
        for dep in neg_edges[n]:
            if stratum[dep] >= stratum[n]:
                raise PolicyCompileError(
                    f"Stratified negation violation: {n!r} negates {dep!r} "
                    f"at stratum {stratum[dep]}, which is not strictly "
                    f"lower than {n!r}'s stratum {stratum[n]}."
                )

    num_iterations = (max(stratum.values()) + 1) if stratum else 1
    num_iterations = min(num_iterations, max_iterations)

    body_rule_l: list[int] = []
    body_idx_l: list[int] = []
    body_sign_l: list[float] = []
    exc_rule_l: list[int] = []
    exc_idx_l: list[int] = []
    exc_sign_l: list[float] = []
    head_idx_l: list[int] = []
    rule_stratum_l: list[int] = []
    conf_l: list[float] = []

    for r_i, rule in enumerate(policy.rules):
        head_idx_l.append(name_to_index[rule.head])
        rule_stratum_l.append(stratum[rule.head])
        conf_l.append(rule.confidence)
        for lit in rule.body:
            body_rule_l.append(r_i)
            body_idx_l.append(name_to_index[lit.predicate])
            body_sign_l.append(-1.0 if lit.negated else 1.0)
        for lit in rule.unless:
            exc_rule_l.append(r_i)
            exc_idx_l.append(name_to_index[lit.predicate])
            # Sign is flipped relative to the body convention: we want
            # gather_literal_log to directly yield log(1 - exception
            # truth), i.e. the defeat contribution, not the literal's own
            # truth.
            exc_sign_l.append(1.0 if lit.negated else -1.0)

    log_conf = torch.log(
        torch.tensor(conf_l or [1.0], dtype=torch.float32).clamp(min=1e-7)
    )
    if not conf_l:
        log_conf = log_conf[:0]

    return RuleTensor(
        num_predicates=len(ordered_names),
        num_base=num_base,
        num_rules=len(policy.rules),
        predicate_names=tuple(ordered_names),
        name_to_index=name_to_index,
        verdict_names=tuple(verdict),
        rule_ids=tuple(rule.id for rule in policy.rules),
        body_idx=torch.tensor(body_idx_l, dtype=torch.long),
        body_rule=torch.tensor(body_rule_l, dtype=torch.long),
        body_sign=torch.tensor(body_sign_l, dtype=torch.float32),
        exc_idx=torch.tensor(exc_idx_l, dtype=torch.long),
        exc_rule=torch.tensor(exc_rule_l, dtype=torch.long),
        exc_sign=torch.tensor(exc_sign_l, dtype=torch.float32),
        head_idx=torch.tensor(head_idx_l, dtype=torch.long),
        rule_stratum=torch.tensor(rule_stratum_l, dtype=torch.long),
        log_rule_conf=log_conf,
        num_iterations=num_iterations,
        fingerprint=_fingerprint(policy),
    )
