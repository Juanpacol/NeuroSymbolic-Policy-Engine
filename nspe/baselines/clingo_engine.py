"""Clingo (ASP) baseline: the exact-inference reference for H2.

Clingo does exact inference over crisp facts; `PolicyKGReasoner` does
approximate graded inference. Timing them against each other is only a
fair comparison on the fragment where they agree exactly, so this
module exists to make that agreement checkable, not to make a fast
production ASP engine.
"""

from __future__ import annotations

import clingo

from nspe.policy.schema import Literal, Policy


def _literal_to_asp(lit: Literal, is_exception: bool) -> str:
    """Renders one literal as an ASP body atom.

    Args:
        lit: the literal to render.
        is_exception: ``True`` for an ``unless`` (exception) literal. An
            un-negated exception defeats the rule when it holds, so the
            rule only fires when it does *not* hold -- the opposite
            polarity of an ordinary body literal.

    Returns:
        The ASP atom, e.g. ``"slur_present"`` or ``"not educational"``.
    """
    negate = (not lit.negated) if is_exception else lit.negated
    return f"not {lit.predicate}" if negate else lit.predicate


def policy_to_asp(policy: Policy) -> str:
    """Translates a policy's rules into a stratified ASP program.

    Args:
        policy: the policy to translate. Must already satisfy
            :mod:`nspe.policy.compiler`'s acyclicity and
            stratified-negation constraints; this function does not
            re-validate them.

    Returns:
        An ASP program covering the rules only. Base predicates are
        left undeclared here; :class:`ClingoEngine` declares them as
        choice atoms so their truth can be fixed per case via solver
        assumptions, without re-grounding.
    """
    lines = []
    for rule in policy.rules:
        parts = [_literal_to_asp(lit, is_exception=False) for lit in rule.body]
        parts += [_literal_to_asp(lit, is_exception=True) for lit in rule.unless]
        body = ", ".join(parts)
        lines.append(f"{rule.head} :- {body}." if body else f"{rule.head}.")
    return "\n".join(lines)


class ClingoEngine:
    """Multi-shot Clingo baseline: grounds a policy once, reuses it.

    Base predicates are declared as choice atoms; their truth per case
    is fixed via solver assumptions rather than by adding facts and
    re-grounding, which is the standard multi-shot pattern and the fair
    comparison point for :class:`~nspe.reasoner.PolicyKGReasoner`.

    Args:
        policy: the policy to ground.
    """

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.base_names = policy.predicate_names("base")
        self._atoms = {name: clingo.Function(name) for name in self.base_names}

        choice_rules = "\n".join(f"{{{name}}}." for name in self.base_names)
        program = f"{choice_rules}\n{policy_to_asp(policy)}"

        self.control = clingo.Control(["--warn=none"])
        self.control.add("base", [], program)
        self.control.ground([("base", [])])

    def infer(self, facts: set[str]) -> set[str]:
        """Derives the stable model's atoms for one crisp fact set.

        Args:
            facts: names of base predicates that are true; every other
                base predicate is assumed false.

        Returns:
            Names of every true atom (facts plus derived predicates) in
            the stable model. The underlying program is stratified, so
            the model is unique.
        """
        assumptions = [(atom, name in facts) for name, atom in self._atoms.items()]
        derived: set[str] = set()
        with self.control.solve(assumptions=assumptions, yield_=True) as handle:
            for model in handle:
                derived = {str(sym) for sym in model.symbols(atoms=True)}
                break
        return derived
