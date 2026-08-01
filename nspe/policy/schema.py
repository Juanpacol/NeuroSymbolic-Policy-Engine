"""Plain-dataclass representation of a policy, before tensor compilation."""

from __future__ import annotations

from dataclasses import dataclass

_VALID_KINDS = frozenset({"base", "derived", "verdict"})


@dataclass(frozen=True)
class Predicate:
    """A named proposition tracked by the reasoner.

    Args:
        name: unique predicate name.
        kind: one of ``"base"`` (produced by the neural extractor),
            ``"derived"`` (concluded by a rule, used as an intermediate),
            or ``"verdict"`` (a rule-concluded predicate exposed as
            reasoner output).
        description: human-readable note, not used by the compiler.
        modality: free-form tag (e.g. ``"text"``, ``"image"``,
            ``"multimodal"``), not used by the compiler.
    """

    name: str
    kind: str
    description: str = ""
    modality: str = ""

    def __post_init__(self) -> None:
        """Validates `kind` is one of the recognized predicate kinds."""
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"Predicate {self.name!r} has invalid kind {self.kind!r}; "
                f"expected one of {sorted(_VALID_KINDS)}"
            )


@dataclass(frozen=True)
class Literal:
    """A (possibly negated) reference to a predicate inside a rule."""

    predicate: str
    negated: bool = False

    @classmethod
    def parse(cls, raw: object) -> Literal:
        """Parses a literal from YAML-friendly input.

        Args:
            raw: either a plain predicate name (positive literal) or a
                single-key mapping ``{"not": name}`` (negated literal).

        Returns:
            The parsed :class:`Literal`.

        Raises:
            ValueError: if ``raw`` is neither shape.
        """
        if isinstance(raw, str):
            return cls(predicate=raw, negated=False)
        if isinstance(raw, dict) and set(raw) == {"not"}:
            return cls(predicate=raw["not"], negated=True)
        raise ValueError(
            f"Cannot parse literal from {raw!r}; expected a predicate name "
            "or a mapping {'not': name}"
        )


@dataclass(frozen=True)
class Rule:
    """A policy rule: ``head <- body[0] AND ... UNLESS unless[0] OR ...``.

    Args:
        id: unique rule identifier.
        head: name of the predicate this rule concludes.
        body: conjunction of literals that must hold for the rule to fire.
        unless: exception literals; if any holds, the rule's contribution
            is defeated (multiplicatively, not removed outright).
        confidence: rule prior in ``[0, 1]``, applied as a firing weight.
        cite: free-form citation, e.g. a link to the source policy text.
    """

    id: str
    head: str
    body: tuple[Literal, ...]
    unless: tuple[Literal, ...] = ()
    confidence: float = 1.0
    cite: str = ""

    def __post_init__(self) -> None:
        """Validates confidence range and that the body is non-empty."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Rule {self.id!r} confidence must be in [0, 1], got {self.confidence}"
            )
        if not self.body:
            raise ValueError(f"Rule {self.id!r} must have at least one body literal")


@dataclass(frozen=True)
class Policy:
    """A full policy: its predicate vocabulary and its rule set."""

    name: str
    predicates: tuple[Predicate, ...]
    rules: tuple[Rule, ...]
    version: int = 1
    source: str = ""

    def __post_init__(self) -> None:
        """Validates predicate names and rule ids are each unique."""
        names = [p.name for p in self.predicates]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate predicate names: {dupes}")
        rule_ids = [r.id for r in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            dupes = sorted({i for i in rule_ids if rule_ids.count(i) > 1})
            raise ValueError(f"Duplicate rule ids: {dupes}")

    def predicate_names(self, kind: str | None = None) -> tuple[str, ...]:
        """Returns predicate names, optionally filtered by ``kind``."""
        if kind is None:
            return tuple(p.name for p in self.predicates)
        return tuple(p.name for p in self.predicates if p.kind == kind)

    def predicate_descriptions(self, kind: str | None = None) -> dict[str, str]:
        """Returns predicate name to description, filtered by ``kind``.

        The compiler ignores descriptions, but they are the only
        per-predicate supervision signal available for grounding a
        neural predicate layer -- see
        :meth:`~nspe.extractor.NeuroSymbolicLayer.init_heads_from_descriptions`.

        Args:
            kind: predicate kind to filter by, or ``None`` for all.

        Returns:
            A dict mapping predicate name to its description. Predicates
            without one map to the empty string.
        """
        return {
            p.name: p.description
            for p in self.predicates
            if kind is None or p.kind == kind
        }
