"""YAML loading for :class:`~nspe.policy.schema.Policy` objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nspe.policy.schema import Literal, Policy, Predicate, Rule


def _parse_literals(raw_list: list[Any] | None) -> tuple[Literal, ...]:
    return tuple(Literal.parse(item) for item in (raw_list or []))


def policy_from_dict(data: dict[str, Any]) -> Policy:
    """Builds a :class:`Policy` from a parsed YAML/JSON-like mapping.

    Args:
        data: mapping with ``name``, ``predicates``, and ``rules`` keys,
            following the ``nspe`` policy DSL (see ``docs/policy-dsl.md``).

    Returns:
        A validated :class:`Policy`.
    """
    predicates = tuple(
        Predicate(
            name=p["name"],
            kind=p["kind"],
            description=p.get("description", ""),
            modality=p.get("modality", ""),
        )
        for p in data["predicates"]
    )
    rules = tuple(
        Rule(
            id=r["id"],
            head=r["head"],
            body=_parse_literals(r.get("body")),
            unless=_parse_literals(r.get("unless")),
            confidence=float(r.get("confidence", 1.0)),
            cite=r.get("cite", ""),
        )
        for r in data["rules"]
    )
    return Policy(
        name=data["name"],
        predicates=predicates,
        rules=rules,
        version=int(data.get("version", 1)),
        source=data.get("source", ""),
    )


def load_policy(path: str | Path) -> Policy:
    """Loads and validates a policy from a YAML file.

    Args:
        path: path to a policy YAML file.

    Returns:
        A validated :class:`Policy`.
    """
    data = yaml.safe_load(Path(path).read_text())
    return policy_from_dict(data)
