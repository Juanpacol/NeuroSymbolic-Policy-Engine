"""YAML loading and dumping for :class:`~nspe.policy.schema.Policy`."""

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


def _dump_literals(literals: tuple[Literal, ...]) -> list[Any]:
    return [
        {"not": lit.predicate} if lit.negated else lit.predicate for lit in literals
    ]


def policy_to_dict(policy: Policy) -> dict[str, Any]:
    """Renders a :class:`Policy` as a mapping :func:`policy_from_dict` reads.

    Optional fields are omitted when empty rather than written out as
    blanks, so a dumped policy is diffable against a handwritten one.

    Args:
        policy: the policy to render.

    Returns:
        A YAML/JSON-serializable mapping.
    """
    predicates: list[dict[str, Any]] = []
    for pred in policy.predicates:
        entry: dict[str, Any] = {"name": pred.name, "kind": pred.kind}
        if pred.modality:
            entry["modality"] = pred.modality
        if pred.description:
            entry["description"] = pred.description
        predicates.append(entry)

    rules: list[dict[str, Any]] = []
    for rule in policy.rules:
        item: dict[str, Any] = {
            "id": rule.id,
            "head": rule.head,
            "body": _dump_literals(rule.body),
        }
        if rule.unless:
            item["unless"] = _dump_literals(rule.unless)
        item["confidence"] = rule.confidence
        if rule.cite:
            item["cite"] = rule.cite
        rules.append(item)

    data: dict[str, Any] = {"version": policy.version, "name": policy.name}
    if policy.source:
        data["source"] = policy.source
    data["predicates"] = predicates
    data["rules"] = rules
    return data


def dump_policy(policy: Policy, path: str | Path, header: str = "") -> None:
    """Writes a policy to a YAML file readable by :func:`load_policy`.

    Args:
        policy: the policy to write.
        path: destination path.
        header: optional comment block placed above the document, each
            line prefixed with ``#``. Used to record how a generated
            policy was produced.
    """
    body = yaml.safe_dump(policy_to_dict(policy), sort_keys=False, width=88)
    prefix = ""
    if header:
        prefix = (
            "\n".join(f"# {line}".rstrip() for line in header.splitlines()) + "\n\n"
        )
    Path(path).write_text(prefix + body)


def load_policy(path: str | Path) -> Policy:
    """Loads and validates a policy from a YAML file.

    Args:
        path: path to a policy YAML file.

    Returns:
        A validated :class:`Policy`.
    """
    data = yaml.safe_load(Path(path).read_text())
    return policy_from_dict(data)
