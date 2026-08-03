"""Exhaustive sweep over every rewiring of a policy's base predicates.

The scrambled-policy control retrains on a *random* derangement and
finds no AUROC cost (`docs/h1_h3_findings.md`). That leaves the obvious
follow-up: was the random derangement simply not a hard one? With six
base predicates there are only 720 permutations, so the question does
not need a search for an adversarial wiring -- it can be answered by
enumerating all of them, which also yields the worst case for free.

The sweep holds the *trained predicate layer fixed* and varies only the
wiring, which is what makes it cheap and what makes it complementary to
the control: the control lets each wiring train its own heads, this one
asks how much the wiring is worth to heads that cannot adapt. Together
they separate "the wiring carries signal" from "the heads adapt to
whatever wiring they are given".

Rewiring a policy is exactly equivalent to permuting ``mu0``'s columns.
:func:`~nspe.policy.compiler.compile_policy` orders predicates
``base + derived + verdict`` with base in declaration order, and
:meth:`~nspe.reasoner.PolicyKGReasoner.forward` writes
``log_mu[:, :num_base] = log(mu0)``, so rewiring bodies by a bijection
``s`` over base names is the same as feeding ``mu0[:, perm]`` with
``perm[i] = column_of(s(names[i]))``. The sweep therefore compiles one
reasoner and gathers columns, rather than rebuilding 720 policies;
`test_eval_wiring_sweep.py` pins the equivalence, because getting the
direction of ``s`` backwards is otherwise a silent error.

Scores are the *raw* verdict, never the calibrated one. AUROC is
invariant to any monotone map so calibration cannot change a number
here, and the calibrator was fitted under the intact wiring -- applying
it across wirings would be the sweep's only asymmetry. That is also
what lets the sweep run from a dumped ``mu0`` with no checkpoint.

Usage:
    python -m nspe.eval.wiring_sweep --policy nspe/policies/hateful_memes.yaml \
        --mu0 mu0_val_s0.json --mu0 mu0_val_s1.json --out wiring_sweep_val.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nspe.eval.metrics import auroc, mean_std
from nspe.eval.significance import sign_permutation_test
from nspe.policy.compiler import compile_policy
from nspe.policy.loader import load_policy
from nspe.policy.schema import Policy
from nspe.reasoner import PolicyKGReasoner


def sweep_wirings(
    policy: Policy, mu0: Tensor, labels: Tensor, verdict: str = "hateful"
) -> list[dict[str, Any]]:
    """Scores every permutation of the policy's base predicates.

    Args:
        policy: the intact policy.
        mu0: base predicate truth degrees, shape ``(n, P)``, with columns
            in ``policy.predicate_names("base")`` order.
        labels: binary ground truth, shape ``(n,)``.
        verdict: name of the verdict predicate to score.

    Returns:
        One row per permutation, each with ``permutation`` (column
        indices), ``mapping`` (predicate name to the name that replaces
        it, ready for
        :func:`~nspe.policy.scramble.apply_permutation`),
        ``is_derangement``, and ``auroc``. The identity permutation --
        the intact wiring -- appears exactly once.

    Raises:
        ValueError: if ``mu0``'s width does not match the policy's base
            predicate count, or ``verdict`` is not a verdict predicate.
    """
    names = policy.predicate_names("base")
    if mu0.shape[-1] != len(names):
        raise ValueError(
            f"mu0 has {mu0.shape[-1]} columns but the policy declares "
            f"{len(names)} base predicates: {names}"
        )
    if verdict not in policy.predicate_names("verdict"):
        raise ValueError(
            f"{verdict!r} is not a verdict predicate; policy declares "
            f"{policy.predicate_names('verdict')}"
        )

    reasoner = PolicyKGReasoner(policy, store_trace=False)
    rows = []
    with torch.no_grad():
        for permutation in itertools.permutations(range(len(names))):
            scores = reasoner(mu0[:, permutation]).verdicts[verdict]
            rows.append(
                {
                    "permutation": list(permutation),
                    "mapping": {
                        names[i]: names[permutation[i]] for i in range(len(names))
                    },
                    "is_derangement": all(i != p for i, p in enumerate(permutation)),
                    "auroc": auroc(scores, labels),
                }
            )
    return rows


def _population(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    values = [row["auroc"] for row in rows]
    mean, std = mean_std(values)
    return {
        "population": label,
        "count": len(values),
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def summarize_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduces one seed's sweep to the numbers worth reporting.

    The headline is the **spread**, not the intact wiring's rank. The
    predicate layer being swept was trained under the intact wiring, so
    intact scoring near the top is expected by construction and reads as
    circular; how far the worst wiring falls is the part that carries
    information.

    Args:
        rows: the output of :func:`sweep_wirings`.

    Returns:
        A dict with ``intact_auroc``, per-population statistics over all
        permutations and over derangements only, the ``worst`` and
        ``best`` rows, and where intact ranks. Rank is reported as both
        ``strictly_below`` and ``at_or_below`` because ties are likely --
        distinct wirings can still score identically.
    """
    intact = next(
        row for row in rows if row["permutation"] == sorted(row["permutation"])
    )
    derangements = [row for row in rows if row["is_derangement"]]
    scores = [row["auroc"] for row in rows]

    return {
        "intact_auroc": intact["auroc"],
        "all_permutations": _population(rows, "all_permutations"),
        "derangements": _population(derangements, "derangements"),
        "worst": min(rows, key=lambda row: row["auroc"]),
        "best": max(rows, key=lambda row: row["auroc"]),
        "intact_rank": {
            "strictly_below": sum(1 for s in scores if s < intact["auroc"]),
            "at_or_below": sum(1 for s in scores if s <= intact["auroc"]),
            "total": len(scores),
        },
    }


def load_mu0_dump(path: str | Path, policy: Policy) -> tuple[Tensor, Tensor]:
    """Loads a ``--dump-mu0`` artifact, checking it matches the policy.

    Args:
        path: a JSON file written by ``nspe.eval.cli --dump-mu0``.
        policy: the policy the sweep will run.

    Returns:
        A tuple of ``mu0`` shape ``(n, P)`` and ``labels`` shape ``(n,)``.

    Raises:
        ValueError: if the dump's predicate names or policy fingerprint
            disagree with ``policy``. Either mismatch would silently
            reinterpret the columns, which is the one error this whole
            analysis cannot detect from its own output.
    """
    dump = json.loads(Path(path).read_text())
    names = policy.predicate_names("base")
    if tuple(dump["predicate_names"]) != names:
        raise ValueError(
            f"{path} carries predicate_names {dump['predicate_names']} but the "
            f"policy declares {list(names)}; the mu0 columns would be misread"
        )
    fingerprint = compile_policy(policy).fingerprint
    if dump["policy_fingerprint"] != fingerprint:
        raise ValueError(
            f"{path} was produced under policy_fingerprint "
            f"{dump['policy_fingerprint'][:12]}... but this policy compiles to "
            f"{fingerprint[:12]}..."
        )
    # float32 both, matching what the eval path produces: labels are
    # written as ints for compactness but consumed as truth degrees.
    return (
        torch.tensor(dump["mu0"], dtype=torch.float32),
        torch.tensor(dump["labels"], dtype=torch.float32),
    )


def build_parser() -> argparse.ArgumentParser:
    """Builds the sweep CLI's argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="nspe/policies/hateful_memes.yaml")
    parser.add_argument(
        "--mu0",
        action="append",
        required=True,
        help="a --dump-mu0 artifact; repeat once per seed",
    )
    parser.add_argument("--verdict", default="hateful")
    parser.add_argument("--out", default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Sweeps every wiring for each dumped seed and reports the spread."""
    args = build_parser().parse_args(argv)
    policy = load_policy(args.policy)

    per_seed = []
    for path in args.mu0:
        mu0, labels = load_mu0_dump(path, policy)
        summary = summarize_sweep(sweep_wirings(policy, mu0, labels, args.verdict))
        summary["source"] = Path(path).name
        summary["num_examples"] = int(mu0.shape[0])
        per_seed.append(summary)
        print(
            f"{summary['source']}: intact {summary['intact_auroc']:.4f} | "
            f"worst {summary['worst']['auroc']:.4f} | "
            f"spread {summary['all_permutations']['spread']:.4f}"
        )

    gaps = [s["intact_auroc"] - s["worst"]["auroc"] for s in per_seed]
    spreads = [s["all_permutations"]["spread"] for s in per_seed]
    mean_spread, std_spread = mean_std(spreads)
    result = {
        "schema": 1,
        "policy_name": policy.name,
        "policy_fingerprint": compile_policy(policy).fingerprint,
        "verdict": args.verdict,
        "per_seed": per_seed,
        "across_seeds": {
            "mean_spread": mean_spread,
            "std_spread": std_spread,
            "intact_minus_worst": sign_permutation_test(gaps, alternative="greater"),
        },
    }

    print(
        f"\nacross {len(per_seed)} seeds: mean spread {mean_spread:.4f} "
        f"+-{std_spread:.4f}"
    )
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
