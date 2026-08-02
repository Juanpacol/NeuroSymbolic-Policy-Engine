"""Benchmark CLI: reasoner vs. Clingo, across batch sizes.

Usage:
    python -m nspe.bench.cli --device cpu --out bench_results/cpu.json

Times inference only (from an already-on-device ``mu0`` to a verdict
tensor for the reasoner; from a fact set to a stable model for Clingo).
Policy compilation, Clingo grounding, and host-to-device transfer are
excluded from the timed region, since production inference would not
repeat that work per case either.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from nspe.baselines.clingo_engine import ClingoEngine
from nspe.bench.harness import TimingStats, benchmark
from nspe.policy.loader import load_policy
from nspe.policy.schema import Policy
from nspe.reasoner import PolicyKGReasoner

_DEFAULT_BATCH_SIZES = (1, 8, 64, 256, 1024)
_DEFAULT_POLICY = "nspe/policies/meta_community_standards.yaml"
_MIN_CLINGO_REPS = 5
_SCHEMA_VERSION = 2


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _environment_metadata(device: str) -> dict[str, Any]:
    import clingo

    return {
        "torch_version": torch.__version__,
        "clingo_version": clingo.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "git_commit": _git_commit(),
    }


def _random_facts(base_names: tuple[str, ...], rng: random.Random) -> set[str]:
    return {name for name in base_names if rng.random() < 0.5}


def _crisp_mu0(
    fact_sets: list[set[str]], base_names: tuple[str, ...], device: str
) -> torch.Tensor:
    """Renders crisp fact sets as the reasoner's 0/1 input tensor.

    This is the bridge that makes the crisp arm an exact semantic match
    for the Clingo arm: the tensor returned here encodes the same cases
    as ``fact_sets``, and a reasoner built with ``tnorm="crisp"`` over it
    is the configuration ``test/test_clingo_agreement.py`` proves yields
    verdicts identical to :meth:`ClingoEngine.infer`.

    Args:
        fact_sets: one set of true base predicate names per case.
        base_names: base predicate names, in reasoner column order.
        device: device to build the tensor on.

    Returns:
        Tensor of shape ``(len(fact_sets), len(base_names))`` holding
        exactly 0.0 or 1.0.
    """
    return torch.tensor(
        [[1.0 if name in facts else 0.0 for name in base_names] for facts in fact_sets],
        device=device,
    )


def _clingo_reps(elapsed_ms: float, reps: int, budget_s: float) -> int:
    """Picks a Clingo rep count that fits a wall-clock budget.

    One Clingo rep is ``batch_size`` sequential solves, so a fixed rep
    count costs time linear in batch size -- the previous formula
    degraded to 5 reps at batch 1024 against the reasoner's 200, which
    is too few for a percentile and sits on the baseline side of a
    speedup claim. Budgeting by wall clock keeps the rep count high
    where solves are cheap and bounded where they are not.

    Args:
        elapsed_ms: measured duration of a single rep.
        reps: the requested rep count, used as an upper bound.
        budget_s: wall-clock seconds to spend on this configuration.

    Returns:
        At most ``reps``, and at least ``_MIN_CLINGO_REPS`` unless
        ``reps`` is itself smaller -- the floor exists to stop the
        budget starving the count, not to override an explicit request.
    """
    affordable = int((budget_s * 1000.0) / max(elapsed_ms, 1e-6))
    return min(reps, max(_MIN_CLINGO_REPS, affordable))


def run_sweep(
    policy: Policy,
    device: str,
    batch_sizes: tuple[int, ...],
    warmup: int,
    reps: int,
    clingo_budget_s: float = 30.0,
) -> list[dict[str, Any]]:
    """Benchmarks the reasoner and Clingo across a batch-size sweep.

    Times three arms per batch size. ``reasoner_product`` is the
    deployed configuration: graded truth degrees under the product
    t-norm. ``reasoner_crisp`` runs the crisp t-norm over the *same*
    fact sets Clingo receives, which is the only arm whose output is
    certified identical to Clingo's, and is therefore the one a speedup
    claim should lead with. ``clingo`` is the baseline.

    The graded arm draws its own input rather than sharing Clingo's.
    That is harmless: the reasoner is dense tensor arithmetic with no
    value-dependent branching, so its runtime does not depend on the
    values, only on the shapes -- which are identical across both
    reasoner arms.

    One Clingo rep is ``batch_size`` sequential solves, while one
    reasoner rep is a single batched call. Their ``median_ms`` are
    therefore comparable (both are "time to process one batch") but
    their ``p95_ms``/``p99_ms`` are not: Clingo's tails are per-batch
    aggregates over many solves, not per-case latencies. Compare
    ``per_item_median_ms`` across arms.

    Args:
        policy: the policy to benchmark.
        device: device to run the reasoner on (``"cpu"``, ``"mps"``, or
            ``"cuda"``).
        batch_sizes: batch sizes to sweep.
        warmup: warmup reps per configuration.
        reps: timed reps per configuration, an upper bound for Clingo.
        clingo_budget_s: wall-clock seconds to spend timing Clingo at
            each batch size.

    Returns:
        A list of per-batch-size result rows, each carrying the three
        arms' :class:`~nspe.bench.harness.TimingStats` as dicts plus a
        median speedup against Clingo for each reasoner arm.
    """
    reasoner = PolicyKGReasoner(policy, tnorm="product", store_trace=False).to(device)
    reasoner_crisp = PolicyKGReasoner(policy, tnorm="crisp", store_trace=False).to(
        device
    )
    engine = ClingoEngine(policy)
    base_names = policy.predicate_names("base")
    rng = random.Random(0)

    rows = []
    for batch_size in batch_sizes:
        # Drawn first: the crisp reasoner arm and the Clingo arm must
        # consume the identical cases for their outputs to be comparable.
        fact_sets = [_random_facts(base_names, rng) for _ in range(batch_size)]
        mu0_crisp = _crisp_mu0(fact_sets, base_names, device)
        mu0_graded = torch.rand(batch_size, len(base_names), device=device)

        def product_call(mu0: torch.Tensor = mu0_graded) -> None:
            reasoner(mu0)

        def crisp_call(mu0: torch.Tensor = mu0_crisp) -> None:
            reasoner_crisp(mu0)

        def clingo_call(fact_sets: list[set[str]] = fact_sets) -> None:
            for facts in fact_sets:
                engine.infer(facts)

        product_stats = benchmark(
            product_call, device=device, warmup=warmup, reps=reps, batch_size=batch_size
        )
        crisp_stats = benchmark(
            crisp_call, device=device, warmup=warmup, reps=reps, batch_size=batch_size
        )

        # One untimed rep both warms Clingo and sizes the budget.
        probe_start = time.perf_counter_ns()
        clingo_call()
        probe_ms = (time.perf_counter_ns() - probe_start) / 1e6
        clingo_stats = benchmark(
            clingo_call,
            device="cpu",
            warmup=0,
            reps=_clingo_reps(probe_ms, reps, clingo_budget_s),
            batch_size=batch_size,
        )

        rows.append(
            {
                "batch_size": batch_size,
                "reasoner_product": vars(product_stats),
                "reasoner_crisp": vars(crisp_stats),
                "clingo": vars(clingo_stats),
                "speedup_median_crisp": _speedup(clingo_stats, crisp_stats),
                "speedup_median_product": _speedup(clingo_stats, product_stats),
            }
        )
    return rows


def _speedup(baseline: TimingStats, arm: TimingStats) -> float:
    return baseline.median_ms / arm.median_ms if arm.median_ms > 0 else float("inf")


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    print(
        "| batch | crisp (ms) | product (ms) | clingo (ms) "
        "| speedup (crisp) | speedup (product) |"
    )
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['batch_size']} "
            f"| {row['reasoner_crisp']['median_ms']:.3f} "
            f"| {row['reasoner_product']['median_ms']:.3f} "
            f"| {row['clingo']['median_ms']:.3f} "
            f"| {row['speedup_median_crisp']:.2f}x "
            f"| {row['speedup_median_product']:.2f}x |"
        )


def build_parser() -> argparse.ArgumentParser:
    """Builds the benchmark CLI parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--policy",
        default=None,
        help=f"Path to a policy YAML file. Defaults to {_DEFAULT_POLICY}.",
    )
    source.add_argument(
        "--synthetic",
        type=int,
        nargs=2,
        metavar=("NUM_BASE", "NUM_RULES"),
        default=None,
        help="Generate a random stratified policy of this size instead "
        "of loading one, for the rule-count scaling sweep.",
    )
    parser.add_argument("--synthetic-layers", type=int, default=3)
    parser.add_argument("--synthetic-seed", type=int, default=0)

    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(_DEFAULT_BATCH_SIZES)
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument(
        "--clingo-budget-s",
        type=float,
        default=30.0,
        help="Wall-clock seconds to spend timing Clingo at each batch "
        "size. One Clingo rep is batch_size sequential solves, so a "
        "fixed rep count would cost time linear in batch size.",
    )
    parser.add_argument("--out", type=str, default=None)
    return parser


def _policy_from_args(args: argparse.Namespace) -> Policy:
    """Loads the policy named by the CLI, or generates a synthetic one.

    Args:
        args: a namespace from :func:`build_parser`.

    Returns:
        The policy to benchmark.
    """
    if args.synthetic is not None:
        from nspe.data.synthetic import make_layered_policy

        num_base, num_rules = args.synthetic
        return make_layered_policy(
            num_base,
            num_rules,
            num_layers=args.synthetic_layers,
            seed=args.synthetic_seed,
        )
    return load_policy(args.policy or _DEFAULT_POLICY)


def _build_envelope(
    policy: Policy, device: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Wraps benchmark rows with the metadata needed to reproduce them.

    Args:
        policy: the benchmarked policy.
        device: device the reasoner arms ran on.
        rows: rows from :func:`run_sweep`.

    Returns:
        The JSON-serializable result envelope.
    """
    return {
        "schema_version": _SCHEMA_VERSION,
        "environment": _environment_metadata(device),
        "policy_name": policy.name,
        "policy_fingerprint": PolicyKGReasoner(policy).rule_tensor.fingerprint,
        "results": rows,
    }


def main() -> None:
    """Entry point for ``python -m nspe.bench.cli``."""
    args = build_parser().parse_args()

    policy = _policy_from_args(args)
    rows = run_sweep(
        policy,
        args.device,
        tuple(args.batch_sizes),
        args.warmup,
        args.reps,
        clingo_budget_s=args.clingo_budget_s,
    )
    _print_markdown(rows)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(_build_envelope(policy, args.device, rows), indent=2)
        )
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
