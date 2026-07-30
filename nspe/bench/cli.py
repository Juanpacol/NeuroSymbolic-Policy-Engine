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
from pathlib import Path
from typing import Any

import torch

from nspe.baselines.clingo_engine import ClingoEngine
from nspe.bench.harness import benchmark
from nspe.policy.loader import load_policy
from nspe.policy.schema import Policy
from nspe.reasoner import PolicyKGReasoner

_DEFAULT_BATCH_SIZES = (1, 8, 64, 256, 1024)


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


def run_sweep(
    policy: Policy,
    device: str,
    batch_sizes: tuple[int, ...],
    warmup: int,
    reps: int,
) -> list[dict[str, Any]]:
    """Benchmarks the reasoner and Clingo across a batch-size sweep.

    Args:
        policy: the policy to benchmark.
        device: device to run the reasoner on (``"cpu"``, ``"mps"``, or
            ``"cuda"``).
        batch_sizes: batch sizes to sweep.
        warmup: warmup reps per configuration.
        reps: timed reps per configuration.

    Returns:
        A list of per-batch-size result rows, each with reasoner and
        Clingo :class:`~nspe.bench.harness.TimingStats` as dicts.
    """
    reasoner = PolicyKGReasoner(policy, tnorm="product", store_trace=False).to(device)
    engine = ClingoEngine(policy)
    base_names = policy.predicate_names("base")
    rng = random.Random(0)

    rows = []
    for batch_size in batch_sizes:
        mu0 = torch.rand(batch_size, len(base_names), device=device)

        def reasoner_call(mu0: torch.Tensor = mu0) -> None:
            reasoner(mu0)

        reasoner_stats = benchmark(
            reasoner_call,
            device=device,
            warmup=warmup,
            reps=reps,
            batch_size=batch_size,
        )

        fact_sets = [_random_facts(base_names, rng) for _ in range(batch_size)]

        def clingo_call(fact_sets: list[set[str]] = fact_sets) -> None:
            for facts in fact_sets:
                engine.infer(facts)

        clingo_reps = max(5, reps // max(1, batch_size // 8 + 1))
        clingo_stats = benchmark(
            clingo_call,
            device="cpu",
            warmup=min(5, warmup),
            reps=clingo_reps,
            batch_size=batch_size,
        )

        rows.append(
            {
                "batch_size": batch_size,
                "reasoner": vars(reasoner_stats),
                "clingo": vars(clingo_stats),
                "speedup_median": clingo_stats.median_ms / reasoner_stats.median_ms
                if reasoner_stats.median_ms > 0
                else float("inf"),
            }
        )
    return rows


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| batch | reasoner median (ms) | clingo median (ms) | speedup |")
    print("|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['batch_size']} "
            f"| {row['reasoner']['median_ms']:.3f} "
            f"| {row['clingo']['median_ms']:.3f} "
            f"| {row['speedup_median']:.2f}x |"
        )


def main() -> None:
    """Entry point for ``python -m nspe.bench.cli``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--policy",
        default="nspe/policies/meta_community_standards.yaml",
        help="Path to a policy YAML file.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(_DEFAULT_BATCH_SIZES),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    policy = load_policy(args.policy)
    rows = run_sweep(
        policy, args.device, tuple(args.batch_sizes), args.warmup, args.reps
    )
    _print_markdown(rows)

    result = {
        "environment": _environment_metadata(args.device),
        "policy_name": policy.name,
        "policy_fingerprint": PolicyKGReasoner(policy).rule_tensor.fingerprint,
        "results": rows,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
