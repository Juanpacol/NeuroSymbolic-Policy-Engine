"""Ablation sweep: which reasoner components earn their place?

Usage:
    python -m nspe.ablate.cli --cache-dir ./emb_cache \
        --clip-model ViT-L-14 --device cuda --out ablations.json

Trains and evaluates every configuration in ``_ABLATIONS`` across
several seeds, writing one row per (configuration, seed). The headline
H1/H3 numbers come from `docs/colab_h1_h3.md`; this is the secondary
table that says which pieces of the design the numbers actually depend
on.

Every knob swept here is reasoner-only -- `nspe.train.cli._build_model`
passes `learnable_confidence`/`aggregate` solely to the
`PolicyKGReasoner`, and `_make_aux_loss` returns `None` for anything
that is not a `PolicyEngine` -- so the neural baseline is invariant
across the whole sweep and is trained once per seed rather than once
per configuration.

The sweep is resumable. A free-tier GPU session can disappear without
warning partway through twenty-odd runs, so the results file is
rewritten after every row and a rerun skips whatever it already
contains.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from nspe.eval.cli import run_eval
from nspe.eval.metrics import mean_std
from nspe.policy.loader import load_policy
from nspe.reasoner import PolicyKGReasoner
from nspe.train.cli import build_parser, train_one

# Anything a configuration leaves unset falls through to the training
# CLI's own defaults, so `anchor_0.1` re-runs the primary configuration
# and doubles as this sweep's internal reference row.
_ABLATIONS: tuple[dict[str, Any], ...] = (
    {"name": "anchor_0.0", "lambda_anchor": 0.0},
    {"name": "anchor_0.03", "lambda_anchor": 0.03},
    {"name": "anchor_0.1", "lambda_anchor": 0.1},
    {"name": "anchor_0.3", "lambda_anchor": 0.3},
    {"name": "learnable_confidence", "learnable_confidence": True},
    {"name": "pmean", "aggregate": "pmean"},
)

# Overrides that are store_true flags: present or absent, never valued.
_BOOLEAN_OVERRIDES = frozenset({"learnable_confidence"})


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
    return {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "git_commit": _git_commit(),
    }


def config_names() -> tuple[str, ...]:
    """Returns the name of every known ablation configuration."""
    return tuple(config["name"] for config in _ABLATIONS)


def _overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Strips the bookkeeping ``name`` from a configuration."""
    return {k: v for k, v in config.items() if k != "name"}


def train_tokens(
    model: str,
    seed: int,
    out: str,
    overrides: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    """Renders one training configuration as training-CLI arguments.

    Args:
        model: ``"reasoner"`` or ``"baseline"``.
        seed: the seed for this run.
        out: checkpoint path to write.
        overrides: configuration-specific settings, from
            :data:`_ABLATIONS`.
        args: this CLI's namespace, supplying the settings shared by
            every run.

    Returns:
        An argument list for :func:`~nspe.train.cli.build_parser`.
    """
    tokens = [
        "--model",
        model,
        "--out",
        out,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--policy",
        args.policy,
        "--clip-model",
        args.clip_model,
        "--clip-pretrained",
        args.clip_pretrained,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.cache_dir is not None:
        tokens += ["--cache-dir", args.cache_dir]
    for limit, value in (
        ("--limit-train", args.limit_train),
        ("--limit-val", args.limit_val),
    ):
        if value is not None:
            tokens += [limit, str(value)]

    # The baseline is invariant to every ablation knob, so applying the
    # overrides to it would be meaningless at best and misleading at
    # worst -- two "different" baselines that are bit-identical.
    if model == "reasoner":
        for key, value in overrides.items():
            flag = "--" + key.replace("_", "-")
            if key in _BOOLEAN_OVERRIDES:
                if value:
                    tokens.append(flag)
            else:
                tokens += [flag, str(value)]
    return tokens


def _summarize(
    run_id: str,
    config: dict[str, Any],
    seed: int,
    trained: dict[str, Any],
    evaluated: dict[str, Any],
) -> dict[str, Any]:
    """Reduces one run's training and eval output to a result row."""
    h1 = evaluated["h1_consistency"]
    h3 = evaluated["h3_explainability"]
    return {
        "run_id": run_id,
        "config": config,
        "seed": seed,
        "best_epoch": trained["best_epoch"],
        "reasoner_auroc": h3["reasoner"]["auroc"],
        "baseline_auroc": h3["baseline"]["auroc"],
        "auroc_gap": h3["auroc_gap"],
        "reasoner_accuracy": h3["reasoner"]["accuracy"],
        "adjusted_consistency": h1["reasoner"]["adjusted_consistency"],
        "degenerate": h1["reasoner"]["degenerate"],
        "num_classes": h1["reasoner"]["num_classes"],
        "signature_entropy": h1["reasoner"]["signature_entropy"],
        # Where the anchor sweep's story lives: a collapsing predicate
        # layer shows up here as activation rates pinned at 0 or 1 long
        # before it shows up in AUROC.
        "predicate_stats": h1["predicate_stats"],
    }


def run_ablations(
    args: argparse.Namespace, done: set[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Trains and evaluates every requested configuration and seed.

    Args:
        args: this CLI's namespace.
        done: ``run_id`` values to skip, from a previous interrupted
            invocation.

    Yields:
        One result row per (configuration, seed), in order, so the
        caller can persist after each and lose at most one run.
    """
    done = done or set()
    checkpoints = Path(args.ckpt_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    selected = [c for c in _ABLATIONS if c["name"] in set(args.configs)]

    for config in selected:
        overrides = _overrides(config)
        for seed in args.seeds:
            run_id = f"{config['name']}/seed{seed}"
            if run_id in done:
                print(f"skip {run_id} (already in results)")
                continue

            baseline_path = checkpoints / f"baseline_seed{seed}.pt"
            if not baseline_path.exists():
                print(f"train baseline seed={seed}")
                train_one(
                    build_parser().parse_args(
                        train_tokens("baseline", seed, str(baseline_path), {}, args)
                    )
                )

            reasoner_path = checkpoints / f"{config['name']}_seed{seed}.pt"
            print(f"train {run_id}")
            trained = train_one(
                build_parser().parse_args(
                    train_tokens("reasoner", seed, str(reasoner_path), overrides, args)
                )
            )

            # The same overrides drive training and evaluation from one
            # place: aggregation is not stored in the checkpoint, so
            # letting these drift silently evaluates a model under an
            # aggregation it never saw.
            evaluated = run_eval(
                args.policy,
                str(reasoner_path),
                str(baseline_path),
                args.split,
                args.device,
                batch_size=args.batch_size,
                clip_model=args.clip_model,
                clip_pretrained=args.clip_pretrained,
                cache_dir=args.cache_dir,
                learnable_confidence=overrides.get("learnable_confidence", False),
                aggregate=overrides.get("aggregate", "tconorm"),
                pmean_p=overrides.get("pmean_p", 2.0),
            )
            yield _summarize(run_id, config, seed, trained, evaluated)


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    print("\n| config | seed | reasoner auroc | gap | adj. consistency | classes |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['config']['name']} | {row['seed']} "
            f"| {row['reasoner_auroc']:.4f} | {row['auroc_gap']:+.4f} "
            f"| {row['adjusted_consistency']:.4f} | {row['num_classes']} |"
        )

    print("\n| config | auroc | adj. consistency | classes |")
    print("|---|---|---|---|")
    for name in config_names():
        group = [r for r in rows if r["config"]["name"] == name]
        if not group:
            continue
        auroc = mean_std([r["reasoner_auroc"] for r in group])
        adjusted = mean_std([r["adjusted_consistency"] for r in group])
        classes = mean_std([float(r["num_classes"]) for r in group])
        print(
            f"| {name} | {auroc[0]:.4f} ± {auroc[1]:.4f} "
            f"| {adjusted[0]:.4f} ± {adjusted[1]:.4f} "
            f"| {classes[0]:.1f} ± {classes[1]:.1f} |"
        )


def _write(path: Path, envelope: dict[str, Any]) -> None:
    """Writes the results file atomically.

    Rewritten after every row, so a session killed mid-write would
    otherwise truncate exactly the file the next run resumes from.
    """
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.parent.mkdir(parents=True, exist_ok=True)
    with open(temp, "w") as f:
        json.dump(envelope, f, indent=2)
    temp.replace(path)


def main() -> None:
    """Entry point for ``python -m nspe.ablate.cli``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="nspe/policies/hateful_memes.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--configs",
        nargs="*",
        default=list(config_names()),
        choices=list(config_names()),
        help="Configurations to run. Defaults to all; narrow it to "
        "re-run a single one.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--cache-dir", type=str, default=None)
    # No "test": running a six-configuration sweep against held-out
    # data is the multiple-comparisons problem the split exists to
    # avoid, so the choice is unreachable rather than merely discouraged.
    parser.add_argument("--split", default="validation", choices=["validation"])
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/ablations")
    parser.add_argument("--out", type=str, default="ablations.json")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out)
    rows: list[dict[str, Any]] = []
    if out.exists():
        rows = json.loads(out.read_text()).get("results", [])
        print(f"resuming: {len(rows)} row(s) already in {out}")

    policy = load_policy(args.policy)
    envelope = {
        "environment": _environment_metadata(args.device),
        "policy_name": policy.name,
        "policy_fingerprint": PolicyKGReasoner(policy).rule_tensor.fingerprint,
        "seeds": args.seeds,
        "results": rows,
    }

    for row in run_ablations(args, done={r["run_id"] for r in rows}):
        rows.append(row)
        envelope["results"] = rows
        _write(out, envelope)
        print(f"  {row['run_id']}: auroc={row['reasoner_auroc']:.4f}")

    _print_markdown(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
