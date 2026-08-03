"""Aggregates evaluation artifacts into the tables the paper reports.

Usage:
    python -m nspe.eval.aggregate 'docs/results/h1_h3/results_s*.json'

The mean±std figures in `docs/h1_h3_findings.md` were originally
computed by hand in a notebook. Now that the artifacts are committed,
this derives them from the files instead, so adding a seed is a rerun
rather than a recalculation.

Read-only by design: it prints markdown and writes nothing.
`docs/results/README.md` states that curating which artifacts back a
claim is a manual, deliberate act, and a tool that wrote files there
would quietly take that over.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nspe.eval.metrics import mean_std
from nspe.eval.significance import sign_permutation_test

_ARMS = ("reasoner", "baseline")

# Gap fields carrying a paired per-seed difference, so an exact sign
# test is meaningful on them.
_PAIRED_FIELDS = ("auroc_gap", "accuracy_gap", "f1_gap")
_MAX_TESTABLE_N = 20

# Fields averaged across runs within a group. Anything absent from a
# given artifact shape is skipped rather than defaulted -- see
# normalize_row on why a missing baseline field must not become 0.0.
_AGGREGATED = (
    "reasoner_auroc",
    "baseline_auroc",
    "auroc_gap",
    "reasoner_accuracy",
    "baseline_accuracy",
    "reasoner_adjusted_consistency",
    "baseline_adjusted_consistency",
    "reasoner_num_classes",
)


def load_results(patterns: Iterable[str]) -> list[tuple[str, dict[str, Any]]]:
    """Loads every artifact matching the given globs.

    Globs are expanded here rather than by the shell so that quoting is
    not load-bearing -- an unquoted pattern that the shell happens to
    expand and a quoted one that it does not must behave the same.

    Args:
        patterns: file paths or glob patterns.

    Returns:
        A list of ``(filename, parsed_json)`` pairs, sorted by path.

    Raises:
        FileNotFoundError: if a pattern matches nothing.
    """
    loaded = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or (
            [pattern] if Path(pattern).exists() else []
        )
        if not matches:
            raise FileNotFoundError(f"no artifact matched {pattern!r}")
        for match in matches:
            loaded.append((Path(match).name, json.loads(Path(match).read_text())))
    return loaded


def _backbone(result: dict[str, Any], source: str) -> str:
    """Identifies the backbone, falling back to the filename.

    The ten `results_*.json` committed before `reasoner_config` existed
    carry no record of their backbone, so for those the `results_b32_`
    prefix is the only signal there is.
    """
    recorded = result.get("reasoner_config", {}).get("clip_model")
    if recorded:
        return recorded
    return "ViT-B-32-quickgelu" if "_b32_" in source else "ViT-L-14"


def normalize_row(result: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """Flattens one artifact into comparable rows.

    The committed artifacts come in two incompatible shapes: a single
    evaluation (`results_*.json`, nested under `h1_consistency` and
    `h3_explainability`) and an ablation sweep (`ablations.json`, a flat
    row per configuration and seed). Converging them here is the whole
    point of this module.

    Args:
        result: the parsed artifact.
        source: its filename, used to identify legacy backbones.

    Returns:
        One row for a single evaluation, or one row per sweep entry.
    """
    if "h3_explainability" in result:
        return [_row_from_evaluation(result, source)]
    return [_row_from_ablation(row, result, source) for row in result["results"]]


def _row_from_evaluation(result: dict[str, Any], source: str) -> dict[str, Any]:
    h1, h3 = result["h1_consistency"], result["h3_explainability"]
    row: dict[str, Any] = {
        "source": source,
        "split": result.get("dataset", {}).get("split", "unknown"),
        "num_examples": result.get("dataset", {}).get("num_examples"),
        "clip_model": _backbone(result, source),
        "policy_name": result.get("policy_name", "unknown"),
        "threshold_source": h3.get("threshold_source"),
        "auroc_gap": h3.get("auroc_gap"),
        "accuracy_gap": h3.get("accuracy_gap"),
        "f1_gap": h3.get("f1_gap"),
        "majority_class_accuracy": h3.get("majority_class_accuracy"),
    }
    for arm in _ARMS:
        for field in ("auroc", "accuracy", "f1", "precision", "recall", "threshold"):
            row[f"{arm}_{field}"] = h3[arm].get(field)
        for field in ("adjusted_consistency", "num_classes", "degenerate"):
            row[f"{arm}_{field}"] = h1[arm].get(field)
        # positive_rate exists under both blocks and means different
        # things: H1's is taken at the verdict threshold of 0.5, H3's at
        # the fitted operating point. Never emit it unqualified.
        row[f"{arm}_h1_positive_rate"] = h1[arm].get("positive_rate")
        row[f"{arm}_h3_positive_rate"] = h3[arm].get("positive_rate")
    return row


def _row_from_ablation(
    entry: dict[str, Any], result: dict[str, Any], source: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": f"{source}:{entry['run_id']}",
        "split": "validation",
        "num_examples": None,
        "clip_model": _backbone(result, source),
        "policy_name": result.get("policy_name", "unknown"),
        "config": entry["config"]["name"],
        "seed": entry.get("seed"),
        "auroc_gap": entry.get("auroc_gap"),
        "reasoner_auroc": entry.get("reasoner_auroc"),
        "baseline_auroc": entry.get("baseline_auroc"),
        "reasoner_accuracy": entry.get("reasoner_accuracy"),
        "reasoner_adjusted_consistency": entry.get("adjusted_consistency"),
        "reasoner_num_classes": entry.get("num_classes"),
        "reasoner_degenerate": entry.get("degenerate"),
    }
    # The sweep records only the reasoner arm plus the baseline's AUROC,
    # so these genuinely do not exist here. They stay None so the table
    # prints n/a; defaulting them to 0.0 would drag any mean silently.
    row["baseline_accuracy"] = None
    row["baseline_adjusted_consistency"] = None
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float, int]]:
    """Summarizes rows field by field, skipping missing values.

    Args:
        rows: normalized rows sharing a group.

    Returns:
        A dict from field name to ``(mean, population_std, n)``. ``n``
        is reported because it varies by field: the ablation shape
        carries no baseline accuracy, so a mixed group averages some
        fields over fewer runs than others.
    """
    summary = {}
    for field in _AGGREGATED:
        values = [r[field] for r in rows if r.get(field) is not None]
        if values:
            mean, std = mean_std(values)
            summary[field] = (mean, std, len(values))
    return summary


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Returns the ``(split, backbone, policy)`` a row belongs to.

    Every component earns its place by preventing a specific silent
    averaging error: split keeps validation and test apart, backbone
    keeps ViT-L-14 and ViT-B-32 apart, and policy keeps a control run
    (a scrambled policy) from being averaged into the intact result it
    is supposed to be compared against.
    """
    return (row["split"], row["clip_model"], row["policy_name"])


def _format(value: float | None, places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def print_markdown(rows: list[dict[str, Any]]) -> None:
    """Prints a per-run table and a per-group mean±std table."""
    print("\n| source | split | n | reasoner auroc | baseline auroc | gap |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['source']} | {row['split']} "
            f"| {row['num_examples'] if row['num_examples'] is not None else 'n/a'} "
            f"| {_format(row.get('reasoner_auroc'))} "
            f"| {_format(row.get('baseline_auroc'))} "
            f"| {_format(row.get('auroc_gap'))} |"
        )

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)

    for (split, backbone, policy), group in sorted(groups.items()):
        print(f"\n### {split} / {backbone} / {policy}  ({len(group)} runs)")
        print("\n| metric | mean | std | n |")
        print("|---|---|---|---|")
        for field, (mean, std, count) in aggregate(group).items():
            print(f"| {field} | {mean:.4f} | {std:.4f} | {count} |")
        _print_significance(group)


def _print_significance(group: list[dict[str, Any]]) -> None:
    """Prints an exact paired test per gap field, with its floor."""
    tested = [
        (field, [r[field] for r in group if r.get(field) is not None])
        for field in _PAIRED_FIELDS
    ]
    tested = [(f, g) for f, g in tested if 3 <= len(g) <= _MAX_TESTABLE_N]
    if not tested:
        return

    print("\n| paired field | mean gap | positive | p (two-sided) | floor |")
    print("|---|---|---|---|---|")
    for field, gaps in tested:
        result = sign_permutation_test(gaps)
        print(
            f"| {field} | {result['mean_gap']:+.4f} "
            f"| {result['num_positive']}/{result['n']} "
            f"| {result['p_value']:.4f} | {result['min_achievable_p']:.4f} |"
        )
    print(
        "\nThe null is that the per-seed gap distribution is symmetric "
        "about zero, over *retrains on fixed data* -- evidence about "
        "reliability across seeds, not about generalization. A p equal "
        "to the floor means the design was saturated, not that the "
        "effect was marginal."
    )


def main() -> None:
    """Entry point for ``python -m nspe.eval.aggregate``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Artifact paths or globs.")
    args = parser.parse_args()

    rows = []
    for source, result in load_results(args.paths):
        rows.extend(normalize_row(result, source))
    print_markdown(rows)


if __name__ == "__main__":
    main()
