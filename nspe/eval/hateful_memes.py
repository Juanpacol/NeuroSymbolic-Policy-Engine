"""Core H1/H3 metric computation for the reasoner vs. neural baseline.

Kept free of any dataset/CLIP/checkpoint I/O so it is unit-testable with
hand-built tensors; `nspe/eval/cli.py` wires this to real data and
checkpoints.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from nspe.consistency import ConsistencyChecker
from nspe.eval.metrics import best_threshold, binary_metrics
from nspe.explain import Explanation
from nspe.reasoner import PolicyKGReasoner, ReasonerOutput


def _report_to_dict(report: Any) -> dict[str, Any]:
    return report.as_dict()


def compute_h1(
    mu0: Tensor,
    reasoner_verdict: Tensor,
    baseline_verdict: Tensor,
    tau: float = 0.5,
    verdict_threshold: float = 0.5,
) -> dict[str, Any]:
    """Computes paired H1 consistency reports for the reasoner and baseline.

    Both verdicts are grouped into predicate-equivalence classes using
    the SAME ``mu0`` -- the reasoner's own predicate layer. The neural
    baseline has no interpretable predicate layer of its own to group
    by, so this asks "under the reasoner's notion of which cases are
    equivalent, does each model actually treat them the same way." This
    single call site (one ``mu0`` argument, two verdict arguments) makes
    it structurally impossible to accidentally group each model by a
    different signature.

    Args:
        mu0: base predicate truth degrees from the reasoner's extractor,
            shape ``(batch, num_base_predicates)``.
        reasoner_verdict: reasoner verdict truth degrees, shape
            ``(batch,)``.
        baseline_verdict: baseline verdict truth degrees, shape
            ``(batch,)``.
        tau: threshold used to binarize ``mu0`` into equivalence classes.
        verdict_threshold: threshold used to binarize each verdict.

    Returns:
        A dict with ``reasoner`` and ``baseline`` sub-dicts (each with
        ``inconsistency_rate``, ``purity``, ``num_classes``) and
        ``worst_classes_reasoner`` / ``worst_classes_baseline`` lists.
    """
    checker = ConsistencyChecker(tau=tau, verdict_threshold=verdict_threshold)
    reasoner_report = checker(mu0, reasoner_verdict)
    baseline_report = checker(mu0, baseline_verdict)
    return {
        "reasoner": _report_to_dict(reasoner_report),
        "baseline": _report_to_dict(baseline_report),
        "worst_classes_reasoner": reasoner_report.worst_classes(n=5),
        "worst_classes_baseline": baseline_report.worst_classes(n=5),
    }


def compute_h3(
    reasoner_verdict: Tensor,
    baseline_verdict: Tensor,
    labels: Tensor,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Computes H3 accuracy/F1/AUROC for the reasoner and baseline.

    AUROC is the headline number. It is threshold-free, so it does not
    depend on an operating point that the verdict calibrator can move,
    and it cannot be won by a model that collapses to the base rate.
    Accuracy and F1 are reported alongside at a stated threshold.

    The "not significant accuracy loss" bar is intentionally left
    undecided here -- this reports the raw reasoner-minus-baseline gap
    for the paper's methods section to interpret, rather than inventing
    a significance test that was never agreed on.

    Args:
        reasoner_verdict: reasoner verdict truth degrees, shape
            ``(batch,)``.
        baseline_verdict: baseline verdict truth degrees, shape
            ``(batch,)``.
        labels: ground-truth binary labels, shape ``(batch,)``.
        threshold: threshold used to binarize each verdict. ``None``
            fits one per model by maximizing F1 on *these* data, which
            is only legitimate when they are the validation split --
            fitting and reporting a threshold on the same split inflates
            the result, so a reported test split must be passed a
            threshold fitted on validation.

    Returns:
        A dict with ``reasoner``/``baseline`` metric dicts, the
        ``threshold`` each used, a ``threshold_source`` of ``"fitted"``
        or ``"provided"``, and ``accuracy_gap``/``f1_gap``/``auroc_gap``
        (reasoner minus baseline).
    """
    if threshold is None:
        source = "fitted"
        reasoner_threshold, _ = best_threshold(reasoner_verdict, labels)
        baseline_threshold, _ = best_threshold(baseline_verdict, labels)
    else:
        source = "provided"
        reasoner_threshold = baseline_threshold = threshold

    reasoner_metrics = binary_metrics(reasoner_verdict, labels, reasoner_threshold)
    baseline_metrics = binary_metrics(baseline_verdict, labels, baseline_threshold)
    reasoner_metrics["threshold"] = reasoner_threshold
    baseline_metrics["threshold"] = baseline_threshold

    return {
        "reasoner": reasoner_metrics,
        "baseline": baseline_metrics,
        "threshold_source": source,
        "majority_class_accuracy": max(
            labels.float().mean().item(), 1.0 - labels.float().mean().item()
        ),
        "accuracy_gap": reasoner_metrics["accuracy"] - baseline_metrics["accuracy"],
        "f1_gap": reasoner_metrics["f1"] - baseline_metrics["f1"],
        "auroc_gap": reasoner_metrics["auroc"] - baseline_metrics["auroc"],
    }


def sample_explanations(
    policy: Any,
    reasoner: PolicyKGReasoner,
    mu0: Tensor,
    case_indices: list[int],
    target: str = "hateful",
) -> list[dict[str, Any]]:
    """Produces rule -> predicate -> verdict audit chains for select cases.

    Re-runs the reasoner with ``store_trace=True`` (bulk metrics use
    ``store_trace=False`` for speed) restricted to the requested rows,
    since :func:`~nspe.explain.explain` needs the fire/mu traces.

    Args:
        policy: the :class:`~nspe.policy.schema.Policy` ``reasoner`` was
            built from (the reasoner itself only keeps the compiled
            tensors, not the original policy object).
        reasoner: a reasoner sharing that policy, used for bulk metrics
            with ``store_trace=False`` (no learnable reasoner parameters
            by default, so a fresh trace-enabled instance is equivalent).
        mu0: base predicate truth degrees for the full batch, shape
            ``(batch, num_base_predicates)``.
        case_indices: row indices into ``mu0`` to explain.
        target: verdict predicate name to explain.

    Returns:
        A list of dicts with ``case_index`` and rendered explanation
        ``chain`` text, one per requested index.
    """
    if not case_indices:
        return []
    trace_reasoner = PolicyKGReasoner(
        policy, tnorm=reasoner.tnorm.name, store_trace=True
    )
    trace_reasoner.load_state_dict(reasoner.state_dict())
    subset = mu0[case_indices]
    out: ReasonerOutput = trace_reasoner(subset)

    results = []
    for local_idx, case_index in enumerate(case_indices):
        explanations: list[Explanation] = trace_reasoner.explain(
            out, targets=[target], case_index=local_idx
        )
        results.append(
            {
                "case_index": case_index,
                "chain": explanations[0].render(),
            }
        )
    return results
