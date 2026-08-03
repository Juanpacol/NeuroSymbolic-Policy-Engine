"""Exact paired significance testing over per-seed gaps.

The findings docs report results as "positive in 5/5 seeds", which is
suggestive but not a test. This supplies the test, exactly rather than
approximately: at the sample sizes involved (5 to 10 seeds) every
normal-approximation method -- Wilcoxon signed-rank included -- is
invalid, while enumerating all ``2**n`` sign assignments is both cheap
and exact. It also needs no new dependency, which matters because
:mod:`nspe.eval.metrics` states a no-third-party-numerics rule and
scipy is not in this project's dependency set.

**What the null actually says.** The hypothesis rejected is that the
per-seed gap distribution is symmetric about zero, where the unit of
observation is *a retrain of the same two models on the same fixed
dataset*. A small p-value therefore licenses "the reasoner reliably
beats the baseline across random initializations" -- it does **not**
license "the reasoner generalizes better", which would require
independent samples of data, not of seeds. Say so wherever these
numbers are reported.

**The floor matters as much as the p-value.** With ``n`` paired
observations the smallest reachable two-sided p is ``2 / 2**n``: at
n=5 that is 0.0625, so *no* result at five seeds can reach p<0.05
two-sided no matter how large the effect. :func:`min_achievable_p`
exists so that fact is reported alongside the p-value rather than
discovered by a reviewer.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, cast

_MAX_N = 20
_ALTERNATIVES = ("two-sided", "greater", "less")


def min_achievable_p(n: int, alternative: str = "two-sided") -> float:
    """Returns the smallest p-value reachable at this sample size.

    Args:
        n: number of paired observations.
        alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns:
        The floor on the p-value, which is ``2 / 2**n`` two-sided and
        ``1 / 2**n`` one-sided. A reported p equal to this value means
        the design was saturated, not that the effect was marginal.

    Raises:
        ValueError: if ``alternative`` is unknown or ``n`` is not
            positive.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"unknown alternative: {alternative}")
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    tails = 2 if alternative == "two-sided" else 1
    return cast(float, tails / 2**n)


def sign_permutation_test(
    gaps: Sequence[float], alternative: str = "two-sided"
) -> dict[str, Any]:
    """Runs an exact sign-flip permutation test on paired gaps.

    Under the null that each gap's sign is equally likely to be positive
    or negative, every one of the ``2**n`` sign assignments is equally
    probable. Enumerating them gives the exact null distribution of the
    summed gap, with no distributional assumption and no tie correction.

    Args:
        gaps: one paired difference per seed, e.g. reasoner AUROC minus
            baseline AUROC. Must be non-empty.
        alternative: ``"two-sided"`` (the default),  ``"greater"`` if the
            direction was hypothesized in advance, or ``"less"``.

    Returns:
        A dict with ``n``, ``mean_gap``, ``num_positive``,
        ``statistic`` (the observed sum), ``p_value``, ``alternative``,
        ``min_achievable_p``, and ``num_permutations``.

    Raises:
        ValueError: if ``gaps`` is empty, ``alternative`` is unknown, or
            ``n`` exceeds 20 -- beyond which exhaustive enumeration
            stops being the right tool and a Monte-Carlo test should be
            used instead.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"unknown alternative: {alternative}")
    values = [float(g) for g in gaps]
    if not values:
        raise ValueError("gaps must be non-empty")
    if len(values) > _MAX_N:
        raise ValueError(
            f"exhaustive enumeration needs n <= {_MAX_N}, got {len(values)}; "
            "use a Monte-Carlo permutation test at this size"
        )

    observed = sum(values)
    magnitudes = [abs(v) for v in values]
    extreme = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(values)):
        total = sum(s * m for s, m in zip(signs, magnitudes, strict=True))
        if alternative == "two-sided":
            extreme += abs(total) >= abs(observed) - 1e-12
        elif alternative == "greater":
            extreme += total >= observed - 1e-12
        else:
            extreme += total <= observed + 1e-12

    num_permutations = 2 ** len(values)
    return {
        "n": len(values),
        "mean_gap": observed / len(values),
        "num_positive": sum(1 for v in values if v > 0),
        "statistic": observed,
        "p_value": extreme / num_permutations,
        "alternative": alternative,
        "min_achievable_p": min_achievable_p(len(values), alternative),
        "num_permutations": num_permutations,
    }
