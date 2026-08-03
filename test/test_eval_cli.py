"""Tests for the evaluation CLI's threshold resolution.

`resolve_thresholds` is where test-set protocol is enforced, so these
cover the ways a run could quietly end up at the wrong operating point:
fitting on test, chaining a threshold that was never fitted, or reading
one from an artifact produced by a different backbone.
"""

import argparse
import json
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.eval.cli import resolve_thresholds

_POLICY = "nspe/policies/hateful_memes.yaml"


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        split="validation",
        policy=_POLICY,
        clip_model="ViT-L-14",
        threshold=None,
        thresholds_from=None,
        reasoner_threshold=None,
        baseline_threshold=None,
    )
    return argparse.Namespace(**{**defaults, **overrides})


def _validation_artifact(
    tmp: str,
    reasoner: float = 0.0658,
    baseline: float = 0.1510,
    threshold_source: str = "fitted",
    clip_model: str | None = "ViT-L-14",
) -> str:
    payload = {
        "h3_explainability": {
            "threshold_source": threshold_source,
            "reasoner": {"threshold": reasoner},
            "baseline": {"threshold": baseline},
        },
        # A different operating point entirely; must never be read.
        "h1_consistency": {
            "reasoner": {"positive_rate": 0.4164},
            "baseline": {"positive_rate": 0.4236},
        },
    }
    if clip_model is not None:
        payload["reasoner_config"] = {"clip_model": clip_model}

    path = Path(tmp) / "results_val.json"
    path.write_text(json.dumps(payload))
    return str(path)


class TestResolveThresholds(TestCase):
    def test_validation_with_nothing_fits_per_arm(self):
        self.assertIsNone(resolve_thresholds(_args()))

    def test_scalar_threshold_passes_through(self):
        self.assertEqual(resolve_thresholds(_args(threshold=0.5)), 0.5)

    def test_reads_both_arms_from_a_validation_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_thresholds(
                _args(thresholds_from=_validation_artifact(tmp), split="test")
            )
        self.assertEqual(resolved, (0.0658, 0.1510))

    def test_refuses_to_fit_on_test(self):
        with self.assertRaisesRegex(ValueError, "test split"):
            resolve_thresholds(_args(split="test"))

    def test_rejects_a_source_that_was_not_itself_fitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _validation_artifact(tmp, threshold_source="provided")
            with self.assertRaisesRegex(ValueError, "threshold_source"):
                resolve_thresholds(_args(thresholds_from=source, split="test"))

    def test_rejects_a_backbone_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _validation_artifact(tmp, clip_model="ViT-B-32-quickgelu")
            with self.assertRaisesRegex(ValueError, "clip_model"):
                resolve_thresholds(_args(thresholds_from=source, split="test"))

    def test_accepts_a_legacy_artifact_without_reasoner_config(self):
        # The ten committed results_*.json predate that block.
        with tempfile.TemporaryDirectory() as tmp:
            source = _validation_artifact(tmp, clip_model=None)
            self.assertEqual(
                resolve_thresholds(_args(thresholds_from=source)), (0.0658, 0.1510)
            )

    def test_explicit_per_arm_flags(self):
        resolved = resolve_thresholds(
            _args(reasoner_threshold=0.1, baseline_threshold=0.2, split="test")
        )
        self.assertEqual(resolved, (0.1, 0.2))

    def test_one_per_arm_flag_alone_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "given together"):
            resolve_thresholds(_args(reasoner_threshold=0.1))
        with self.assertRaisesRegex(ValueError, "given together"):
            resolve_thresholds(_args(baseline_threshold=0.2))


class TestMu0Dump(TestCase):
    """The dump the eval CLI writes must be one the sweep can read.

    These two are the only producer and consumer of that format, so the
    round trip is what keeps them from drifting apart silently.
    """

    def test_written_dump_loads_back_through_the_sweep(self):
        import torch

        from nspe.eval.cli import _write_mu0_dump
        from nspe.eval.wiring_sweep import load_mu0_dump
        from nspe.policy.loader import load_policy

        policy = load_policy(_POLICY)
        names = policy.predicate_names("base")
        mu0 = torch.rand(8, len(names))
        labels = (torch.arange(8) % 2).float()

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "mu0.json")
            _write_mu0_dump(path, mu0, labels, policy, names, "cpu")
            loaded_mu0, loaded_labels = load_mu0_dump(path, policy)
            payload = json.loads(Path(path).read_text())

        self.assertEqual(tuple(loaded_mu0.shape), (8, len(names)))
        torch.testing.assert_close(loaded_labels, labels)
        # Rounded for size, so compare at the recorded precision.
        torch.testing.assert_close(loaded_mu0, mu0, atol=1e-6, rtol=0)
        self.assertEqual(payload["predicate_names"], list(names))

    def test_a_reordered_policy_is_rejected_by_the_consumer(self):
        import torch

        from nspe.eval.cli import _write_mu0_dump
        from nspe.eval.wiring_sweep import load_mu0_dump
        from nspe.policy.loader import load_policy
        from nspe.policy.schema import Policy

        policy = load_policy(_POLICY)
        names = policy.predicate_names("base")
        base = [p for p in policy.predicates if p.kind == "base"]
        others = [p for p in policy.predicates if p.kind != "base"]
        reordered = Policy(
            name=policy.name,
            predicates=tuple([base[1], base[0], *base[2:], *others]),
            rules=policy.rules,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "mu0.json")
            _write_mu0_dump(
                path, torch.rand(8, len(names)), torch.zeros(8), policy, names, "cpu"
            )
            with self.assertRaisesRegex(ValueError, "predicate_names"):
                load_mu0_dump(path, reordered)


class TestSpreadSample(TestCase):
    """Published explanations must not all come from one class.

    The dataset's rows are label-sorted, so head-of-list sampling drew
    every explanation from the same label.
    """

    def test_spreads_across_the_range_instead_of_taking_the_head(self):
        import torch

        from nspe.eval.cli import _spread_sample

        disagreements = torch.arange(100)
        picked = _spread_sample(disagreements, 5)

        self.assertEqual(len(picked), 5)
        self.assertEqual(picked[0], 0)
        self.assertEqual(picked[-1], 99)
        # The old behaviour was [0, 1, 2, 3, 4] -- entirely the head.
        self.assertNotEqual(picked, list(range(5)))

    def test_returns_everything_when_there_are_too_few(self):
        import torch

        from nspe.eval.cli import _spread_sample

        self.assertEqual(_spread_sample(torch.tensor([7, 9]), 5), [7, 9])

    def test_indices_stay_ascending_and_unique(self):
        import torch

        from nspe.eval.cli import _spread_sample

        picked = _spread_sample(torch.arange(0, 200, 2), 5)
        self.assertEqual(picked, sorted(picked))
        self.assertEqual(len(set(picked)), len(picked))


if __name__ == "__main__":
    run_tests()
