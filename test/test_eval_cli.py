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


if __name__ == "__main__":
    run_tests()
