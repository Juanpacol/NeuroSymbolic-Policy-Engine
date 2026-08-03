"""Tests for nspe.eval.aggregate.

The load-bearing test is
``test_reproduces_the_published_validation_figures``: this module exists
to replace a hand-computed table, so it is only trustworthy if it lands
on the same numbers the paper already reports.
"""

import json
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.eval.aggregate import (
    aggregate,
    group_key,
    load_results,
    normalize_row,
    policy_family,
)
from nspe.eval.metrics import mean_std

_COMMITTED = "docs/results/h1_h3"


def _evaluation(split: str = "validation", auroc: float = 0.72) -> dict:
    return {
        "dataset": {"split": split, "num_examples": 831},
        "h1_consistency": {
            arm: {
                "adjusted_consistency": 0.5,
                "num_classes": 40,
                "degenerate": False,
                "positive_rate": 0.41,
            }
            for arm in ("reasoner", "baseline")
        },
        "h3_explainability": {
            "threshold_source": "fitted",
            "auroc_gap": 0.03,
            "accuracy_gap": 0.02,
            "majority_class_accuracy": 0.568,
            "reasoner": {"auroc": auroc, "accuracy": 0.62, "positive_rate": 0.71},
            "baseline": {
                "auroc": auroc - 0.03,
                "accuracy": 0.60,
                "positive_rate": 0.68,
            },
        },
    }


class TestNormalizeRow(TestCase):
    def test_nested_evaluation_shape(self):
        rows = normalize_row(_evaluation(), "results_s0.json")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["split"], "validation")
        self.assertEqual(rows[0]["reasoner_auroc"], 0.72)
        self.assertEqual(rows[0]["baseline_auroc"], 0.69)
        self.assertEqual(rows[0]["clip_model"], "ViT-L-14")

    def test_flat_ablation_shape_yields_one_row_per_entry(self):
        sweep = {
            "results": [
                {
                    "run_id": "pmean/seed0",
                    "config": {"name": "pmean"},
                    "seed": 0,
                    "reasoner_auroc": 0.72,
                    "baseline_auroc": 0.68,
                    "auroc_gap": 0.04,
                    "reasoner_accuracy": 0.62,
                    "adjusted_consistency": 0.5,
                    "num_classes": 40,
                    "degenerate": False,
                },
                {
                    "run_id": "pmean/seed1",
                    "config": {"name": "pmean"},
                    "seed": 1,
                    "reasoner_auroc": 0.71,
                    "baseline_auroc": 0.68,
                    "auroc_gap": 0.03,
                    "reasoner_accuracy": 0.61,
                    "adjusted_consistency": 0.4,
                    "num_classes": 38,
                    "degenerate": False,
                },
            ]
        }
        rows = normalize_row(sweep, "ablations.json")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["config"], "pmean")
        self.assertEqual(rows[0]["reasoner_auroc"], 0.72)

    def test_absent_baseline_fields_are_none_not_zero(self):
        # The sweep records only the reasoner arm plus baseline AUROC.
        # Defaulting the rest to 0.0 would silently drag every mean.
        sweep = {
            "results": [
                {
                    "run_id": "pmean/seed0",
                    "config": {"name": "pmean"},
                    "seed": 0,
                    "reasoner_auroc": 0.72,
                    "baseline_auroc": 0.68,
                }
            ]
        }
        row = normalize_row(sweep, "ablations.json")[0]

        self.assertIsNone(row["baseline_accuracy"])
        self.assertIsNone(row["baseline_adjusted_consistency"])

    def test_backbone_falls_back_to_the_filename(self):
        # The ten committed results_*.json predate reasoner_config.
        self.assertEqual(
            normalize_row(_evaluation(), "results_b32_s0.json")[0]["clip_model"],
            "ViT-B-32-quickgelu",
        )
        self.assertEqual(
            normalize_row(_evaluation(), "results_s0.json")[0]["clip_model"],
            "ViT-L-14",
        )

    def test_recorded_backbone_wins_over_the_filename(self):
        result = _evaluation()
        result["reasoner_config"] = {"clip_model": "ViT-B-16"}
        self.assertEqual(
            normalize_row(result, "results_s0.json")[0]["clip_model"], "ViT-B-16"
        )

    def test_h1_and_h3_positive_rates_stay_distinct(self):
        # Same key name, different operating points: H1's is at the
        # verdict threshold of 0.5, H3's at the fitted threshold.
        row = normalize_row(_evaluation(), "results_s0.json")[0]

        self.assertEqual(row["reasoner_h1_positive_rate"], 0.41)
        self.assertEqual(row["reasoner_h3_positive_rate"], 0.71)
        self.assertNotIn("positive_rate", row)


class TestGrouping(TestCase):
    def test_splits_never_mix(self):
        validation = normalize_row(_evaluation("validation"), "results_s0.json")[0]
        test = normalize_row(_evaluation("test"), "results_test_s0.json")[0]
        self.assertNotEqual(group_key(validation), group_key(test))

    def test_backbones_never_mix(self):
        l14 = normalize_row(_evaluation(), "results_s0.json")[0]
        b32 = normalize_row(_evaluation(), "results_b32_s0.json")[0]
        self.assertNotEqual(group_key(l14), group_key(b32))

    def test_policies_never_mix(self):
        """The guard that keeps a control out of the result it controls.

        A scrambled-policy run shares its split and backbone with the
        intact run. Without policy in the key the two would be averaged
        into a single meaningless mean, silently and with no error.
        """
        intact = _evaluation()
        intact["policy_name"] = "hateful_memes_policy"
        scrambled = _evaluation()
        scrambled["policy_name"] = "hateful_memes_policy_scrambled_s0"

        self.assertNotEqual(
            group_key(normalize_row(intact, "results_test_s0.json")[0]),
            group_key(normalize_row(scrambled, "results_scram_test_s0.json")[0]),
        )

    def test_missing_policy_name_does_not_crash(self):
        # The twenty already-committed artifacts predate nothing here,
        # but an artifact from another tool might omit it.
        row = normalize_row(_evaluation(), "results_s0.json")[0]
        self.assertEqual(row["policy_name"], "unknown")
        self.assertEqual(len(group_key(row)), 3)

    def test_scrambled_seeds_pool_with_each_other(self):
        """Ten scramble seeds are ten repeats of one control, not ten configs.

        Without this, each `_scrambled_s{seed}` name forms its own
        group of n=1 -- too small to summarize, and never matching the
        n=10 the intact result reports.
        """
        seed0 = _evaluation()
        seed0["policy_name"] = "hateful_memes_policy_scrambled_s0"
        seed7 = _evaluation()
        seed7["policy_name"] = "hateful_memes_policy_scrambled_s7"

        self.assertEqual(
            group_key(normalize_row(seed0, "results_scram_test_s0.json")[0]),
            group_key(normalize_row(seed7, "results_scram_test_s7.json")[0]),
        )

    def test_policy_family_only_strips_a_trailing_scramble_seed(self):
        self.assertEqual(
            policy_family("hateful_memes_policy_scrambled_s3"),
            "hateful_memes_policy_scrambled",
        )
        # Not a scramble suffix: left alone, so two genuinely different
        # policies still never pool.
        self.assertEqual(policy_family("hateful_memes_policy"), "hateful_memes_policy")
        self.assertEqual(
            policy_family("meta_community_standards"), "meta_community_standards"
        )


class TestAggregate(TestCase):
    def test_skips_missing_values_and_reports_the_count(self):
        rows = [
            {"reasoner_auroc": 0.7, "baseline_accuracy": 0.6},
            {"reasoner_auroc": 0.8, "baseline_accuracy": None},
        ]
        summary = aggregate(rows)

        self.assertEqual(summary["reasoner_auroc"][2], 2)
        self.assertEqual(summary["baseline_accuracy"][2], 1)
        self.assertAlmostEqual(summary["reasoner_auroc"][0], 0.75)

    def test_uses_population_std(self):
        rows = [{"reasoner_auroc": v} for v in (0.7, 0.8)]
        self.assertAlmostEqual(aggregate(rows)["reasoner_auroc"][1], 0.05)


class TestPublishedFigures(TestCase):
    def test_reproduces_the_published_validation_figures(self):
        """Pins the aggregator to docs/h1_h3_findings.md.

        Those tables were computed by hand before these artifacts were
        committed. If this drifts -- most plausibly by someone switching
        to a sample standard deviation -- every error bar in the paper
        would move without anything else failing.
        """
        if not Path(_COMMITTED).is_dir():
            self.skipTest("committed artifacts not present")

        # results_s[0-9].json, not results_s*.json: the wildcard form
        # also matches results_scram_test_s0.json ("results_s" + "cram_
        # test_s0" + ".json"), which inflates n from 5 to 25 now that
        # scrambled-control artifacts share the directory.
        rows = []
        for source, result in load_results([f"{_COMMITTED}/results_s[0-9].json"]):
            rows.extend(normalize_row(result, source))
        self.assertEqual(len(rows), 5)

        summary = aggregate(rows)
        for field, mean, std in (
            ("reasoner_auroc", 0.7193, 0.0060),
            ("baseline_auroc", 0.6866, 0.0096),
            ("reasoner_accuracy", 0.6253, 0.0170),
            ("baseline_accuracy", 0.5853, 0.0239),
            ("reasoner_adjusted_consistency", 0.6703, 0.1504),
            ("baseline_adjusted_consistency", 0.2998, 0.1025),
        ):
            self.assertAlmostEqual(summary[field][0], mean, places=4, msg=field)
            self.assertAlmostEqual(summary[field][1], std, places=4, msg=field)


class TestLoadResults(TestCase):
    def test_expands_globs_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in (1, 0):
                (Path(tmp) / f"results_s{seed}.json").write_text(
                    json.dumps(_evaluation())
                )
            loaded = load_results([f"{tmp}/results_s*.json"])

        self.assertEqual(
            [name for name, _ in loaded], ["results_s0.json", "results_s1.json"]
        )

    def test_unmatched_pattern_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_results(["/nonexistent/results_*.json"])


class TestMeanStd(TestCase):
    def test_population_not_sample(self):
        mean, std = mean_std([0.7, 0.8])
        self.assertAlmostEqual(mean, 0.75)
        self.assertAlmostEqual(std, 0.05)


if __name__ == "__main__":
    run_tests()
