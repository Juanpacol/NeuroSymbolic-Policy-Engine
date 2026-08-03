"""Tests for the ablation sweep runner.

Training and evaluation are stubbed out: what needs testing here is the
bookkeeping -- that configurations render to arguments the training CLI
actually accepts, that the baseline is not retrained per configuration,
and that an interrupted sweep resumes without redoing or losing work.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.ablate import cli as ablate
from nspe.train.cli import build_parser

_SEEDS = [0, 1, 2]


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        policy="nspe/policies/hateful_memes.yaml",
        seeds=list(_SEEDS),
        configs=list(ablate.config_names()),
        epochs=1,
        hidden_dim=256,
        dropout=0.2,
        batch_size=32,
        device="cpu",
        clip_model="ViT-L-14",
        clip_pretrained="openai",
        cache_dir=None,
        split="validation",
        ckpt_dir="",
        out="",
        limit_train=None,
        limit_val=None,
    )
    return argparse.Namespace(**{**defaults, **overrides})


def _stub_sweep(test, args, done=None):
    """Runs the sweep with training and eval stubbed; returns the calls."""
    trained: list[argparse.Namespace] = []
    evaluated: list[dict] = []

    def fake_train_one(parsed):
        trained.append(parsed)
        # train_one's contract: it leaves a checkpoint at --out. The
        # baseline-reuse guard keys off that file existing.
        Path(parsed.out).write_bytes(b"")
        return {"best_epoch": 0}

    def fake_run_eval(*call_args, **kwargs):
        evaluated.append(kwargs)
        return {
            "h1_consistency": {
                "reasoner": {
                    "adjusted_consistency": 0.5,
                    "degenerate": False,
                    "num_classes": 40,
                    "signature_entropy": 4.0,
                },
                "predicate_stats": {},
            },
            "h3_explainability": {
                "reasoner": {"auroc": 0.7, "accuracy": 0.6},
                "baseline": {"auroc": 0.68},
                "auroc_gap": 0.02,
            },
        }

    original_train, original_eval = ablate.train_one, ablate.run_eval
    ablate.train_one, ablate.run_eval = fake_train_one, fake_run_eval
    try:
        rows = list(ablate.run_ablations(args, done=done))
    finally:
        ablate.train_one, ablate.run_eval = original_train, original_eval
    return rows, trained, evaluated


class _TempDirMixin:
    def _tmp(self) -> str:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        return directory


class TestConfigMatrix(_TempDirMixin, TestCase):
    def test_matrix_shape_and_unique_run_ids(self):
        rows, _, _ = _stub_sweep(self, _args(ckpt_dir=self._tmp()))

        self.assertEqual(len(rows), len(ablate.config_names()) * len(_SEEDS))
        self.assertEqual(len({r["run_id"] for r in rows}), len(rows))

    def test_configs_can_be_narrowed(self):
        rows, _, _ = _stub_sweep(
            self, _args(configs=["pmean"], seeds=[0], ckpt_dir=self._tmp())
        )
        self.assertEqual([r["run_id"] for r in rows], ["pmean/seed0"])


class TestTrainTokens(TestCase):
    def test_every_config_parses_under_the_training_parser(self):
        """The reason build_parser is exported rather than reimplemented.

        Rendering configurations as arguments and parsing them with the
        real parser means a flag renamed in the training CLI fails here
        instead of silently producing a sweep that trains something
        other than what it reports.
        """
        args = _args()
        for config in ablate._ABLATIONS:
            overrides = ablate._overrides(config)
            tokens = ablate.train_tokens("reasoner", 0, "/tmp/x.pt", overrides, args)
            parsed = build_parser().parse_args(tokens)

            for key, value in overrides.items():
                self.assertEqual(getattr(parsed, key), value, f"{config['name']}.{key}")

    def test_boolean_overrides_render_as_bare_flags(self):
        args = _args()
        on = ablate.train_tokens(
            "reasoner", 0, "/tmp/x.pt", {"learnable_confidence": True}, args
        )
        off = ablate.train_tokens(
            "reasoner", 0, "/tmp/x.pt", {"learnable_confidence": False}, args
        )

        self.assertIn("--learnable-confidence", on)
        # A valued form would not parse: the flag is store_true.
        self.assertNotIn("True", on)
        self.assertNotIn("--learnable-confidence", off)

    def test_baseline_ignores_ablation_overrides(self):
        # The baseline has no reasoner, so applying these would produce
        # "different" baselines that are bit-identical.
        tokens = ablate.train_tokens(
            "baseline", 0, "/tmp/b.pt", {"aggregate": "pmean"}, _args()
        )
        self.assertNotIn("--aggregate", tokens)
        self.assertEqual(build_parser().parse_args(tokens).aggregate, "tconorm")

    def test_optional_passthroughs_are_omitted_when_unset(self):
        tokens = ablate.train_tokens("reasoner", 0, "/tmp/x.pt", {}, _args())
        self.assertNotIn("--cache-dir", tokens)
        self.assertNotIn("--limit-train", tokens)

        tokens = ablate.train_tokens(
            "reasoner", 0, "/tmp/x.pt", {}, _args(cache_dir="/c", limit_train=8)
        )
        parsed = build_parser().parse_args(tokens)
        self.assertEqual(parsed.cache_dir, "/c")
        self.assertEqual(parsed.limit_train, 8)


class TestSweepBookkeeping(_TempDirMixin, TestCase):
    def test_baseline_is_trained_once_per_seed_per_trunk_config(self):
        _, trained, _ = _stub_sweep(self, _args(ckpt_dir=self._tmp()))

        baselines = [a for a in trained if a.model == "baseline"]
        reasoners = [a for a in trained if a.model == "reasoner"]
        self.assertEqual(len(reasoners), len(ablate.config_names()) * len(_SEEDS))

        # Every reasoner-only config shares one baseline per seed; each
        # distinct trunk configuration needs its own, or the arms would
        # no longer be capacity-matched.
        tags = {ablate.baseline_tag(ablate._overrides(c)) for c in ablate._ABLATIONS}
        self.assertEqual(len(baselines), len(tags) * len(_SEEDS))

    def test_resume_skips_completed_runs(self):
        done = {"anchor_0.0/seed0", "anchor_0.0/seed1", "pmean/seed2"}
        rows, trained, _ = _stub_sweep(self, _args(ckpt_dir=self._tmp()), done=done)

        expected = len(ablate.config_names()) * len(_SEEDS) - len(done)
        self.assertEqual(len(rows), expected)
        self.assertFalse(done & {r["run_id"] for r in rows})
        self.assertEqual(len([a for a in trained if a.model == "reasoner"]), len(rows))

    def test_eval_receives_the_same_overrides_as_training(self):
        rows, _, evaluated = _stub_sweep(
            self,
            _args(
                configs=["pmean", "learnable_confidence"],
                seeds=[0],
                ckpt_dir=self._tmp(),
            ),
        )
        by_run = dict(zip([r["run_id"] for r in rows], evaluated, strict=True))

        self.assertEqual(by_run["pmean/seed0"]["aggregate"], "pmean")
        self.assertFalse(by_run["pmean/seed0"]["learnable_confidence"])
        self.assertTrue(by_run["learnable_confidence/seed0"]["learnable_confidence"])
        self.assertEqual(by_run["learnable_confidence/seed0"]["aggregate"], "tconorm")


class TestOverrideRouting(_TempDirMixin, TestCase):
    """The baseline mounts the same trunk, so trunk knobs must reach it.

    Applying one to the reasoner alone would compare arms of different
    capacity while looking like a clean ablation.
    """

    def test_shared_trunk_override_reaches_the_baseline(self):
        args = _args()
        tokens = ablate.train_tokens(
            "baseline", 0, "/tmp/b.pt", {"hidden_dim": 0}, args
        )
        self.assertIn("--hidden-dim", tokens)
        self.assertEqual(build_parser().parse_args(tokens).hidden_dim, 0)

    def test_reasoner_only_override_does_not_reach_the_baseline(self):
        tokens = ablate.train_tokens(
            "baseline", 0, "/tmp/b.pt", {"aggregate": "pmean"}, _args()
        )
        self.assertNotIn("--aggregate", tokens)
        self.assertEqual(build_parser().parse_args(tokens).aggregate, "tconorm")

    def test_every_ablation_key_is_classified(self):
        known = ablate._REASONER_ONLY | ablate._BOTH_ARMS
        for config in ablate._ABLATIONS:
            for key in ablate._overrides(config):
                self.assertIn(key, known, f"{config['name']}.{key}")

    def test_taxonomy_sets_are_disjoint(self):
        self.assertEqual(ablate._REASONER_ONLY & ablate._BOTH_ARMS, set())


class TestBaselineTag(_TempDirMixin, TestCase):
    def test_reasoner_only_configs_share_one_baseline(self):
        self.assertEqual(ablate.baseline_tag({"aggregate": "pmean"}), "shared")
        self.assertEqual(ablate.baseline_tag({"lambda_anchor": 0.3}), "shared")
        self.assertEqual(ablate.baseline_tag({}), "shared")

    def test_trunk_configs_get_their_own_baseline(self):
        self.assertNotEqual(ablate.baseline_tag({"hidden_dim": 0}), "shared")

    def test_different_widths_do_not_collide(self):
        self.assertNotEqual(
            ablate.baseline_tag({"hidden_dim": 0}),
            ablate.baseline_tag({"hidden_dim": 512}),
        )

    def test_baseline_checkpoints_differ_across_trunk_widths(self):
        rows, trained, _ = _stub_sweep(
            self,
            _args(
                configs=["anchor_0.1", "linear_probe"],
                seeds=[0],
                ckpt_dir=self._tmp(),
            ),
        )
        baselines = {a.out for a in trained if a.model == "baseline"}

        self.assertEqual(len(rows), 2)
        # One baseline per distinct trunk configuration, not one total.
        self.assertEqual(len(baselines), 2)


class TestHiddenDimReachesEval(_TempDirMixin, TestCase):
    def test_eval_receives_the_configs_width_not_the_default(self):
        rows, _, evaluated = _stub_sweep(
            self, _args(configs=["linear_probe"], seeds=[0], ckpt_dir=self._tmp())
        )
        self.assertEqual(len(rows), 1)
        # Without this, run_eval would rebuild a 256-wide trunk and fail
        # to load a 0-width checkpoint on the head shapes.
        self.assertEqual(evaluated[0]["hidden_dim"], 0)

    def test_other_configs_pass_the_cli_default(self):
        _, _, evaluated = _stub_sweep(
            self, _args(configs=["pmean"], seeds=[0], ckpt_dir=self._tmp())
        )
        self.assertEqual(evaluated[0]["hidden_dim"], 256)


if __name__ == "__main__":
    run_tests()
