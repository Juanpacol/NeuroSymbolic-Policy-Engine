"""Tests for the Hateful Memes policy (nspe/policies/hateful_memes.yaml)."""

from pathlib import Path

from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.policy.compiler import compile_policy
from nspe.policy.loader import load_policy

_POLICY_PATH = (
    Path(__file__).parent.parent / "nspe" / "policies" / "hateful_memes.yaml"
)


class TestHatefulMemesPolicy(TestCase):
    def test_loads(self):
        policy = load_policy(_POLICY_PATH)
        self.assertEqual(policy.name, "hateful_memes_policy")
        self.assertEqual(len(policy.rules), 4)

    def test_compiles(self):
        policy = load_policy(_POLICY_PATH)
        rt = compile_policy(policy)
        self.assertEqual(rt.num_base, 6)
        self.assertEqual(set(rt.verdict_names), {"hateful"})
        self.assertGreaterEqual(rt.num_iterations, 2)

    def test_predicate_names(self):
        policy = load_policy(_POLICY_PATH)
        base_names = set(policy.predicate_names("base"))
        self.assertEqual(
            base_names,
            {
                "slur_present",
                "targets_protected_group",
                "dehumanizing_comparison",
                "condemnation_context",
                "mocking_tone",
                "benign_context",
            },
        )
        self.assertEqual(policy.predicate_names("derived"), ("hate_speech_signal",))
        self.assertEqual(policy.predicate_names("verdict"), ("hateful",))


if __name__ == "__main__":
    run_tests()
