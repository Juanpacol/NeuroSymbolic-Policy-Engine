"""Tests for nspe.policy.loader and the bundled example policy."""

from pathlib import Path

from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.policy.compiler import compile_policy
from nspe.policy.loader import load_policy, policy_from_dict

_EXAMPLE_POLICY = (
    Path(__file__).parent.parent / "nspe" / "policies" / "meta_community_standards.yaml"
)


class TestLoadPolicy(TestCase):
    def test_loads_example_policy(self):
        policy = load_policy(_EXAMPLE_POLICY)
        self.assertEqual(policy.name, "meta_community_standards_hate_speech")
        self.assertEqual(len(policy.rules), 4)

    def test_example_policy_compiles(self):
        policy = load_policy(_EXAMPLE_POLICY)
        rt = compile_policy(policy)
        self.assertEqual(rt.num_base, 7)
        self.assertEqual(set(rt.verdict_names), {"remove", "escalate"})
        self.assertGreaterEqual(rt.num_iterations, 2)

    def test_negated_literal_parses(self):
        data = {
            "name": "t",
            "predicates": [
                {"name": "a", "kind": "base"},
                {"name": "v", "kind": "verdict"},
            ],
            "rules": [{"id": "R1", "head": "v", "body": [{"not": "a"}]}],
        }
        policy = policy_from_dict(data)
        self.assertTrue(policy.rules[0].body[0].negated)
        self.assertEqual(policy.rules[0].body[0].predicate, "a")


if __name__ == "__main__":
    run_tests()
