"""Tests for nspe.explain."""

import json

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.explain import attribution
from nspe.policy.schema import Literal, Policy, Predicate, Rule
from nspe.reasoner import PolicyKGReasoner


def _hate_speech_policy() -> Policy:
    return Policy(
        name="hs",
        predicates=(
            Predicate("slur", "base"),
            Predicate("target", "base"),
            Predicate("edu", "base"),
            Predicate("news", "base"),
            Predicate("hate", "derived"),
            Predicate("remove", "verdict"),
        ),
        rules=(
            Rule(
                id="HS1",
                head="hate",
                body=(Literal("slur"), Literal("target")),
                unless=(Literal("edu"),),
                confidence=0.95,
            ),
            Rule(
                id="REMOVE1",
                head="remove",
                body=(Literal("hate"),),
                unless=(Literal("news"),),
            ),
        ),
    )


class TestExplainStructure(TestCase):
    def test_explanation_cites_correct_rule_chain(self):
        policy = _hate_speech_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product")
        mu0 = torch.tensor([[0.9, 0.9, 0.05, 0.05]])
        out = reasoner(mu0)

        explanations = reasoner.explain(out, targets=["remove"])
        self.assertEqual(len(explanations), 1)
        exp = explanations[0]
        self.assertEqual(exp.verdict, "remove")
        self.assertEqual(exp.root.rule_id, "REMOVE1")
        self.assertEqual(len(exp.root.children), 1)
        self.assertEqual(exp.root.children[0].predicate, "hate")
        self.assertEqual(exp.root.children[0].rule_id, "HS1")

        hate_node = exp.root.children[0]
        child_names = {c.predicate for c in hate_node.children}
        self.assertEqual(child_names, {"slur", "target"})
        self.assertEqual({name for name, _ in hate_node.defeated_by}, {"edu"})
        self.assertEqual({name for name, _ in exp.root.defeated_by}, {"news"})

    def test_body_predicates_all_exceed_threshold_self_consistency(self):
        # Audit-trail self-consistency: if the cited rule fired strongly,
        # its cited body predicates should indeed be true.
        policy = _hate_speech_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product")
        mu0 = torch.tensor([[0.95, 0.95, 0.02, 0.02]])
        out = reasoner(mu0)
        exp = reasoner.explain(out, targets=["remove"])[0]
        self.assertGreater(exp.truth, 0.5)

        hate_node = exp.root.children[0]
        for child in hate_node.children:
            self.assertGreater(child.truth, 0.5)

    def test_render_produces_readable_text(self):
        policy = _hate_speech_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product")
        mu0 = torch.tensor([[0.9, 0.9, 0.05, 0.05]])
        out = reasoner(mu0)
        text = reasoner.explain(out, targets=["remove"])[0].render()
        self.assertIn("remove", text)
        self.assertIn("REMOVE1", text)
        self.assertIn("defeated_by: news", text)

    def test_policy_fingerprint_recorded(self):
        policy = _hate_speech_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product")
        mu0 = torch.tensor([[0.9, 0.9, 0.05, 0.05]])
        out = reasoner(mu0)
        exp = reasoner.explain(out, targets=["remove"])[0]
        self.assertEqual(exp.policy_fingerprint, reasoner.rule_tensor.fingerprint)

    def test_requires_stored_trace(self):
        policy = _hate_speech_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product", store_trace=False)
        mu0 = torch.tensor([[0.9, 0.9, 0.05, 0.05]])
        out = reasoner(mu0)
        with self.assertRaises(ValueError):
            reasoner.explain(out, targets=["remove"])


class TestAttribution(TestCase):
    def test_attribution_matches_manual_grad(self):
        policy = _hate_speech_policy()
        reasoner = PolicyKGReasoner(policy, tnorm="product")
        mu0 = torch.tensor([[0.7, 0.6, 0.1, 0.1]], requires_grad=True)
        out = reasoner(mu0)
        grad = attribution(out.verdicts["remove"], mu0)

        mu0_2 = mu0.detach().clone().requires_grad_(True)
        out2 = reasoner(mu0_2)
        out2.verdicts["remove"].sum().backward()
        torch.testing.assert_close(grad, mu0_2.grad)


class TestAsDict(TestCase):
    def test_round_trips_through_json(self):
        policy = _hate_speech_policy()
        reasoner = PolicyKGReasoner(policy)
        out = reasoner(torch.tensor([[0.9, 0.8, 0.1, 0.1]]))
        explanation = reasoner.explain(out)[0]

        # Must be JSON-serializable with no custom encoder: the whole
        # point is landing in a results file.
        payload = json.loads(json.dumps(explanation.as_dict()))

        self.assertEqual(payload["verdict"], explanation.verdict)
        self.assertEqual(payload["policy_fingerprint"], explanation.policy_fingerprint)
        self.assertIn("root", payload)

    def test_tree_shape_matches_the_rendered_chain(self):
        policy = _hate_speech_policy()
        reasoner = PolicyKGReasoner(policy)
        out = reasoner(torch.tensor([[0.9, 0.8, 0.1, 0.1]]))
        explanation = reasoner.explain(out)[0]

        def count(node: dict) -> int:
            return 1 + sum(count(c) for c in node["children"])

        # One rendered line per node, plus one per defeated_by entry.
        def defeats(node: dict) -> int:
            return len(node["defeated_by"]) + sum(defeats(c) for c in node["children"])

        payload = explanation.as_dict()
        lines = len(explanation.render().splitlines())
        self.assertEqual(lines, count(payload["root"]) + defeats(payload["root"]))

    def test_leaf_carries_no_rule(self):
        policy = _hate_speech_policy()
        reasoner = PolicyKGReasoner(policy)
        out = reasoner(torch.tensor([[0.9, 0.8, 0.1, 0.1]]))
        root = reasoner.explain(out)[0].as_dict()["root"]

        # root is `remove` -> `hate` (derived, via HS1) -> base leaves.
        derived = root["children"][0]
        self.assertEqual(derived["kind"], "derived")
        self.assertEqual(derived["rule_id"], "HS1")
        for leaf in derived["children"]:
            self.assertEqual(leaf["kind"], "base")
            self.assertIsNone(leaf["rule_id"])
            self.assertIsNone(leaf["confidence"])
            self.assertEqual(leaf["children"], [])

    def test_records_the_concluding_rule(self):
        policy = _hate_speech_policy()
        reasoner = PolicyKGReasoner(policy)
        out = reasoner(torch.tensor([[0.9, 0.8, 0.1, 0.1]]))
        root = reasoner.explain(out)[0].as_dict()["root"]

        self.assertEqual(root["rule_id"], "REMOVE1")
        self.assertIsNotNone(root["confidence"])


if __name__ == "__main__":
    run_tests()
