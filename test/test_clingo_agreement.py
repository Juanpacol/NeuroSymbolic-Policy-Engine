"""Requires exact agreement between the fuzzy reasoner (crisp inputs)
and Clingo's stable-model semantics.

This is the project's single most important credibility artifact for
H2: if the fuzzy reasoner does not reproduce Clingo's answers on crisp
inputs, no benchmark latency comparison between them means anything.
"""

import random

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.baselines.clingo_engine import ClingoEngine
from nspe.reasoner import PolicyKGReasoner
from test.test_reasoner import _crisp_oracle, _make_random_acyclic_policy


class TestClingoAgreesWithReasoner(TestCase):
    def test_agreement_across_random_policies(self):
        mismatches = []
        for seed in range(60):
            policy = _make_random_acyclic_policy(seed)
            engine = ClingoEngine(policy)
            reasoner = PolicyKGReasoner(policy, tnorm="crisp")
            base_names = policy.predicate_names("base")

            rng = random.Random(seed + 10_000)
            facts = {b for b in base_names if rng.random() < 0.5}

            clingo_atoms = engine.infer(facts)
            oracle_atoms = _crisp_oracle(policy, facts)

            mu0 = torch.tensor([[1.0 if b in facts else 0.0 for b in base_names]])
            out = reasoner(mu0)

            for name, idx in reasoner.rule_tensor.name_to_index.items():
                reasoner_true = out.mu[0, idx].item() >= 0.5
                clingo_true = name in clingo_atoms
                oracle_true = name in oracle_atoms
                if not (reasoner_true == clingo_true == oracle_true):
                    mismatches.append(
                        (seed, name, reasoner_true, clingo_true, oracle_true)
                    )

        self.assertEqual(
            mismatches,
            [],
            f"reasoner/clingo/oracle disagreement (seed, name, reasoner, "
            f"clingo, oracle): {mismatches[:10]}",
        )

    def test_agreement_on_bundled_example_policy(self):
        from nspe.policy.loader import load_policy

        policy = load_policy("nspe/policies/meta_community_standards.yaml")
        engine = ClingoEngine(policy)
        reasoner = PolicyKGReasoner(policy, tnorm="crisp")
        base_names = policy.predicate_names("base")

        rng = random.Random(0)
        for trial in range(20):
            facts = {b for b in base_names if rng.random() < 0.4}
            clingo_atoms = engine.infer(facts)
            mu0 = torch.tensor([[1.0 if b in facts else 0.0 for b in base_names]])
            out = reasoner(mu0)
            for name in policy.predicate_names("verdict"):
                idx = reasoner.rule_tensor.name_to_index[name]
                reasoner_true = out.mu[0, idx].item() >= 0.5
                clingo_true = name in clingo_atoms
                self.assertEqual(
                    reasoner_true,
                    clingo_true,
                    f"trial={trial} verdict={name}",
                )


class TestInferVerdictsMatchesInfer(TestCase):
    """The benchmark times ``infer_verdicts``; agreement is proved for
    ``infer``. If the fast path ever diverged, the certified comparison
    and the timed one would quietly come apart.
    """

    def test_agreement_across_random_policies(self):
        for seed in range(30):
            policy = _make_random_acyclic_policy(seed)
            engine = ClingoEngine(policy)
            base_names = policy.predicate_names("base")
            verdict_names = set(policy.predicate_names("verdict"))
            rng = random.Random(seed)

            for _ in range(5):
                facts = {b for b in base_names if rng.random() < 0.5}
                self.assertEqual(
                    engine.infer_verdicts(facts),
                    engine.infer(facts) & verdict_names,
                    f"seed={seed} facts={sorted(facts)}",
                )


if __name__ == "__main__":
    run_tests()
