"""CPU vs MPS parity tests for PolicyKGReasoner."""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.policy.schema import Literal, Policy, Predicate, Rule
from nspe.reasoner import PolicyKGReasoner

_MPS_AVAILABLE = torch.backends.mps.is_available()


def _policy_with_exceptions() -> Policy:
    return Policy(
        name="parity",
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
                id="R1",
                head="hate",
                body=(Literal("slur"), Literal("target")),
                unless=(Literal("edu"),),
                confidence=0.9,
            ),
            Rule(
                id="R2",
                head="remove",
                body=(Literal("hate"),),
                unless=(Literal("news"),),
            ),
        ),
    )


class TestDeviceParity(TestCase):
    def test_cpu_and_mps_agree(self):
        if not _MPS_AVAILABLE:
            self.skipTest("MPS not available on this machine")

        policy = _policy_with_exceptions()
        torch.manual_seed(0)
        mu0_cpu = torch.rand(32, 4)

        reasoner_cpu = PolicyKGReasoner(policy, tnorm="product")
        out_cpu = reasoner_cpu(mu0_cpu)

        reasoner_mps = PolicyKGReasoner(policy, tnorm="product").to("mps")
        out_mps = reasoner_mps(mu0_cpu.to("mps"))

        torch.testing.assert_close(
            out_cpu.mu, out_mps.mu.to("cpu"), rtol=1e-4, atol=1e-5
        )
        for name in policy.predicate_names("verdict"):
            torch.testing.assert_close(
                out_cpu.verdicts[name],
                out_mps.verdicts[name].to("cpu"),
                rtol=1e-4,
                atol=1e-5,
            )

    def test_gradients_agree_across_devices(self):
        if not _MPS_AVAILABLE:
            self.skipTest("MPS not available on this machine")

        policy = _policy_with_exceptions()
        torch.manual_seed(1)
        mu0 = torch.rand(8, 4)

        reasoner_cpu = PolicyKGReasoner(policy, tnorm="product")
        x_cpu = mu0.clone().requires_grad_(True)
        reasoner_cpu(x_cpu).verdicts["remove"].sum().backward()

        reasoner_mps = PolicyKGReasoner(policy, tnorm="product").to("mps")
        x_mps = mu0.clone().to("mps").requires_grad_(True)
        reasoner_mps(x_mps).verdicts["remove"].sum().backward()

        torch.testing.assert_close(
            x_cpu.grad, x_mps.grad.to("cpu"), rtol=1e-3, atol=1e-4
        )


if __name__ == "__main__":
    run_tests()
