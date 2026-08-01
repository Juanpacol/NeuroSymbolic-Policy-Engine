"""The differentiable, GPU-native policy reasoner."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nspe.explain import Explanation
from nspe.explain import explain as _explain
from nspe.logic.ops import gather_literal_log
from nspe.logic.tnorm import TNorm, get_tnorm
from nspe.policy.compiler import compile_policy
from nspe.policy.rule_tensor import RuleTensor
from nspe.policy.schema import Policy


@dataclass
class ReasonerOutput:
    """The result of one :class:`PolicyKGReasoner` forward pass.

    Attributes:
        mu: truth degrees for every predicate, shape ``(batch, P)``.
        log_mu: ``log(mu)``, kept around since it is what the fixpoint
            actually computes.
        verdicts: mapping from verdict predicate name to its truth
            degree, shape ``(batch,)`` each.
        fire_trace: per-iteration, per-rule firing log-truth, shape
            ``(T, batch, num_rules)``; ``None`` if ``store_trace=False``
            or there are no rules.
        mu_trace: per-iteration ``log_mu``, shape ``(T, batch, P)``;
            ``None`` if ``store_trace=False``.
        rule_tensor: the compiled policy this output was produced from,
            kept for :meth:`PolicyKGReasoner.explain`.
        calibrated: verdicts mapped through a
            :class:`~nspe.calibration.VerdictCalibrator`, or ``None`` if
            the engine has no calibrator. Set by
            :class:`~nspe.engine.PolicyEngine`, not by the reasoner:
            calibration is about comparing against a binary label, while
            ``verdicts`` remains the truth degree an audit chain
            explains.
    """

    mu: Tensor
    log_mu: Tensor
    verdicts: dict[str, Tensor]
    fire_trace: Tensor | None
    mu_trace: Tensor | None
    rule_tensor: RuleTensor
    calibrated: dict[str, Tensor] | None = None


class PolicyKGReasoner(nn.Module):
    """Differentiable fuzzy forward-chaining reasoner over a policy KG.

    Compiles ``policy`` once at construction time into flat tensor
    buffers (see :mod:`nspe.policy.compiler`) and runs a fixed number of
    unrolled, monotone fixpoint iterations entirely with tensor ops, so
    the whole computation stays inside the autograd graph -- gradients
    flow from any verdict back to the input predicate truths.

    Args:
        policy: the policy to compile and reason over.
        tnorm: name of the t-norm family (``"product"`` by default; see
            :func:`nspe.logic.tnorm.get_tnorm`).
        eps: numerical floor for truth degrees, avoids ``log(0)``.
        learnable_confidence: if ``True``, rule confidences become
            trainable parameters instead of fixed buffers.
        store_trace: if ``True`` (default), retain per-iteration traces
            needed by :meth:`explain`. Set to ``False`` in production
            inference paths that do not need explanations.
    """

    body_idx: Tensor
    body_rule: Tensor
    exc_idx: Tensor
    exc_rule: Tensor
    head_idx: Tensor
    rule_stratum: Tensor
    fire_idx: Tensor
    fire_rule: Tensor
    fire_sign: Tensor
    log_rule_conf: Tensor

    def __init__(
        self,
        policy: Policy,
        tnorm: str = "product",
        eps: float = 1e-7,
        learnable_confidence: bool = False,
        store_trace: bool = True,
    ) -> None:
        super().__init__()
        rt = compile_policy(policy)
        self.rule_tensor = rt
        self.tnorm: TNorm = get_tnorm(tnorm)
        self.eps = eps
        self.store_trace = store_trace
        self.num_iterations = rt.num_iterations

        self.register_buffer("body_idx", rt.body_idx)
        self.register_buffer("body_rule", rt.body_rule)
        self.register_buffer("exc_idx", rt.exc_idx)
        self.register_buffer("exc_rule", rt.exc_rule)
        self.register_buffer("head_idx", rt.head_idx)
        self.register_buffer("rule_stratum", rt.rule_stratum)

        # A rule fires when its body holds AND none of its exceptions
        # hold. Both are conjunctions, so firing is a single conjunction
        # over the concatenated literal set. `exc_sign` is already
        # oriented (see rule_tensor.py) so gathering with it yields
        # log(1 - exception_truth) directly; concatenating it with the
        # body's own signed literals makes this exact for every t-norm
        # family, not just an additive approximation.
        self.register_buffer("fire_idx", torch.cat([rt.body_idx, rt.exc_idx]))
        self.register_buffer("fire_rule", torch.cat([rt.body_rule, rt.exc_rule]))
        self.register_buffer("fire_sign", torch.cat([rt.body_sign, rt.exc_sign]))

        if learnable_confidence:
            self.log_rule_conf = nn.Parameter(rt.log_rule_conf.clone())
        else:
            self.register_buffer("log_rule_conf", rt.log_rule_conf)

    @property
    def num_predicates(self) -> int:
        """Total number of predicates (base + derived + verdict)."""
        return self.rule_tensor.num_predicates

    @property
    def num_base_predicates(self) -> int:
        """Number of base predicates the extractor must supply."""
        return self.rule_tensor.num_base

    def forward(self, mu0: Tensor) -> ReasonerOutput:
        """Runs the fixpoint forward-chaining reasoner.

        Args:
            mu0: base predicate truth degrees, shape
                ``(batch, num_base_predicates)``, values in ``[0, 1]``.

        Returns:
            A :class:`ReasonerOutput` with truth degrees for every
            predicate and a per-verdict-name view of the result.

        Raises:
            ValueError: if ``mu0``'s last dimension does not match the
                policy's base predicate count.
        """
        rt = self.rule_tensor
        if mu0.shape[-1] != rt.num_base:
            raise ValueError(
                f"Expected mu0 with {rt.num_base} base predicates, got {mu0.shape[-1]}"
            )
        batch = mu0.shape[0]
        device, dtype = mu0.device, mu0.dtype

        log_mu = torch.full(
            (batch, rt.num_predicates), math.log(self.eps), device=device, dtype=dtype
        )
        log_mu = log_mu.clone()
        log_mu[:, : rt.num_base] = torch.log(mu0.clamp(min=self.eps, max=1.0))

        mu_trace: list[Tensor] = []
        fire_trace: list[Tensor] = []
        floor = math.log(self.eps)

        for t in range(self.num_iterations):
            if rt.num_rules > 0:
                literal_log = gather_literal_log(log_mu, self.fire_idx, self.fire_sign)
                log_body = self.tnorm.conj_segment(
                    literal_log, self.fire_rule, rt.num_rules
                )
                log_fire_all = self.log_rule_conf.unsqueeze(0) + log_body
                # Rules fire strictly in stratum order: a rule's body or
                # exception literals only ever reference predicates from
                # strictly lower strata, so masking every rule not in the
                # current stratum guarantees negation always reads an
                # already-stabilized value, never a same-sweep partial
                # one (which the monotone update below could never undo).
                in_stratum = self.rule_stratum == t
                log_fire = torch.where(in_stratum.unsqueeze(0), log_fire_all, floor)
                log_derived = self.tnorm.disj_segment(
                    log_fire, self.head_idx, rt.num_predicates
                )
            else:
                log_fire = log_mu.new_zeros((batch, 0))
                log_derived = torch.full_like(log_mu, floor)

            log_mu = self.tnorm.disj_pair(log_mu, log_derived)

            if self.store_trace:
                mu_trace.append(log_mu)
                fire_trace.append(log_fire)

        mu = torch.exp(log_mu)
        verdicts = {name: mu[:, rt.name_to_index[name]] for name in rt.verdict_names}
        return ReasonerOutput(
            mu=mu,
            log_mu=log_mu,
            verdicts=verdicts,
            fire_trace=torch.stack(fire_trace) if fire_trace else None,
            mu_trace=torch.stack(mu_trace) if mu_trace else None,
            rule_tensor=rt,
        )

    def explain(
        self,
        out: ReasonerOutput,
        targets: Sequence[str] | None = None,
        case_index: int = 0,
    ) -> list[Explanation]:
        """Builds rule -> predicate -> verdict explanations for one case.

        Args:
            out: output of a forward pass run with ``store_trace=True``.
            targets: predicate names to explain; defaults to every
                verdict predicate in the policy.
            case_index: which row of the batch to explain.

        Returns:
            One :class:`~nspe.explain.Explanation` per requested target.
        """
        return _explain(self, out, targets=targets, case_index=case_index)
