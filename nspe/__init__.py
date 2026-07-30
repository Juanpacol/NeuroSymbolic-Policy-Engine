"""Differentiable neurosymbolic reasoning over policy knowledge graphs."""

from nspe.consistency import ConsistencyChecker, ConsistencyReport, consistency_loss
from nspe.explain import Explanation, attribution
from nspe.logic.tnorm import get_tnorm
from nspe.policy.loader import load_policy
from nspe.policy.schema import Literal, Policy, Predicate, Rule
from nspe.reasoner import PolicyKGReasoner, ReasonerOutput

__version__ = "0.1.0"

__all__ = [
    "ConsistencyChecker",
    "ConsistencyReport",
    "Explanation",
    "Literal",
    "Policy",
    "PolicyKGReasoner",
    "Predicate",
    "ReasonerOutput",
    "Rule",
    "attribution",
    "consistency_loss",
    "get_tnorm",
    "load_policy",
]
