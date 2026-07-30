# NeuroSymbolic Policy Engine (`nspe`)

Differentiable neurosymbolic reasoning over policy knowledge graphs, for consistent and
auditable content moderation.

`nspe` compiles a set of logical policy rules (with exceptions) into tensor buffers and
runs a batched, GPU-native fuzzy forward-chaining fixpoint entirely inside the PyTorch
autograd graph. This makes policy reasoning:

- **Differentiable end-to-end** — gradients flow from a verdict back through the fired
  rules to the neural predicates that produced them.
- **Auditable** — every verdict comes with a rule -> predicate -> verdict explanation
  chain, extracted at near-zero cost from the same forward pass.
- **Device-agnostic** — identical code path on CPU, Apple Silicon (MPS), and CUDA.

This is a standalone research library, not a PyTorch core contribution. It is not
affiliated with Meta, and the bundled example policy is derived from Meta's *public*
Community Standards documents only — it is our reading of those public documents, not
Meta's internal enforcement logic.

## Status

Early development. See `docs/` for the research question, hypotheses, and literature
review this project is built on.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10.

## Quick example (target API)

```python
from nspe import load_policy, PolicyKGReasoner, ConsistencyChecker

policy = load_policy("nspe/policies/meta_community_standards.yaml")
reasoner = PolicyKGReasoner(policy, tnorm="product")

out = reasoner(mu0)  # mu0: (batch, num_base_predicates), requires_grad=True
out.verdicts["remove"].sum().backward()

explanation = reasoner.explain(out, targets=["remove"])[0]
print(explanation.render())
```

## License

MIT.
