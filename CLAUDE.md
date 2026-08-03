# NeuroSymbolic Policy Engine

## Project summary

A standalone PyTorch-ecosystem library (analogous to `torch_geometric`, not a PyTorch
core contribution) implementing **differentiable neurosymbolic reasoning** — GPU-native
fuzzy logic over a policy Knowledge Graph (KG) — applied to content moderation
consistency and auditability. Target paper: implementation/systems paper with empirical
benchmarks, not a pure research contribution to PyTorch core.

All project artifacts (code, docs, commit messages, comments) are written in **English**,
regardless of the language used in planning conversations.

## Paper title

**"Differentiable Neurosymbolic Reasoning for Consistent and Auditable Content
Moderation"**

## Research question

Can a fuzzy-logic symbolic reasoner implemented as differentiable PyTorch layers on GPU
detect content-moderation inconsistencies and produce auditable explanations, with lower
latency and better consistency than (a) an end-to-end neural classifier and (b) a hybrid
pipeline using a non-differentiable external rule engine (Prolog/ASP)?

## Sub-hypotheses

- **H1 (Consistency):** the symbolic reasoner yields fewer contradictory verdicts on
  cases with equivalent activated predicates than an end-to-end neural baseline.
- **H2 (Performance):** the GPU-native fuzzy logic layer achieves lower latency / higher
  throughput than an external rule engine (ASP/Prolog). This is the project's core
  engineering contribution and research gap — no reviewed source implements KG/symbolic
  reasoning as differentiable, GPU-native layers inside the PyTorch autograd graph.
- **H3 (Explainability without accuracy loss):** an auditable rule → predicate → verdict
  chain is achievable without significant accuracy loss versus the end-to-end baseline.

## Architecture direction (decided, not yet implemented)

- Standalone pip-installable package, `nn.Module`-based components:
  - `NeuroSymbolicLayer` — neural predicate extractor (multimodal: text + image).
  - `PolicyKGReasoner` — fuzzy-logic reasoning engine over a policy Knowledge Graph,
    implemented as vectorized tensor ops (t-norms) so it stays inside the autograd
    graph — no external solver (Prolog/ASP) in the inference path.
  - Consistency-checking module: compares inference chains across cases with equivalent
    predicates to flag divergent verdicts (the H1 mechanism).
- Policy KG built from **public** Meta Community Standards (no access to internal Meta
  data/KGs — do not imply otherwise in the paper).
- Evaluation dataset: **Hateful Memes Challenge** (public, multimodal, Meta/FAIR-released
  — gives legitimacy without requiring internal data access).
- Explicitly NOT targeting PyTorch core inclusion. Do not frame contributions, issues, or
  docs as if this were headed for a `pytorch/pytorch` PR.

## Literature foundation

Full literature review lives in `docs/literature-review.md` (English). Do not duplicate
its content here — read it for source details, citable claims, and how each source maps
to H1/H2/H3. Fuzzy-logic/t-norm grounding (van Krieken et al. 2022; Marra et al. 2023)
and Hateful-Memes-specific prior work (HateXplain, AAAI 2021; UMR, ACL 2024) are covered
as sources 6-9.

## Code style and conventions

Follow PyTorch's own contribution conventions (from
[CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)) as the
style baseline for this library, since it targets the PyTorch ecosystem/audience:

- **Docstrings:** Google style. Lines wrapped to 80 characters. Type info in round
  brackets after variable names in `Args:`/`Returns:`. Type naming rules: capitalize
  `Callable`, `Any`, `Iterable`, `Iterator`, `Generator`; lowercase `list`, `tuple`; never
  pluralize types ("tuple of int", not "tuple of ints"); only `or`/`of` as delimiters;
  `optional` only when a value isn't required; match Python type names (`str`, `bool`,
  `dict`); use `dict[str, int]` syntax; `or` for two types, commas for three-plus
  (`type1, type2, or type3`).
- **Type hints:** required on all public functions/methods, PEP 604 style (`int | None`,
  not `Optional[int]`).
- **Formatting/linting:** treat as `lintrunner`-equivalent discipline — consistent
  formatting enforced project-wide, no ad hoc style per file.
- **Tests:** in `test/`, filenames start with `test_`, one test module per source module,
  pytest-compatible (`pytest -k <name>` for selective runs). Build on
  `torch.testing._internal.common_utils` (`TestCase`, `run_tests`) — ships with the
  regular `torch` pip package, no PyTorch source clone required:

  ```python
  from torch.testing._internal.common_utils import run_tests, TestCase

  class TestFeature(TestCase):
      ...

  if __name__ == "__main__":
      run_tests()
  ```

  Use `@parametrize` for multi-input test cases; use
  `instantiate_device_type_tests` only if/when the library has real
  device-specific (CPU vs. CUDA) numeric behavior worth testing per-device.
- **Module design:** components are `nn.Module` subclasses following standard PyTorch
  patterns (`__init__` builds sub-modules/parameters, `forward` is pure computation, no
  hidden side effects, differentiability preserved end-to-end).
- **Comment/abstraction discipline** (adopted from PyTorch's own internal Claude
  guidance): minimize comments, code should self-document; avoid trivial
  one-use helper functions; prefer explicit state management over cleverness;
  match existing patterns in the codebase; keep single-line code readable
  (don't force line breaks that aren't needed); ASCII-only in new comments;
  prefer the simpler, more concise implementation when in doubt.
- **Commit messages:** explain the *why*/logic of the change, not a restatement
  of the diff; include a short Test Plan section describing how the change
  was verified.
- **No PyTorch-core-specific conventions** (e.g., C++ ATen/dispatcher rules, RFC process,
  `ghstack`, `@pytorchmergebot`, CUDA bindings/Dynamo-internals guidance) apply here —
  those govern modifying PyTorch's own source tree and are not relevant to a downstream
  library that merely depends on `torch`.

## Working language note

The user and assistant may plan/discuss in Spanish, but everything persisted to this
repository — code, comments, docstrings, commit messages, docs — must be written in
English.
