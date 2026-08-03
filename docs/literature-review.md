# Literature Review — NeuroSymbolic Content Policy Validator

## Project Summary

**Title:** Differentiable Neurosymbolic Reasoning for Consistent and Auditable Content Moderation

A standalone PyTorch-ecosystem library (not PyTorch core) implementing differentiable
neurosymbolic reasoning — GPU-native fuzzy logic over a policy Knowledge Graph — for
content moderation consistency and auditability.

## Research Question

Can a fuzzy-logic symbolic reasoner implemented as differentiable PyTorch layers on GPU
detect content-moderation inconsistencies and produce auditable explanations, with lower
latency and better consistency than (a) an end-to-end neural classifier and (b) a hybrid
pipeline using a non-differentiable external rule engine (Prolog/ASP)?

## Sub-Hypotheses

- **H1 (Consistency):** The symbolic reasoner yields fewer contradictory verdicts on
  cases with equivalent activated predicates than an end-to-end neural baseline.
- **H2 (Performance):** The GPU-native fuzzy logic layer achieves lower latency / higher
  throughput than an external rule engine (ASP/Prolog).
- **H3 (Explainability without accuracy loss):** An auditable rule → predicate → verdict
  chain is achievable without significant accuracy loss versus the end-to-end baseline.

## Identified Research Gap

None of the first five sources reviewed address symbolic/KG reasoning made
**differentiable and GPU-native inside the PyTorch autograd graph** — all treat KG
reasoning as external to, or separate from, neural training. This is the gap the
project fills, mapping directly to **H2**. Sources 6-9 below close a second gap
flagged in `CLAUDE.md`: no fuzzy-logic/t-norm-specific sources had been reviewed
before implementing `nspe/logic/tnorm.py`, and no Hateful-Memes-specific prior work
had been reviewed before choosing that benchmark for H1/H3.

## Source Summary Table

| # | Source | Venue | Maps to |
|---|--------|-------|---------|
| 1 | A Survey on Augmenting Knowledge Graphs with LLMs | Springer, 2024 | Related Work / Motivation |
| 2 | Can Knowledge Graphs Reduce Hallucinations in LLMs? | NAACL 2024 | H1 |
| 3 | Neural, Symbolic and Neural-Symbolic Reasoning on Knowledge Graphs | AI Open (Elsevier), 2021 | H3 / Theoretical Framework |
| 4 | Neural-Symbolic Reasoning over Knowledge Graphs: A Survey from a Query Perspective | ACM SIGKDD Explor., 2025 | Methodology (PolicyKGReasoner design) |
| 5 | Unifying Large Language Models and Knowledge Graphs: A Roadmap | arXiv | Related Work / Motivation |
| 6 | Analyzing Differentiable Fuzzy Logic Operators | Artificial Intelligence (Elsevier), 2022 | H2 / t-norm design (`nspe/logic/tnorm.py`) |
| 7 | T-Norms Driven Loss Functions for Machine Learning | Applied Intelligence (Springer), 2023 | H2 / loss-function design (`nspe/train/loop.py`) |
| 8 | Uncertainty-Guided Modal Rebalance for Hateful Memes Detection | ACL 2024 | Related Work (Hateful Memes baselines) |
| 9 | HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection | AAAI 2021 | H3 / Explainability precedent |

## Source Details

### 1. A Survey on Augmenting Knowledge Graphs with LLMs (Springer, *Discover Artificial
Intelligence*, 2024) — Ibrahim, Aboulela, Ibrahim, Kashef

Key contributions:
- Knowledge graphs (KGs) provide structured, verifiable knowledge that complements the
  implicit knowledge of LLMs.
- Integration improves: interpretability, accuracy, factual consistency, reasoning
  capability.
- Classifies integration into three paradigms: **KG-enhanced LLMs**, **LLM-augmented
  KGs**, and **Synergized LLMs + KG** (bidirectional/mutual integration).
- Identifies challenges: scalability, computational overhead, data privacy, and
  maintaining up-to-date KGs.

Technical detail (from full read of the paper):

- **KG-Enhanced LLMs** — embedding KG entities/relations into continuous vector space so
  an LLM can use them at training or inference time. Sub-categorized by phase:
  - *Pre-training*: KG knowledge is exposed to the LLM during pre-training.
  - *Inference*: LLM retrieves from a KG at query time (RAG-style) without retraining.
  - *Interpretability*: KGs used to trace/understand the LLM's reasoning process.
  - Representative models: **KEPLER**, **Pretrain-KGE** — use BERT-like encoders to embed
    textual descriptions of KG entities/relations into vectors, then fine-tune on
    KG-related tasks.
  - Fine-tuning pipeline: NER + relation extraction → embed into vector space via
    **node2vec** or **Graph Neural Networks (GNNs)** → fine-tune LLM on this
    graph-structured data → improves grounding, reduces hallucination, increases
    reasoning capability.
  - Applications: healthcare (diagnostics via medical KGs), finance (risk
    assessment/fraud detection), e-commerce (recommendation via product/customer KGs).

- **LLM-Augmented KGs** — using the LLM's generalization ability to improve KG
  construction/maintenance itself. Two stages:
  - *KG construction*: coreference resolution, NER, relationship extraction from raw
    text to build the graph.
  - *KG utilization*: KG completion (filling missing facts), KG question-answering, KG
    text generation (natural-language description of graph facts).

- **Synergized LLMs + KG** — mutual/bidirectional integration into one framework
  (both technologies co-improve each other).

- KG types taxonomy (Table 3 in paper): domain-specific (e.g., SNOMED CT, FIBO),
  cross-domain (Google KG, DBpedia), enterprise/internal, and open (Wikidata).

- Business use cases discussed: search engines, recommendation systems, clinical
  decision support, supply chain management, fraud detection, CRM — useful as
  precedent/analogy for framing a **policy KG** as a domain-specific/enterprise KG
  applied to content moderation.

Citable claim: *Knowledge graphs improve interpretability and help generate more
consistent responses when integrated with LLMs.*

Relevance to this project: The **KG-Enhanced LLM (interpretability sub-category)** and
**fine-tuning-via-embedded-KG pipeline** (NER/relation extraction → node2vec/GNN
embedding → fine-tune) is the closest existing precedent to your neural predicate
extractor + policy KG design — but note it is *not* framed as differentiable/GPU-native
reasoning integrated into autograd, which remains your gap (H2).

### 2. Can Knowledge Graphs Reduce Hallucinations in LLMs? A Survey (NAACL 2024) —
Agrawal et al., arXiv:2311.07914

Key contributions:
- LLMs hallucinate due to limitations in their internal/parametric knowledge.
- KGs enable retrieval of verifiable information prior to generation, rather than
  relying solely on model parameters.
- Organizes KG-based augmentation approaches into three overarching groups (method
  classification for incorporating KGs into LLMs), comparing trade-offs across them.
- Main finding: leveraging KGs as external structured knowledge demonstrates promising
  results in mitigating hallucinations and improving reasoning performance.

Citable claim: *Knowledge graphs increase the factual consistency of language models by
reducing hallucinations.*

Relevance to this project: directly supports **H1** — grounding the neural predicate
extractor's outputs against a fixed policy KG (rather than trusting free-form LLM
judgment) is the mechanism expected to reduce contradictory/hallucinated moderation
verdicts.

### 3. Neural, Symbolic and Neural-Symbolic Reasoning on Knowledge Graphs (Zhang, Chen,
Zhang, Ke, Ding — *AI Open*, Elsevier, vol. 2, pp. 14–35, 2021; open version:
arXiv:2010.05446)

Abstract summary: KGs are discrete symbolic representations, so KG reasoning naturally
supports symbolic techniques, but symbolic reasoning is intolerant of ambiguous/noisy
data. Neural reasoning (embeddings) is robust to noise but lacks interpretability. The
survey unifies both into a shared reasoning framework, evaluated on two tasks: **KG
completion** and **KG question-answering**.

Taxonomy and methods:
- **Symbolic methods** — explicit logic rules/inference; high interpretability, brittle
  under incomplete/noisy data.
- **Neural methods** — embedding-based (e.g., **TransE**, **DistMult**); robust to noise,
  black-box, low interpretability.
- **Neural-symbolic integration** — three sub-strategies: (a) *rule-enhanced embeddings*
  (logical constraints injected into neural models), (b) *embedding-augmented reasoning*
  (learned representations guide symbolic inference), (c) *joint frameworks* (jointly
  optimize logical consistency and embedding quality via constraint regularization).

Citable claim: *Symbolic reasoning contributes interpretability via explicit rules;
neural-symbolic integration keeps rule transparency while gaining neural robustness to
noisy/incomplete data — the two properties trade off unless combined.*

Relevance to this project: gives the direct theoretical basis for choosing **fuzzy-logic
rule-enhanced reasoning** (rather than pure embeddings or pure symbolic rules) as the
`PolicyKGReasoner` mechanism — supports **H3**.

### 4. Neural-Symbolic Reasoning over Knowledge Graphs: A Survey from a Query Perspective
(Liu et al., *ACM SIGKDD Explorations Newsletter*, published 2025-07-07;
doi:10.1145/3748239.3748249)

Abstract summary: KG reasoning is central to data mining, AI, Web, and social sciences.
Traditional symbolic reasoning struggles with incomplete/noisy KGs. Neural-Symbolic AI
merges deep learning robustness with symbolic precision, aiming for AI systems that are
interpretable, explainable, and versatile — bridging symbolic and neural methodologies,
organized from a **query-answering perspective** (i.e., classifying methods by the type
of query/inference task they solve over the graph rather than by architecture alone).

Citable claim: *Neural-symbolic integration, viewed through the lens of query answering,
improves accuracy and explainability jointly rather than trading one for the other.*

Relevance to this project: the "query perspective" framing is useful methodologically —
frame the `PolicyKGReasoner`'s inference step as answering a structured query
("does content C violate policy P, and via which rule chain?") rather than as a generic
classification, aligning with how this survey structures the reasoning task.

### 5. Unifying Large Language Models and Knowledge Graphs: A Roadmap (Pan et al., IEEE
Transactions on Knowledge and Data Engineering, 2024; arXiv:2306.08302)

Three proposed frameworks:
- **KG-enhanced LLMs** — KGs incorporated during pre-training/inference to fix LLMs'
  weakness as black-box models that "fall short of capturing and accessing factual
  knowledge."
- **LLM-augmented KGs** — LLMs used for KG embedding, completion, construction,
  graph-to-text generation, and QA, addressing that "KGs are difficult to construct and
  evolving by nature."
- **Synergized LLMs + KGs** — bidirectional reasoning driven by both data and knowledge,
  treating LLMs and KGs as equal, mutually-enhancing components.

Core insight: LLMs generalize well but lack interpretable factual grounding; KGs give
structured precision but struggle to generate new facts — unifying both yields a system
with both emergent capability and explicit knowledge representation.

Citable claim: *Knowledge graphs compensate for the black-box nature of LLMs by
providing structured, traceable knowledge.*

### 6. Analyzing Differentiable Fuzzy Logic Operators (*Artificial Intelligence*, 302,
2022) — van Krieken, Acar, van Harmelen

Key contributions:
- Formal and empirical analysis of a large collection of fuzzy logic operators
  (t-norms, t-conorms, implications) specifically for suitability in gradient-based,
  differentiable learning — not just their classical logical properties.
- Central finding: many textbook-standard fuzzy operators are **unsuitable for
  gradient-based training**. In particular, fuzzy implications exhibit severe gradient
  imbalance between antecedent and consequent terms.
- Achieving good learning performance under weak/semi-supervision requires operator
  combinations that deviate from classical logical laws — i.e., "logically correct" is
  not the same as "learnable."
- Introduces **sigmoidal implications**, a new operator family designed to fix the
  antecedent/consequent gradient-imbalance problem.

Technical detail relevant to this project:

- This is the direct theoretical grounding for a design decision already made and
  documented in `nspe/logic/tnorm.py`'s module docstring — that the product t-norm was
  chosen over Gödel specifically because Gödel-style min/max operators route gradient
  to only one literal per conjunction (a hard-argmax selection), which the codebase's
  own comment already attributes to "van Krieken" without a full citation. This source
  is that citation.
- It also substantiates a finding from the H1/H3 remediation work
  (`docs/h1_h3_findings.md`): the reasoner's own dead-gradient bug (predicates reachable
  only through `unless:` clauses receiving zero gradient near saturation, fixed in
  commit `375543d`) is a specific instance of the general gradient-pathology class this
  paper catalogs for fuzzy operators near the boundary of `[0, 1]`.
- The paper's core warning — that operators satisfying classical fuzzy-logic axioms can
  still be poor choices for learning — is a citable justification for *why* this
  project's t-norm choice needed empirical validation (`test/test_reasoner_gradients.py`)
  rather than being settled by logical correctness alone.

Citable claim: *Fuzzy logic operators that are logically well-behaved are not
guaranteed to be well-behaved for gradient-based learning, and this gap must be
analyzed and addressed explicitly when embedding fuzzy logic inside a differentiable
system.*

Relevance to this project: This is the primary citation for the t-norm design choice
underlying `PolicyKGReasoner`, closing the gap `CLAUDE.md` flagged as missing before
the fuzzy reasoning layer was implemented.

### 7. T-Norms Driven Loss Functions for Machine Learning (*Applied Intelligence*,
Springer, 2023; arXiv:1907.11468) — Marra, Giannini, Diligenti, Maggini, Gori

Key contributions:
- Establishes a formal, general connection between the choice of **t-norm generator**
  and the resulting **loss function** in neural-symbolic learning: once a t-norm
  generator is fixed, the loss function for a logical constraint is unambiguously
  determined.
- Shows this framework gives a **theoretical justification for cross-entropy loss**
  as a special case, rather than treating it as an arbitrary engineering choice.
- Extends beyond single-label classification to loss functions for arbitrary
  First-Order Logic knowledge expressed via fuzzy logic, with empirical results showing
  faster convergence than ad hoc alternatives.

Technical detail relevant to this project:

- Directly relevant to `nspe/train/loop.py::_weighted_bce`, which trains the reasoner's
  fuzzy verdict against a binary label via BCE. This paper supplies the theoretical
  frame for treating that choice as principled — the verdict is itself the output of a
  t-norm-based logical derivation, and BCE against the ground-truth label is the
  generator-consistent loss for that construction, rather than an arbitrary choice of
  "treat the fuzzy output like a probability."
- Also relevant context for `nspe/calibration.py::VerdictCalibrator`: this project
  found empirically that the raw fuzzy verdict is poorly scaled for BCE (compressed to
  ~0.14 near init, per `docs/h1_h3_findings.md`) and added a learned monotone
  calibration layer to fix it. This paper's generator-driven framework is a candidate
  theoretical lens for *why* a raw t-norm output and a well-behaved cross-entropy
  target are not automatically the same distributional object, and could motivate a
  more principled calibration derivation than the empirical fix currently in place.

Citable claim: *The choice of t-norm generator in a neural-symbolic system
determines, rather than merely influences, the correct loss function for training it
— and standard cross-entropy is recoverable as a special case of this framework.*

Relevance to this project: Secondary but genuine theoretical support for H2's
engineering claim — it strengthens the argument that `PolicyKGReasoner`'s training
objective (BCE on a t-norm-derived verdict) is a principled instance of a known
framework, not an ad hoc choice.

### 8. Uncertainty-Guided Modal Rebalance for Hateful Memes Detection (ACL 2024) —
Yang, Liu, Zhu, Han, Hu

Key contributions:
- Proposes UMR (Uncertainty-guided Modal Rebalance), a multimodal architecture for
  hateful memes detection that models per-modality uncertainty with Gaussian-distributed
  stochastic representations, adaptively reweighting visual and textual features.
- Diagnoses a specific failure mode in prior multimodal hateful-memes work: models
  over-rely on fused cross-modal features and ignore modality-specific uncertainty,
  producing imbalanced reliance on image vs. text.
- Reports state-of-the-art results across four hateful-memes-adjacent datasets using an
  improved cosine-loss constraint to correct the imbalance.

Technical detail relevant to this project:

- This is a **pure end-to-end neural baseline** for the exact task this project
  benchmarks against (Hateful Memes) — it belongs in Related Work as precedent for
  `nspe.baselines.neural_classifier.NeuralBaselineClassifier`'s design space, and as
  evidence that even strong 2024 neural approaches still explicitly frame the problem
  as "how do we better fuse/weight modalities," not "how do we make the decision
  auditable" — the latter being this project's differentiator (H3).
- Does **not** engage with neurosymbolic reasoning or knowledge graphs at all, so it
  does not narrow the H2 gap; its value is calibrating what a competitive purely-neural
  Hateful Memes system looks like, for framing how strong a baseline this project's
  `NeuralBaselineClassifier` should be expected to be.

Citable claim: *State-of-the-art hateful memes detection as of 2024 is pursued
through modality-uncertainty-aware feature fusion, with no attention to decision
auditability or symbolic consistency guarantees.*

Relevance to this project: Related-work anchor establishing that the neural baseline
this project compares against is representative of the field's actual state of the art
in raw accuracy terms, and that H3's auditability claim is not addressed by competing
approaches.

### 9. HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection (AAAI 2021)
— Mathew, Saha, Yimam, Biemann, Goyal, Mukherjee

Key contributions:
- Introduces a hate-speech dataset with three annotation layers per post: the
  classification label (hate/offensive/normal), the targeted community, and a
  **rationale** — the specific span of text the annotator used to justify the label.
- Empirically shows a **gap between accuracy and explainability**: models that score
  highest on classification accuracy do *not* score highest on explainability metrics
  (plausibility and faithfulness of their rationales) — the two are not correlated by
  default and must be evaluated separately.
- Shows that training with human rationales as supervision improves both explainability
  metrics and reduces unintended bias toward specific target communities.

Technical detail relevant to this project:

- This is the strongest available precedent for H3's core premise — that accuracy and
  explainability are *separate axes that must both be measured*, not a single metric
  where good accuracy implies good explanations. This project's H3 (no significant
  accuracy loss *and* an auditable rule → predicate → verdict chain,
  `nspe/explain.py`) is structurally the same two-axis claim HateXplain's evaluation
  methodology is built around, applied to symbolic rather than extractive-rationale
  explanations.
- HateXplain's rationale spans are extractive (which words matter); this project's
  explanations are structural (which rules and predicates fired, with confidence and
  defeat by exceptions) — a citable methodological contrast worth stating explicitly in
  the paper: this project's explanations are *policy-grounded* rather than
  *text-grounded*, trading token-level faithfulness for auditability against a named,
  versioned policy document.
- Their finding that accuracy-optimized models are not automatically explainable
  models is direct external evidence motivating why H3 needed to be stated and tested
  as its own hypothesis, rather than assumed to follow from H1/accuracy parity.

Citable claim: *High classification accuracy in hate speech detection does not imply
high explainability, and explainability must be evaluated as an independent property
of the model rather than inferred from accuracy.*

Relevance to this project: Primary external justification for treating H3 as a
distinct, separately-measured hypothesis, and a natural citation point for contrasting
this project's rule-grounded explanations against extractive-rationale explainability
work in the same problem domain.

## Overall Conclusion Supported by These Sources

- **Interpretability** improves because knowledge graphs and symbolic reasoning allow
  tracing the origin of information and the rules used to infer answers.
- **Consistency** improves because graphs provide structured, verifiable knowledge,
  reducing hallucinations and increasing the factual fidelity of LLMs.
- **Symbolic reasoning** contributes explicit, rule-based inference, complementing the
  learning capabilities of LLMs and making the overall system more explainable and
  trustworthy.

Together, these nine works form a solid bibliographic foundation for arguing that
combining LLMs, knowledge graphs, and symbolic reasoning can improve the interpretability
and consistency of AI systems — while also flagging challenges such as scalability, graph
maintenance, and computational cost.

Sources 6-7 additionally ground the project's core H2 engineering choice: differentiable
fuzzy logic operators require analysis on their *learnability*, not just their classical
logical correctness, and a t-norm choice determines a principled loss function rather
than an arbitrary one. Sources 8-9 ground H3 specifically: HateXplain (9) provides direct
empirical evidence that accuracy and explainability are independent properties requiring
separate evaluation — the structural justification for why H3 is stated as its own
hypothesis rather than assumed — while the UMR paper (8) calibrates what a competitive
2024 purely-neural Hateful Memes baseline looks like, and confirms that even
state-of-the-art neural approaches in this exact task do not address auditability.

---

*This file is appended incrementally as new sources are reviewed, before implementation
begins.*
