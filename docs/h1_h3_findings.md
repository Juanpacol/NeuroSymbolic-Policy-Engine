# H1/H3 remediation: root causes, fixes, and first real results

Snapshot as of commit `b0eb9bf`. Read this before touching
`nspe/train/`, `nspe/eval/`, `nspe/consistency.py`, or
`nspe/extractor.py` -- it explains *why* those modules look the way
they do, which their docstrings only partially cover.

## The problem this fixed

A full train+eval run on the pre-remediation code produced results
below the majority-class baseline for both arms, with only 5 distinct
predicate activation signatures across 831 validation cases (of 64
possible), and H1 backwards (baseline more "consistent" than the
reasoner). These were not evidence against the hypotheses -- they were
evidence the reasoner path never trained, plus a confounded metric.

Root causes, measured directly on the compiled `hateful_memes.yaml`
policy:

1. **Verdict collapse at init.** Uniform `mu0=0.5` gave a "hateful"
   verdict of 0.1387; realistic init gave 0.1387 ± 0.0036 across a
   whole batch. Cause: the fused CLIP embedding is two L2-normalized
   vectors (~0.03 magnitude), a default `Linear(1024, 6)` gives logit
   std ~0.025, and the product t-norm compresses a verdict behind 4
   literals into a narrow band (reachable range `[3e-7, 0.9985]`). BCE
   against 0/1 labels then makes "raise the constant toward the base
   rate" the cheapest descent direction, and selecting checkpoints on
   validation BCE rewarded exactly that.
2. **Predicate collapse.** All 6 heads read the same embedding and get
   gradient from one scalar verdict, with nothing pushing them apart.
3. **A NaN gradient in `log1mexp`** (`nspe/logic/ops.py`) for
   `log_x > -1e-8`: `torch.where` evaluates both branches, and `0 * inf
   = NaN` poisoned the selected branch's gradient. Negation is the only
   gradient path for a predicate appearing solely under `unless:`
   (`condemnation_context`, `benign_context`), so this could silently
   freeze them. The existing tests only checked the forward value.
4. **The H1 metric was confounded.** `inconsistency_rate`/`purity` are
   trivially maximized by a model that predicts one class for every
   case (0.0 and 1.0 respectively) while carrying no information.
5. **Capacity was not matched.** Baseline was `Linear(1024, 1)` (1025
   params) vs. the reasoner's `Linear(1024, 6)` + fixed logic (6150
   params), despite a docstring claiming otherwise.

## What was implemented (in commit order)

- **`535fc85`** -- `nspe/calibration.py::VerdictCalibrator`: a strictly
  monotone log-odds affine map, bias-fitted to the training base rate
  before epoch 1 (so training starts at, not descends toward, the
  constant solution). Monotone ⇒ AUROC is provably invariant to it.
  `nspe/train/loop.py::train_model` now selects checkpoints on
  validation AUROC (`select_metric`, default `"auroc"`), with
  class-weighted BCE, AdamW + cosine schedule, early stopping, and
  seeding (`nspe/train/seed.py`). Embedding cache keys switched from
  `output_dim` (collides: ViT-B-32 and ViT-B-16 both have 512) to
  `(model_name, pretrained)`.
- **`375543d`** -- `nspe/trunk.py::PredicateTrunk`/`PredicateHead`:
  shared `Linear -> GELU -> LayerNorm -> Dropout` trunk (the LayerNorm
  fixes the ~0.03 feature-magnitude problem at its source) plus a
  per-predicate logit scale/bias. `NeuroSymbolicLayer.
  init_heads_from_descriptions` seeds a residual zero-shot path from
  each predicate's `description:` in the policy YAML -- the only
  per-predicate supervision available, since Hateful Memes has no
  predicate labels. `nspe/train/regularizers.py` adds anchor/
  decorrelation/activation-entropy auxiliary losses. The baseline
  (`nspe/baselines/neural_classifier.py`) now mounts on an identical
  trunk + latent predicate layer, differing from the reasoner only in
  the aggregation step (learned linear vs. the fixed policy). Also
  fixed the `log1mexp` NaN-gradient bug and gave `NeuroSymbolicLayer` a
  strictly interior output range (`mu_eps`), which together keep
  gradients live all the way to saturation.
- **`5a2422c`** -- `nspe/consistency.py::ConsistencyReport` gained
  `positive_rate`, `null_inconsistency` (closed-form
  marginal-preserving null, `2k(n-k)/(n(n-1))`), `adjusted_consistency`
  (chance-corrected: 1.0 perfect, 0.0 chance, negative worse-than-chance,
  `nan` if degenerate), `degenerate` (flags a single-class model, which
  is now disqualified rather than credited), and
  `signature_entropy`/`class_size_histogram`. `nspe/eval/diagnostics.py`
  adds per-predicate activation-rate/correlation stats and the observed
  signature distribution, printed and saved on every eval run.
  `compute_h3` reports AUROC as the headline metric and fits a
  per-model threshold on the given split instead of hardcoding 0.5.
- **`b0eb9bf`** -- `p_mean_segment` wired as an opt-in
  `--aggregate pmean` ablation (default stays the t-conorm).

## First real-data validation (3 seeds, ViT-L-14, validation split)

Ran via `docs/colab_h1_h3.md` on Kaggle T4. Full per-seed output lives
in `neuropolicy-analisis-3.ipynb` (not checked in); this is the
distilled result.

| | reasoner | baseline |
|---|---|---|
| AUROC | **0.7168 ± 0.0066** | 0.6872 ± 0.0097 |
| accuracy | **0.6237 ± 0.0056** | 0.5764 ± 0.0229 |
| num_classes (of 64) | 52 / 57 / 38 | -- |
| majority-class accuracy | 0.5680 | 0.5680 |

Per-seed `adjusted_consistency` (reasoner vs. baseline): **0.73 vs.
0.29**, **0.51 vs. 0.35**, **0.71 vs. 0.46** -- the reasoner wins H1 in
all 3 seeds, and neither arm is `degenerate` in any seed
(`positive_rate` stays in `[0.39, 0.50]`), so the win is not the
constant-model artifact the pre-remediation numbers had.

Read against the targets in the remediation plan (AUROC 0.68-0.72,
`num_classes` 20-45 of 64): both landed inside range. Both arms clear
majority-class for the first time.

### Open observations, not yet investigated

- **Baseline is far less stable across seeds** than the reasoner
  (accuracy std 0.023 vs. 0.006; seed 2 baseline drops to 0.546
  accuracy / 0.487 precision). Worth a sentence in the paper either
  way: it's suggestive that the fixed rule structure acts as a
  regularizer relative to a learned aggregator, but 3 seeds is not
  enough to claim that -- it needs more seeds or a stability metric
  before it's a result rather than an anecdote.
- **Late-training overfitting is visible but handled.** Validation loss
  climbs sharply past the best epoch in every run shown (e.g. one
  reasoner run: 0.89 → 2.06 from epoch 4 to epoch 9); early stopping on
  AUROC is cutting at the right point, but if `--patience` is loosened
  this will need weight decay retuning.

## What's still open (remediation plan phases not yet run)

- **More seeds.** 3 is the plan's floor, not a target; the baseline
  instability above is reason to go to 5+ before reporting a final
  number.
- **Phase 4 ablations**, implemented but not yet run:
  `--clip-model ViT-B-32-quickgelu` as a second reported row,
  `--learnable-confidence`, `--aggregate pmean`, `--lambda-anchor` sweep
  at `{0, 0.03, 0.1, 0.3}` (currently defaults to `0.1`, never varied).
- **Test split.** Explicitly gated in the plan until validation is
  stable across seeds -- do not touch `--split test` before that, and
  when it happens, pass `--threshold` fitted on validation rather than
  letting `compute_h3` fit-and-report on the same split.
- **H2 latency numbers** (`docs/colab_benchmark.md`) are a separate
  track and unaffected by any of this.

## Do-not-do list (carried over from the remediation plan; still applies)

- Don't train on `nspe.consistency.consistency_loss` and then report
  H1 for that model -- it directly optimizes H1's own objective.
- Don't lower `tau` below 0.5 to manufacture more equivalence classes.
- Don't switch the default t-norm off `"product"` for bigger gradients.
- Don't unfreeze CLIP -- breaks the H2 premise and the embedding cache.
- Don't retune `HM_VERDICT_MOCKING` (or any rule) to fit the benchmark;
  the policy is supposed to be the public Meta policy, not a fit rule
  set.
