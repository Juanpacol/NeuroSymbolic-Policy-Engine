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

## Real-data validation (5 seeds, two backbones, validation split)

Ran via `docs/colab_h1_h3.md` on Kaggle T4. Full per-seed output lives
in `neuropolicy-analisis-3.ipynb`/`results_b32_s*.json` (not checked
in); this is the distilled result. `--seed 0..4`, `--epochs 30`, other
flags at their defaults (`--lambda-anchor 0.1`, `--select-metric
auroc`).

| | ViT-L-14 reasoner | ViT-L-14 baseline | ViT-B-32 reasoner | ViT-B-32 baseline |
|---|---|---|---|---|
| AUROC | **0.7193 ± 0.0060** | 0.6866 ± 0.0096 | **0.6933 ± 0.0052** | 0.6693 ± 0.0076 |
| accuracy | 0.6253 ± 0.0170 | 0.5853 ± 0.0239 | 0.5930 ± 0.0184 | 0.5653 ± 0.0253 |
| adjusted_consistency | **0.6703 ± 0.1504** | 0.2998 ± 0.1025 | 0.4662 ± 0.1814 | 0.3835 ± 0.0586 |
| majority-class accuracy | 0.5680 | 0.5680 | 0.5680 | 0.5680 |

Both backbones/arms clear majority-class, and `positive_rate` stayed in
a non-degenerate range in every one of the 20 runs (5 seeds × 2 models
× 2 backbones) -- no arm ever won by predicting a single class.

### Two different findings, of different strength

**AUROC gap is the robust result.** Reasoner beats baseline on AUROC in
all 10 seeds across both backbones, gap always positive
(ViT-L-14: [0.0031, 0.0454, 0.0072, 0.0362, ...]; ViT-B-32: [0.0074,
0.0276, 0.0269, 0.0255, 0.0325]). This is the number to lead with for
H3.

**H1 (consistency) advantage is backbone-dependent, and this matters.**
Per-seed reasoner `adjusted_consistency`:

- ViT-L-14: `[0.7334, 0.5094, 0.7079, 0.5003, 0.9004]` -- reasoner wins
  **5/5** seeds against baseline.
- ViT-B-32: `[0.5646, 0.5617, 0.2074, 0.6945, 0.3028]` -- reasoner wins
  only **3/5** seeds; the baseline outright wins seeds 2 and 4 (0.377
  and 0.421 vs. the reasoner's 0.207 and 0.303 respectively).

Read together: with richer features (ViT-L-14) the reasoner's
structural advantage on consistency is large and never reverses; with
weaker features (ViT-B-32) that advantage shrinks by more than half and
becomes unreliable seed-to-seed. This is consistent with an intuitive
mechanism -- a fixed rule circuit only has an advantage if the
predicate layer under it is discriminating well -- but is reported here
as an observed correlation across two backbones, not a proven causal
claim; it would need a controlled sweep (e.g. degrading ViT-L-14
features synthetically) to go further than that.

**accuracy_gap is not reliable at either backbone.** Per-seed sign
flips at both ViT-L-14 and ViT-B-32 (e.g. ViT-B-32: `[-0.0181, +0.0325,
-0.0229, +0.0590, +0.0878]`). Don't report accuracy_gap as a headline
number; AUROC is threshold-free and doesn't have this problem.

### Open observations, not yet investigated

- **Baseline is less stable across seeds than the reasoner** at both
  backbones (accuracy std ~0.024-0.025 vs. ~0.017-0.019). Consistent
  with the H1 finding above -- a learned aggregator over the latent
  predicate layer appears to be a less stable solution than a fixed
  rule circuit over it -- but again, an observation worth a sentence in
  the paper, not yet a controlled result.
- **Late-training overfitting is visible but handled.** Validation loss
  climbs sharply past the best epoch in most runs (e.g. one ViT-L-14
  reasoner run: 0.89 → 2.06 from epoch 4 to epoch 9); early stopping on
  AUROC is cutting at the right point, but if `--patience` is loosened
  this will need weight decay retuning.
- **Checkpoint disk usage.** Each checkpoint now includes the full
  frozen-CLIP state dict (~1.7GB at ViT-L-14). Running 5 seeds × 2
  models at both backbones in one Kaggle session exhausted
  `/kaggle/working`'s disk quota mid-run (`RuntimeError: [enforce fail
  at inline_container.cc:668]` from `torch.save`); delete `.pt` files
  for checkpoints already evaluated (their `results_*.json` is what
  matters) before starting the next backbone/seed batch. Worth fixing
  properly later by not serializing frozen CLIP weights into every
  checkpoint.

## What's still open (remediation plan phases not yet run)

- **Phase 4 ablations not yet run**: `--learnable-confidence`,
  `--aggregate pmean`, `--lambda-anchor` sweep at `{0, 0.03, 0.1, 0.3}`
  (currently defaults to `0.1`, never varied). The backbone comparison
  (ViT-L-14 vs. ViT-B-32) above is done.
- **A controlled test of the backbone-dependence hypothesis** for H1,
  if it's worth pursuing further than the observational finding above.
- **Test split.** Explicitly gated in the plan until validation is
  stable across seeds -- now that it is (5 seeds, two backbones), this
  is unblocked, but pass `--threshold` fitted on validation rather than
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
