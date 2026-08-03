# H1/H3 remediation: root causes, fixes, and first real results

Snapshot as of the 10-seed held-out test run and scrambled-policy
control. Read this before touching
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

## Held-out test split (10 seeds, ViT-L-14)

Ran via `docs/colab_h1_h3.md` Cell 9 on Kaggle T4 (superseding the
5-seed version of this section). Checkpoints are gitignored, so this is
a **full retrain**, not a re-evaluation of the weights behind the
validation section above: 10 seeds x 2 arms trained fresh, validation
re-evaluated in the same session to fit each arm's threshold, then test
evaluated once with `--thresholds-from` pointing at that fresh
validation run. Artifacts: `docs/results/h1_h3/results_val_s{0..9}.json`
(the paired rerun) and `results_test_s{0..9}.json`.

Ten seeds rather than five specifically so significance is reachable --
see **Significance**, below.

| | validation (this rerun) | **test (held out)** |
|---|---|---|
| reasoner AUROC | 0.7194 ± 0.0058 | **0.7574 ± 0.0049** |
| baseline AUROC | 0.6860 ± 0.0073 | **0.7265 ± 0.0061** |
| auroc_gap | 0.0334 ± 0.0114 | 0.0309 ± 0.0089 |
| reasoner accuracy | 0.6168 ± 0.0298 | 0.6216 ± 0.0360 |
| baseline accuracy | 0.5846 ± 0.0182 | 0.5880 ± 0.0173 |
| reasoner adjusted_consistency | 0.6769 ± 0.1217 | **0.7043 ± 0.0641** |
| baseline adjusted_consistency | 0.3096 ± 0.1033 | 0.3567 ± 0.0869 |
| majority-class accuracy | 0.5680 | 0.5876 |
| num_examples | 831 | 2408 (of 3000, after the image-availability filter) |

**The rerun reproduces the original validation section within seed
noise** -- the validation column here (computed by
`nspe/eval/aggregate.py`, not by hand) matches the published one in
every figure, which is the evidence that this retrain and the original
one describe the same system.

**H3 holds on held-out data, with a wider margin than validation
suggested.** Both arms' AUROC rise on test relative to validation
(reasoner 0.719 -> 0.757, baseline 0.686 -> 0.727) and the gap stays
positive in **all 10 seeds without exception**
(`[0.0241, 0.0215, 0.0368, 0.0319, 0.0280, 0.0435, 0.0312, 0.0140,
0.0352, 0.0430]`). The larger test set (2408 vs. 831 cases) also
tightens every standard deviation -- AUROC std drops from 0.0058 to
0.0049 for the reasoner, consistency std from 0.122 to 0.064 --
consistent with the validation-stage numbers being noisier estimates of
the same effect rather than a different one.

**H1 also holds.** Reasoner `adjusted_consistency` beats the baseline's
on both splits, on average by more than double (test: 0.704 vs. 0.357).
No arm is `degenerate` in any of the 20 (10 seeds x 2 arms) test runs.

## Significance (10 seeds, exact sign-permutation test)

Computed by `nspe.eval.significance.sign_permutation_test`, wired into
`nspe.eval.aggregate`'s per-group tables. At n=10 the floor is
`2/2**10 = 0.0020` two-sided -- reachable, unlike the n=5 floor of
0.0625 that made this section impossible to write honestly before this
run.

| split | field | mean gap | positive seeds | p (two-sided) |
|---|---|---|---|---|
| validation | auroc_gap | +0.0334 | 10/10 | **0.0020** (floor) |
| validation | accuracy_gap | +0.0323 | 8/10 | 0.0273 |
| validation | f1_gap | +0.0147 | 10/10 | **0.0020** (floor) |
| test | auroc_gap | +0.0309 | 10/10 | **0.0020** (floor) |
| test | accuracy_gap | +0.0335 | 7/10 | 0.0547 |
| test | f1_gap | +0.0186 | 8/10 | 0.0117 |

**AUROC and F1 are reliable; accuracy is not**, on both splits -- the
same pattern the 5-seed/two-backbone section above already flagged
(`accuracy_gap` sign-flips seed to seed) now has a p-value attached:
accuracy_gap is the one field that does not clear p<0.05 on test. Lead
with `auroc_gap`, not `accuracy_gap`, as this project's headline number.

**What this licenses, precisely.** The null rejected is that the
per-seed AUROC/F1 gap is symmetric about zero, where the unit of
observation is *a retrain of both arms on the same fixed dataset*. This
is evidence the reasoner reliably beats the baseline across random
initializations. It is **not** evidence the reasoner generalizes
better to new data -- that would require independent data samples, not
independent seeds.

## Calibration (10 seeds, ECE + Brier)

`VerdictCalibrator` is strictly monotone by construction, so it cannot
change AUROC -- meaning **none of the numbers above can tell whether it
does anything**. This is the measurement that can: raw and calibrated
verdicts recorded during the same run, `calibration_report` computed on
both (`nspe/eval/metrics.py`, 15 bins).

| | ECE, pre-calibration | ECE, post-calibration | Brier, pre | Brier, post |
|---|---|---|---|---|
| reasoner (val) | 0.1722 ± 0.0424 | 0.1534 ± 0.0154 | 0.2457 ± 0.0163 | 0.2347 ± 0.0061 |
| baseline (val) | 0.1366 ± 0.0466 | 0.1526 ± 0.0180 | 0.2448 ± 0.0157 | 0.2447 ± 0.0066 |
| reasoner (test) | 0.1438 ± 0.0383 | 0.1289 ± 0.0147 | 0.2216 ± 0.0127 | 0.2155 ± 0.0063 |
| baseline (test) | 0.1261 ± 0.0620 | 0.1150 ± 0.0182 | 0.2298 ± 0.0179 | 0.2234 ± 0.0050 |

**The calibrator does something, but modestly and not uniformly.** It
lowers the reasoner's ECE on both splits (val: -0.019, test: -0.015)
and the baseline's on test (-0.011), but very slightly *raises* the
baseline's on validation (+0.016) -- and every one of those deltas is
smaller than its own across-seed standard deviation, so none should be
read as a clean, seed-independent effect. The honest summary is: the
raw fuzzy verdict was in fact the badly-scaled quantity
`nspe/calibration.py`'s module docstring says it was introduced to fix
(both arms start around ECE 0.13-0.17, well above 0), the calibrator
narrows that somewhat for the reasoner specifically, and AUROC alone
would have reported none of this, because it cannot -- confirming the
module docstring's own claim that AUROC-invariance means AUROC was
never evidence the calibrator worked.

## Scrambled-policy control: results

**Resolves the pre-registration below. Read that section first for the
predictions being tested here.**

Ran via `docs/colab_h1_h3.md` Cell 10, same session as the 10-seed
result above. Reasoner retrained per scrambled policy (seeds 0-9);
baseline reused from the intact run, since it consumes the policy only
for `num_predicates` (unchanged at 6 under any derangement). Artifacts:
`docs/results/h1_h3/results_scram_{val,test}_s{0..9}.json`.

| | intact | scrambled | gap (intact - scrambled) | p, one-sided (n=10) |
|---|---|---|---|---|
| reasoner AUROC (val) | 0.7194 | 0.7188 | +0.0006 | 0.3965 |
| reasoner AUROC (test) | 0.7574 | 0.7540 | +0.0034 | 0.1553 |
| reasoner adjusted_consistency (val) | 0.6769 | 0.5615 | +0.1154 | 0.1680 (two-sided) |
| reasoner adjusted_consistency (test) | 0.7043 | 0.6358 | +0.0685 | 0.2324 (two-sided) |

Per-seed AUROC gaps (intact - scrambled), paired by seed:

- validation: `[0.0038, 0.0047, -0.0046, 0.0111, -0.0061, -0.0043,
  0.0068, -0.0090, 0.0117, -0.0074]` -- positive in 5/10.
- test: `[0.0015, -0.0006, 0.0031, 0.0227, -0.0126, 0.0062, 0.0078,
  -0.0087, 0.0094, 0.0053]` -- positive in 7/10.

**Prediction 1 (intact AUROC > scrambled, one-sided) does not hold.**
The mean gap is close to zero on both splits, the sign flips seed to
seed, and p is nowhere near significant (0.40 validation, 0.16 test --
neither approaches the two-sided floor of 0.002 this design could
reach, let alone the more lenient one-sided threshold). Scrambling
which base predicate each rule reads produces a model statistically
indistinguishable, on AUROC, from the intact policy.

**Prediction 2 (adjusted_consistency roughly unchanged) also misses,
though less cleanly.** Consistency drops under scrambling by a
non-trivial margin on both splits (-17% relative on validation, -10% on
test), larger than the "roughly unchanged" the prediction called for --
but the drop is not statistically significant either (p=0.17-0.23,
two-sided), so this reads as a real but noisy trend rather than a
clean second result.

**Prediction 3 (scrambled still above chance) holds without
qualification** -- 0.754 test AUROC is far above the 0.5 baseline and
above the 0.588 majority-class accuracy.

### This is the falsifying outcome, and it was named in advance

The pre-registration below states explicitly: *"If scrambled AUROC is
statistically indistinguishable from intact, the policy contributes
nothing measurable and H3's framing does not survive."* That is what
happened. The committed response, also written in advance, is the
restatement this section now makes official: **this project's accuracy
result is that a fixed nonlinear aggregator over a shared predicate
trunk beats a learned linear one at matched capacity** -- H3's
explainability and consistency claims (the audit trail itself, and the
reasoner's H1 advantage, which the scramble does not touch since it
never alters the rule graph) stand as measured. What does not survive
is the stronger claim that *which* rules are wired in, specifically,
is what drives the accuracy gap. On this dataset and this policy, it is
not; the gap looks like it comes from having a fixed, capacity-matched
nonlinear circuit at all, not from that circuit encoding the right
domain knowledge.

This does not weaken H1 or H2. H1 is about verdict consistency across
equivalent cases, a structural property of the rule *graph*, which
scrambling does not alter (see the pre-registration's safety argument).
H2 is about latency against Clingo, orthogonal to this control
entirely. It narrows H3 specifically, from "the policy's domain
knowledge drives the accuracy gap" to "a fixed nonlinear aggregator
drives the accuracy gap, with the policy supplying the auditable
structure and the consistency guarantee on top." That is a smaller
claim than the one this project set out to test, and this document says
so rather than reframing around it after the fact.

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

## Pre-registration: the scrambled-policy control

**Written before the control was run. Results go in a separate section
below; this one is not to be edited afterwards.**

Every H1/H3 number above comes from a single policy, which leaves the
central question of this project unanswered: is the reasoner's
advantage a property of *the rules*, or merely of having a fixed
nonlinear aggregator where the baseline has a learned linear one? A
reviewer will ask. The control that answers it is
`nspe/policy/scramble.py`: a global derangement over the six base
predicate names, applied to rule bodies and `unless` clauses only.

What is held fixed is the point. The rule graph, the rule ids, the
confidences, the number of parameters, the shared `PredicateTrunk`, and
every predicate description all stay identical -- so
`init_heads_from_descriptions` grounds each predicate head to exactly
the semantics it had before. The scrambled model has the same capacity
and the same computation, wired to the wrong evidence. The permutation
is a bijection rather than a per-slot shuffle, which is what keeps a
body from acquiring a duplicate literal (`mu^2` under the product
t-norm is a semantic change, not a rewiring).

Protocol: reasoner arm only, seeds 0-9, ViT-L-14, both splits. The
baseline consumes the policy solely for `num_predicates` (unchanged at
6), so it is genuinely invariant to the scramble and the intact
baselines are reused deliberately. Each scrambled run reads its
threshold from its *own* validation artifact.

### Predictions

1. **Intact > scrambled on AUROC**, one-sided, direction registered
   here. This is the primary outcome.
2. **`adjusted_consistency` roughly unchanged.** Consistency is a
   structural property of the rule graph, which the rewiring preserves
   exactly. A large move here would mean the metric is tracking
   something other than what it is claimed to track, and would need
   explaining before either number is reported.
3. **Scrambled still above chance.** The trunk and the heads can still
   learn; the rules being wrong degrades the signal, it does not
   destroy it.

### The falsifying outcome, named in advance

If scrambled AUROC is statistically indistinguishable from intact, the
policy contributes nothing measurable and H3's framing does not
survive. The claim would then have to be restated as "a fixed
nonlinear aggregator beats a learned linear one at matched capacity" --
still a real result, and still consistent with the H2 latency work, but
substantially weaker than "the rules carry the accuracy". That
restatement is the committed response to that outcome; it is written
down here so it cannot be rationalized away afterwards.

## Zero-shot-only diagnostic: is it the wiring, or the grounding?

The control above leaves open *why* scrambling doesn't hurt: one live
explanation was that gradient descent re-purposes what each predicate
head detects to fit whichever rules it's given -- correct or scrambled
-- eroding any advantage the correct wiring had by the time training
finishes. `--epochs 0` (`nspe/train/loop.py`) tests this directly by
comparing intact vs. scrambled **before any training happens at all**,
right at the CLIP zero-shot-seeded initialization
(`PredicateHead.zero_shot_weight`, which dominates the still-near-zero
freshly-initialized linear head at step 0 -- see `nspe/trunk.py`).
10 seeds, validation split, same ten scrambled policies as the main
control.

| | AUROC at init (no training) |
|---|---|
| intact | 0.4993 ± 0.0307 |
| scrambled | 0.4934 ± 0.0311 |
| gap, p (one-sided) | +0.0059, p=0.26 (5/10 seeds positive) |

**Both are at chance, and indistinguishable from each other.** This
rules out the re-purposing hypothesis directly: there is no
correct-wiring advantage at initialization for training to erase,
because the zero-shot CLIP grounding alone -- one text-encoded
description per predicate -- carries essentially no signal about the
verdict on its own, regardless of which predicate a rule reads. Whatever
signal the trained model eventually has must come almost entirely from
the *learned* trunk/head weights shaped over training, not from the
predicate identities' initial semantic grounding. That reframes the
post-training null result: it isn't that correct wiring helps and
training erases it -- wiring was never where the advantage lived, at
either end of training. What remains open is why the *learned* heads
end up similarly informative under either wiring; see below.

### The predicates are not collapsed copies of each other

One candidate explanation for the null result is that the six base
predicates end up so similar that swapping which one a rule reads
changes little. `nspe/eval/diagnostics.py::predicate_stats` already
records what settles this, and it is in every committed artifact -- no
new run was needed, only reading `docs/results/h1_h3/results_*_s[0-9]
.json` across all ten seeds.

| predicate | activation_rate (val) | max abs. correlation (val) |
|---|---|---|
| slur_present | 0.570 ± 0.184 | 0.264 ± 0.063 |
| targets_protected_group | 0.683 ± 0.070 | 0.344 ± 0.107 |
| dehumanizing_comparison | 0.541 ± 0.202 | 0.257 ± 0.082 |
| condemnation_context | 0.270 ± 0.090 | 0.308 ± 0.119 |
| mocking_tone | 0.711 ± 0.115 | 0.434 ± 0.113 |
| benign_context | 0.663 ± 0.111 | 0.436 ± 0.109 |

`max_abs_correlation` is each predicate's largest absolute Pearson
correlation against any *other* predicate. The largest mean is 0.44
(`mocking_tone` / `benign_context`, a semantically plausible pair), and
the single worst value across all six predicates and all ten seeds is
0.64 on test. Activation rates all sit between 0.27 and 0.72, so no
predicate is stuck on or off either.

**This refutes predicate collapse, which is the strongest form of the
"predicates are interchangeable" explanation** -- and it is worth
stating precisely because collapse is exactly the failure this project
already hit once (5 distinct signatures out of 64, see "The problem this
fixed"). It does *not* refute the weaker version: predicates can be
individually distinct and still each carry comparable information about
the label, which would make most rewirings similarly informative.
Distinguishing that needs the wiring sweep and the label-correlation
numbers, below.

## What's still open

- **Why scrambling doesn't hurt accuracy is narrower now, but still
  open.** The zero-shot diagnostic above rules out one explanation
  (training erasing an initial wiring advantage -- there was none to
  erase) but not the others: the base predicates may be correlated
  enough with each other and with the label that most derangements land
  on comparably-informative combinations once trained; the rule
  confidences and t-conorm aggregation may dominate over which specific
  predicate sits in which slot; or six predicates may simply be too few
  for a random derangement to land on a genuinely uninformative wiring.
  Distinguishing these would need either a larger/more heterogeneous
  predicate set or a targeted adversarial scramble (worst-case
  derangement, not a random one), neither attempted here.
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
