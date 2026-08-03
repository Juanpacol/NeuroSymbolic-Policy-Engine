# Running H1/H3 training and evaluation on Colab/Kaggle (CUDA)

Frozen-CLIP forward passes over the ~9.6k available Hateful Memes
examples are slow on this dev machine's CPU/MPS, and training needs
several passes over the data. A free-tier GPU notebook (Colab or
Kaggle; the cells below work on either -- adjust the clone path if
needed) makes that a reasonable iteration time, the same way it did for
the H2 benchmark (see `docs/colab_benchmark.md`).

See `docs/h1_h3_findings.md` for what these numbers mean and why the
CLI has the shape it does; this file is only the runbook.

## Steps

1. Open a GPU notebook (Colab: **Runtime > Change runtime type > T4
   GPU**; Kaggle: **Settings > Accelerator > GPU T4 x2**, and
   **Settings > Internet > On**).
2. Paste each block below into its own cell, in order.
3. If using a HF token, set it via the platform's secret manager (Colab:
   `userdata`; Kaggle: `kaggle_secrets.UserSecretsClient`) -- never
   hardcode it in a cell, it persists in the saved notebook.

### Cell 1 — clone or update, and install

```python
![ -d nspe-repo ] \
  && git -C nspe-repo pull \
  || git clone https://github.com/Juanpacol/NeuroSymbolic-Policy-Engine.git nspe-repo
%cd nspe-repo
!git log --oneline -1  # confirm this matches the commit you expect
!pip install -e ".[train]" -q
!pip install expecttest clingo -q
```

### Cell 2 — confirm CUDA is visible

```python
import sys, torch
print(sys.executable)
print(torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

### Cell 3 — run the test suite (sanity check before training)

```python
!{sys.executable} -m pytest test/ -q
```

Everything should pass (the `train` extra covers CLIP + the Hateful
Memes dataset deps). Expect ~168 passed under Python 3.10+; a couple of
gradcheck/explain tests self-skip under older interpreters.

### Cell 4 — train the reasoner (one seed)

```python
!{sys.executable} -m nspe.train.cli --model reasoner \
  --clip-model ViT-L-14 --clip-pretrained openai \
  --cache-dir /kaggle/working/emb_cache \
  --seed 0 --epochs 30 \
  --out /kaggle/working/checkpoints/reasoner_s0.pt \
  --metrics-out /kaggle/working/checkpoints/reasoner_s0_metrics.json
```

The first run against a given (split, backbone) pair encodes and caches
embeddings to `--cache-dir`; every run after that trains off the cache
instead of re-hitting CLIP, which is most of the wall-clock cost. See
`nspe/train/cli.py --help` for every flag -- notably `--select-metric`
(default `auroc`, not `bce`), `--lambda-anchor/--lambda-decorr/
--lambda-entropy` (predicate anti-collapse regularizers), and
`--learnable-confidence` (Phase 4 ablation, off by default).

### Cell 5 — train the baseline (same seed, same backbone)

```python
!{sys.executable} -m nspe.train.cli --model baseline \
  --clip-model ViT-L-14 --clip-pretrained openai \
  --cache-dir /kaggle/working/emb_cache \
  --seed 0 --epochs 30 \
  --out /kaggle/working/checkpoints/baseline_s0.pt \
  --metrics-out /kaggle/working/checkpoints/baseline_s0_metrics.json
```

### Cell 6 — evaluate H1 (consistency) and H3 (explainability)

```python
!{sys.executable} -m nspe.eval.cli \
  --clip-model ViT-L-14 --clip-pretrained openai \
  --cache-dir /kaggle/working/emb_cache \
  --reasoner-checkpoint /kaggle/working/checkpoints/reasoner_s0.pt \
  --baseline-checkpoint /kaggle/working/checkpoints/baseline_s0.pt \
  --split validation \
  --out /kaggle/working/results_s0.json
```

Reports AUROC (the headline H3 number), per-model fitted thresholds,
chance-corrected consistency (`adjusted_consistency`, not the raw rate
-- see findings doc), and per-predicate activation diagnostics.

### Cell 7 — repeat across seeds and aggregate

A single seed is not a reportable number by itself -- see
`docs/h1_h3_findings.md` for why. Repeat cells 4-6 for `seed in [1, 2]`
(or loop them), then:

```python
import json, numpy as np

runs = [json.load(open(f"/kaggle/working/results_s{s}.json")) for s in (0, 1, 2)]
for model in ("reasoner", "baseline"):
    auroc = [r["h3_explainability"][model]["auroc"] for r in runs]
    acc = [r["h3_explainability"][model]["accuracy"] for r in runs]
    print(f"{model}: AUROC {np.mean(auroc):.4f} ± {np.std(auroc):.4f} | "
          f"accuracy {np.mean(acc):.4f} ± {np.std(acc):.4f}")
```

### Cell 8 — ablation sweep (Phase 4)

Once the headline numbers are in, one command runs the whole ablation
table: 6 configurations x 3 seeds, reusing one baseline per seed.

```python
!{sys.executable} -m nspe.ablate.cli \
  --clip-model ViT-L-14 --clip-pretrained openai \
  --cache-dir /kaggle/working/emb_cache \
  --device cuda \
  --ckpt-dir /kaggle/working/checkpoints/ablations \
  --out /kaggle/working/ablations.json
```

Rerunning the same command skips whatever already landed in
`ablations.json`, so a dropped session costs at most one run. Narrow to
a single configuration with `--configs pmean`, and see
`--help` for the full list.

### Cell 9 — held-out test evaluation

**Prerequisite: this must run in the same session as the training.**
Checkpoints are gitignored, so a test run always evaluates weights
retrained in that session — and the operating point a calibrator
settles on is not stable across retrains even at a fixed seed. The
thresholds therefore have to be re-fitted on validation *here*, against
these exact weights; reusing the ones recorded in
`docs/results/h1_h3/results_s*.json` would apply an operating point
belonging to weights that no longer exist. `--thresholds-from` enforces
the pairing, and the eval CLI refuses `--split test` without it.

```python
for seed in range(10):
    !{sys.executable} -m nspe.train.cli --model reasoner \
      --clip-model ViT-L-14 --clip-pretrained openai \
      --cache-dir /kaggle/working/emb_cache --seed {seed} --epochs 30 \
      --out /kaggle/working/checkpoints/reasoner_s{seed}.pt
    !{sys.executable} -m nspe.train.cli --model baseline \
      --clip-model ViT-L-14 --clip-pretrained openai \
      --cache-dir /kaggle/working/emb_cache --seed {seed} --epochs 30 \
      --out /kaggle/working/checkpoints/baseline_s{seed}.pt

    # Validation first: this is what fits each arm's operating point.
    !{sys.executable} -m nspe.eval.cli \
      --clip-model ViT-L-14 --cache-dir /kaggle/working/emb_cache \
      --reasoner-checkpoint /kaggle/working/checkpoints/reasoner_s{seed}.pt \
      --baseline-checkpoint /kaggle/working/checkpoints/baseline_s{seed}.pt \
      --split validation --device cuda \
      --out /kaggle/working/results_val_s{seed}.json

    # Test once, at the point validation just fitted.
    !{sys.executable} -m nspe.eval.cli \
      --clip-model ViT-L-14 --cache-dir /kaggle/working/emb_cache \
      --reasoner-checkpoint /kaggle/working/checkpoints/reasoner_s{seed}.pt \
      --baseline-checkpoint /kaggle/working/checkpoints/baseline_s{seed}.pt \
      --split test --device cuda \
      --thresholds-from /kaggle/working/results_val_s{seed}.json \
      --out /kaggle/working/results_test_s{seed}.json
```

Ten seeds, not five: the exact sign-permutation test used for
significance has a floor of p=0.0625 two-sided at n=5, so no result at
five seeds can reach p<0.05 however large the effect. At n=10 the floor
drops to 0.002. Cell 10 depends on the baselines this loop leaves
behind, so run it over the full `range(10)`.

The first `--split test` triggers a one-time CLIP encode of the test
split, shared by every seed. **Check the printed `num_examples`
before trusting anything**: expect roughly 2400 of 3000 rows after the
image-availability filter (validation goes 1040 → 831, about 80%). A
number near 3000 means the filter changed; a much smaller one means
something truncated, which matters here because this dataset's rows are
ordered by label.

Then aggregate:

```python
!{sys.executable} -m nspe.eval.aggregate '/kaggle/working/results_test_s*.json'
!{sys.executable} -m nspe.eval.aggregate '/kaggle/working/results_val_s*.json'
```

Download **both** sets. The paired validation files are what show this
rerun reproduces the published validation numbers within seed noise,
which is the only evidence that the test table and the original
validation table describe the same system.

### Cell 10 — the scrambled-policy control

**Prerequisite: cell 9 must have run in this session, with
`range(10)`.** Ten seeds rather than five is not a preference: the
exact sign-permutation test's floor is p=0.0625 two-sided at n=5, so
p<0.05 is *mathematically unreachable* there. At n=10 the floor is
0.002.

This control asks whether the reasoner's advantage comes from the rules
or merely from having a fixed nonlinear aggregator. See the
pre-registered prediction in `docs/h1_h3_findings.md` — read it before
looking at the output.

The reasoner arm is retrained per scrambled policy; the baseline is
**not**, and that is deliberate. It consumes the policy solely for
`num_predicates`, which the scramble leaves at 6, so it is genuinely
invariant and the intact `baseline_s{seed}.pt` is the correct
comparison. That halves the cost.

```python
POL = "/kaggle/working/nspe-repo/docs/results/h1_h3/policies_scrambled"

for seed in range(10):
    policy = f"{POL}/hateful_memes_scrambled_s{seed}.yaml"

    # --policy must reach training and BOTH evals. Omitting it anywhere
    # silently evaluates a model under a wiring it was not trained on.
    !{sys.executable} -m nspe.train.cli --model reasoner --policy {policy} \
      --clip-model ViT-L-14 --clip-pretrained openai \
      --cache-dir /kaggle/working/emb_cache --seed {seed} --epochs 30 \
      --out /kaggle/working/checkpoints/reasoner_scram_s{seed}.pt

    !{sys.executable} -m nspe.eval.cli --policy {policy} \
      --clip-model ViT-L-14 --cache-dir /kaggle/working/emb_cache \
      --reasoner-checkpoint /kaggle/working/checkpoints/reasoner_scram_s{seed}.pt \
      --baseline-checkpoint /kaggle/working/checkpoints/baseline_s{seed}.pt \
      --split validation --device cuda \
      --out /kaggle/working/results_scram_val_s{seed}.json

    # --thresholds-from must point at this run's OWN validation file.
    !{sys.executable} -m nspe.eval.cli --policy {policy} \
      --clip-model ViT-L-14 --cache-dir /kaggle/working/emb_cache \
      --reasoner-checkpoint /kaggle/working/checkpoints/reasoner_scram_s{seed}.pt \
      --baseline-checkpoint /kaggle/working/checkpoints/baseline_s{seed}.pt \
      --split test --device cuda \
      --thresholds-from /kaggle/working/results_scram_val_s{seed}.json \
      --out /kaggle/working/results_scram_test_s{seed}.json
```

The embedding cache is keyed by `(split, model_name, pretrained)` and
is independent of the policy, so nothing is re-encoded here.

`resolve_thresholds` only **warns** on a policy-fingerprint mismatch
rather than failing — watch the output for it. A warning here means a
`--policy` flag was dropped somewhere above, and the run is worthless.

Then check the grouping:

```python
!{sys.executable} -m nspe.eval.aggregate '/kaggle/working/results_*s*.json'
```

This must print **four** groups: validation and test, each intact and
scrambled. Fewer means a control run is being averaged into the main
result, which corrupts the headline number rather than merely losing
the control — stop and check `policy_name` in the artifacts before
reading anything else.

### Cell 11 — zero-shot-only control (why doesn't scrambling hurt?)

The scrambled-policy control (Cell 10) came back negative: after full
training, intact and scrambled AUROC were statistically
indistinguishable (`docs/h1_h3_findings.md`, "Scrambled-policy control:
results"). One live explanation is that gradient descent *re-purposes*
what each predicate head detects to fit whichever rules it's given --
correct or scrambled -- eroding the wiring's own effect over training.
This cell tests that directly, by comparing intact vs. scrambled
**before any training happens at all**, i.e. right at the CLIP
zero-shot-seeded initialization (`--epochs 0`, added for exactly this).

This needs no held-out test-set discipline -- it is a mechanism check,
not a headline claim -- so it only touches validation, and reuses one
zero-shot baseline checkpoint per seed across intact and every
scrambled policy (the baseline consumes the policy only for
`num_predicates`, same as in Cell 10).

```python
POL = "/kaggle/working/nspe-repo/docs/results/h1_h3/policies_scrambled"

for seed in range(10):
    !{sys.executable} -m nspe.train.cli --model reasoner --epochs 0 \
      --clip-model ViT-L-14 --clip-pretrained openai \
      --cache-dir /kaggle/working/emb_cache --seed {seed} \
      --out /kaggle/working/checkpoints/reasoner_zs_s{seed}.pt

    !{sys.executable} -m nspe.train.cli --model baseline --epochs 0 \
      --clip-model ViT-L-14 --clip-pretrained openai \
      --cache-dir /kaggle/working/emb_cache --seed {seed} \
      --out /kaggle/working/checkpoints/baseline_zs_s{seed}.pt

    !{sys.executable} -m nspe.eval.cli \
      --clip-model ViT-L-14 --cache-dir /kaggle/working/emb_cache \
      --reasoner-checkpoint /kaggle/working/checkpoints/reasoner_zs_s{seed}.pt \
      --baseline-checkpoint /kaggle/working/checkpoints/baseline_zs_s{seed}.pt \
      --split validation --device cuda \
      --out /kaggle/working/results_zs_val_s{seed}.json

    policy = f"{POL}/hateful_memes_scrambled_s{seed}.yaml"
    !{sys.executable} -m nspe.train.cli --model reasoner --policy {policy} --epochs 0 \
      --clip-model ViT-L-14 --clip-pretrained openai \
      --cache-dir /kaggle/working/emb_cache --seed {seed} \
      --out /kaggle/working/checkpoints/reasoner_zs_scram_s{seed}.pt

    !{sys.executable} -m nspe.eval.cli --policy {policy} \
      --clip-model ViT-L-14 --cache-dir /kaggle/working/emb_cache \
      --reasoner-checkpoint /kaggle/working/checkpoints/reasoner_zs_scram_s{seed}.pt \
      --baseline-checkpoint /kaggle/working/checkpoints/baseline_zs_s{seed}.pt \
      --split validation --device cuda \
      --out /kaggle/working/results_zs_scram_val_s{seed}.json
```

No training loop runs -- each iteration is one CLIP text encode (the
zero-shot residual) plus one validation pass, so all 10 seeds should
finish in a couple of minutes even on a modest GPU. Compare
`h3_explainability.reasoner.auroc` between
`results_zs_val_s{seed}.json` and `results_zs_scram_val_s{seed}.json`
per seed:

- **If intact clearly beats scrambled here** (unlike after full
  training), that supports the re-purposing hypothesis: the correct
  wiring has a real advantage at init that training away erases.
- **If intact and scrambled are already indistinguishable at
  init**, the flat result after training isn't about re-purposing at
  all -- it points instead to the base predicates being too correlated
  with each other and the label, or too few (six), for a random
  derangement to land on a genuinely uninformative wiring.

### Cell 12 — download checkpoints and results

```python
import shutil
shutil.make_archive("/kaggle/working/checkpoints", "zip", "/kaggle/working/checkpoints")
shutil.make_archive("/kaggle/working/results", "zip", "/kaggle/working", base_dir=".")
```
Then download via the notebook's Output tab (Kaggle) or `files.download(...)` (Colab).

## Notes

- Checkpoints exclude the frozen CLIP backbone (it is rebuilt from the
  pretrained tag on load), so they are ~1MB rather than ~1.7GB. That is
  what makes a 20-run session fit in the disk quota.
- `--cache-dir` embeddings are keyed by `(split, model_name,
  pretrained)` and refuse to load under a mismatched backbone -- delete
  the cache directory if you change `--clip-model`.
- If the session disconnects mid-run, `--resume <checkpoint>` reloads
  weights (not optimizer state) and continues; the embedding cache
  already on disk is reused as-is.
- H3's "not significant accuracy loss" threshold is intentionally not
  fixed -- `results_s*.json` reports the raw `auroc_gap`/`accuracy_gap`/
  `f1_gap` (reasoner minus baseline) for the paper's methods section to
  interpret, not a pass/fail verdict.
- Only evaluate `--split test` once, and pass the operating point via
  `--thresholds-from <validation results.json>` produced by the same
  checkpoints. The CLI refuses to fit a threshold on test, and
  `compute_h3` refuses a single-class label set outright — this
  dataset's rows are ordered by label, so a truncated split would
  otherwise report AUROC 0.5 and an `auroc_gap` of exactly 0.0, which
  reads as a real null result. See `docs/h1_h3_findings.md`.
- `python -m nspe.eval.aggregate '<glob>'` derives the mean±std tables
  from result JSONs, grouping by split, backbone **and policy** so
  validation never averages with test, ViT-L-14 never with ViT-B-32,
  and a scrambled control run never with the intact result it exists to
  be compared against.
- Be precise with that glob. Artifacts from separate sessions of the
  *same* configuration (`results_s*.json` and `results_val_s*.json`,
  say) pool into one group with seeds counted twice, which quietly
  inflates `n` and therefore any p-value computed from it.
