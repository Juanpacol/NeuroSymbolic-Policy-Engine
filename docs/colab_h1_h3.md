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
for seed in range(5):
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

The first `--split test` triggers a one-time CLIP encode of the test
split, shared by all five seeds. **Check the printed `num_examples`
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

### Cell 10 — download checkpoints and results

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
  from result JSONs, grouping by split and backbone so validation and
  test never average together.
