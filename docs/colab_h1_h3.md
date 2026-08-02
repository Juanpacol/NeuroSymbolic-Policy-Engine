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

### Cell 9 — download checkpoints and results

```python
import shutil
shutil.make_archive("/kaggle/working/checkpoints", "zip", "/kaggle/working/checkpoints")
shutil.make_archive("/kaggle/working/results", "zip", "/kaggle/working", base_dir=".")
```
Then download via the notebook's Output tab (Kaggle) or `files.download(...)` (Colab).

## Notes

- Checkpoints now include the full frozen-CLIP state dict, so a
  ViT-L-14 checkpoint is large (~1.7GB). Budget disk/download time
  accordingly.
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
- Only evaluate `--split test` once, with a threshold fitted on
  `validation` passed explicitly via `--threshold`. See findings doc.
