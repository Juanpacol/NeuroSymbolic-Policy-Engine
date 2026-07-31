# Running H1/H3 training and evaluation on Google Colab (CUDA)

Frozen-CLIP forward passes over the ~9.6k available Hateful Memes
examples are slow on this dev machine's CPU/MPS, and training needs
several passes over the data. Colab's free-tier GPU makes that a
reasonable iteration time, the same way it did for the H2 benchmark
(see `docs/colab_benchmark.md`).

## Steps

1. Open https://colab.research.google.com, create a new notebook.
2. **Runtime > Change runtime type > Hardware accelerator > T4 GPU**,
   then save.
3. Paste each block below into its own cell, in order.

### Cell 1 — clone and install

```python
!git clone https://github.com/Juanpacol/NeuroSymbolic-Policy-Engine.git
%cd NeuroSymbolic-Policy-Engine
!pip install -e ".[train]" -q
!pip install expecttest -q
```

### Cell 2 — confirm CUDA is visible

```python
import torch
print(torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

### Cell 3 — run the test suite (sanity check before training)

```python
!python -m pytest test/ -q
```

Everything should pass (the `train` extra covers CLIP + the Hateful
Memes dataset deps, so nothing here should self-skip for missing
dependencies).

### Cell 4 — smoke-test training (1 epoch, both models)

Confirms the CLI, the lazy per-item image download, and the loss curve
all work before committing to a full run.

```python
!python -m nspe.train.cli --model reasoner \
  --policy nspe/policies/hateful_memes.yaml \
  --epochs 1 --device cuda --out checkpoints/reasoner_smoke.pt

!python -m nspe.train.cli --model baseline \
  --epochs 1 --device cuda --out checkpoints/baseline_smoke.pt
```

### Cell 5 — train the reasoner

```python
!python -m nspe.train.cli --model reasoner \
  --policy nspe/policies/hateful_memes.yaml \
  --epochs 10 --lr 1e-3 --batch-size 32 --device cuda \
  --out checkpoints/reasoner.pt
```

### Cell 6 — train the baseline

```python
!python -m nspe.train.cli --model baseline \
  --epochs 10 --lr 1e-3 --batch-size 32 --device cuda \
  --out checkpoints/baseline.pt
```

### Cell 7 — evaluate H1 (consistency) and H3 (explainability)

```python
!python -m nspe.eval.cli \
  --policy nspe/policies/hateful_memes.yaml \
  --reasoner-checkpoint checkpoints/reasoner.pt \
  --baseline-checkpoint checkpoints/baseline.pt \
  --split validation --device cuda \
  --out eval_results/h1_h3.json
```

### Cell 8 — download checkpoints and results

```python
from google.colab import files
import shutil
shutil.make_archive("checkpoints", "zip", "checkpoints")
shutil.make_archive("eval_results", "zip", "eval_results")
files.download("checkpoints.zip")
files.download("eval_results.zip")
```

Send both zips back and they get folded into the paper's H1/H3
tables/figures, the same way `bench_results.zip` was for H2.

## Notes

- Images are downloaded lazily, one per item, the first time each is
  touched -- the first epoch of Cell 4/5/6 is network-bound, not just
  compute-bound. If unsure whether things are wired correctly, always
  run the Cell 4 smoke test (`--epochs 1`) first.
- `checkpoints/` and `eval_results/` are gitignored on purpose, same
  reasoning as `bench_results/` in `docs/colab_benchmark.md`.
- H3's "not significant accuracy loss" threshold is not fixed yet --
  `eval_results/h1_h3.json` reports the raw `accuracy_gap`/`f1_gap`
  (reasoner minus baseline) for the paper's methods section to
  interpret, not a pass/fail verdict.
- If Colab disconnects mid-run, re-run from Cell 1; nothing depends on
  prior in-memory state, only on files already written to
  `checkpoints/`.
