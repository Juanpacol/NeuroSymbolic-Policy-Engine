# Running the H2 benchmark on Google Colab (CUDA)

This machine has no CUDA (Apple Silicon / MPS only), so the H2
(performance) numbers that go in the paper need to come from a real
NVIDIA GPU. Colab's free tier is enough for this.

## Steps

1. Open https://colab.research.google.com, create a new notebook.
2. **Runtime > Change runtime type > Hardware accelerator > T4 GPU**,
   then save.
3. Paste each block below into its own cell, in order.

### Cell 1 — clone and install

```python
!git clone https://github.com/Juanpacol/NeuroSymbolic-Policy-Engine.git
%cd NeuroSymbolic-Policy-Engine
!pip install -e ".[bench]" -q
```

### Cell 2 — confirm CUDA is visible

```python
import torch
print(torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

### Cell 3 — run the test suite (sanity check before benchmarking)

```python
!python -m pytest test/ -q
```

Everything should pass, including `test_clingo_agreement.py` — that is
the gate that makes the timing numbers below meaningful at all.

Each sweep times three arms per batch size, and which one you quote
matters — see `docs/h2_findings.md` for the reasoning:

- `reasoner_crisp` — the crisp t-norm over the *same* fact sets Clingo
  receives. The only arm whose verdicts are certified identical to
  Clingo's (that is what `test_clingo_agreement.py` proves), so it is
  the arm a speedup claim should lead with.
- `reasoner_product` — the deployed configuration: graded truth degrees
  under the product t-norm.
- `clingo` — the baseline, timed via `infer_verdicts` so it is not
  charged for stringifying every atom in its model.

### Cell 4 — sweep on the bundled example policy (CUDA)

```python
!python -m nspe.bench.cli \
  --device cuda \
  --batch-sizes 1 8 64 256 1024 8192 \
  --warmup 20 --reps 200 --clingo-budget-s 30 \
  --out bench_results/h2_cuda_meta.json
```

### Cell 4b — the same sweep on CPU

Not optional. The headline speedup conflates three separate advantages
— vectorization, GPU hardware, and batching — and this is what
separates them. At `batch=1` on CPU, hardware and batching are held
fixed, so the comparison is purely "vectorized fuzzy evaluation vs.
stable-model search"; the CPU→CUDA delta at large batch is what the GPU
buys. Expect Clingo to *win* at batch 1: that is the honest result, and
reporting it is what makes the batched numbers credible.

```python
!python -m nspe.bench.cli \
  --device cpu \
  --batch-sizes 1 8 64 256 1024 \
  --warmup 20 --reps 200 --clingo-budget-s 30 \
  --out bench_results/h2_cpu_meta.json
```

### Cell 5 — scaling with the rule base

One cell each, so a disconnect costs at most one sweep. Capped at batch
1024: at 8192 on `b100_r1000` a single Clingo rep is 8192 solves over a
1000-rule ground program.

```python
!python -m nspe.bench.cli --device cuda --synthetic 10 20 \
  --batch-sizes 1 64 1024 --warmup 20 --reps 200 --clingo-budget-s 30 \
  --out bench_results/h2_cuda_synthetic_b10_r20.json
```

```python
!python -m nspe.bench.cli --device cuda --synthetic 50 200 \
  --batch-sizes 1 64 1024 --warmup 20 --reps 200 --clingo-budget-s 30 \
  --out bench_results/h2_cuda_synthetic_b50_r200.json
```

```python
!python -m nspe.bench.cli --device cuda --synthetic 100 1000 \
  --batch-sizes 1 64 1024 --warmup 20 --reps 200 --clingo-budget-s 30 \
  --out bench_results/h2_cuda_synthetic_b100_r1000.json
```

### Cell 6 — locating the CPU/GPU crossover

`docs/h2_findings.md` brackets the crossover between batch 64 and 1024
but does not locate it, which matters because a real deployment's batch
size may fall in that range. This narrows it to a factor of two. Both
devices, same policy, same rep budget — the pair is the comparison.

```python
!python -m nspe.bench.cli --device cuda \
  --batch-sizes 64 128 256 512 --warmup 20 --reps 200 --clingo-budget-s 30 \
  --out bench_results/h2_cuda_meta_crossover.json
```

```python
!python -m nspe.bench.cli --device cpu \
  --batch-sizes 64 128 256 512 --warmup 20 --reps 200 --clingo-budget-s 30 \
  --out bench_results/h2_cpu_meta_crossover.json
```

### Cell 7 — locating the rule-base-scaling inflection

The advantage is flat at `b50_r200` and no longer flat at `b100_r1000`,
so the bend sits between them. One policy in between locates it, and
filling in the batch sizes `b100_r1000` never ran (it did only 1, 64,
1024) shows whether the bend is smooth or a knee.

```python
!python -m nspe.bench.cli --device cuda --synthetic 70 500 \
  --batch-sizes 1 64 1024 --warmup 20 --reps 200 --clingo-budget-s 30 \
  --out bench_results/h2_cuda_synthetic_b70_r500.json
```

```python
!python -m nspe.bench.cli --device cuda --synthetic 100 1000 \
  --batch-sizes 8 128 256 512 --warmup 20 --reps 200 --clingo-budget-s 30 \
  --out bench_results/h2_cuda_synthetic_b100_r1000_fill.json
```

### Cell 8 — does the crossover move with policy size?

Only CUDA was ever run on the synthetic policies, so whether a larger
rule base shifts the CPU/GPU crossover is untested. `b100_r1000`'s
reasoner latency is no longer launch-overhead-bound at large batch, so
its crossover may well sit elsewhere than the meta policy's.

```python
!python -m nspe.bench.cli --device cpu --synthetic 100 1000 \
  --batch-sizes 1 64 256 1024 --warmup 20 --reps 200 --clingo-budget-s 30 \
  --out bench_results/h2_cpu_synthetic_b100_r1000.json
```

### Cell 9 — download the results

```python
from google.colab import files
import shutil
shutil.make_archive("bench_results", "zip", "bench_results")
files.download("bench_results.zip")
```

## Notes

- **`bench_results/` is gitignored; `docs/results/` is not.** Raw sweep
  output is environment-specific scratch and is regenerated per
  machine. The curated subset that backs a table in the paper is copied
  into `docs/results/` and committed deliberately, so every number in
  the paper is checkable. See `docs/results/README.md`.
- `--clingo-budget-s` sizes Clingo's rep count by wall clock rather
  than a fixed integer, because one Clingo rep is `batch_size`
  sequential solves. Raise it to 60 for the final run if the printed
  `clingo.reps` at the largest batch drops near its floor of 5.
- Clingo's `p95_ms`/`p99_ms` are per-batch aggregates over many solves,
  not per-case latencies, so they are **not** comparable to the
  reasoner's. Compare `per_item_median_ms` across arms.
- If Colab disconnects mid-sweep, just re-run from Cell 1 — nothing here
  depends on prior state.
- Free-tier Colab GPUs are shared and can vary run-to-run; if the
  numbers look noisy, increase `--reps` or re-run and keep the more
  stable of two runs, and note the GPU model from Cell 2 in the paper.
