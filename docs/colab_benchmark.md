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

### Cell 4 — run the benchmark sweep on the bundled example policy

```python
!python -m nspe.bench.cli \
  --device cuda \
  --batch-sizes 1 8 64 256 1024 8192 \
  --warmup 20 \
  --reps 200 \
  --out bench_results/cuda_meta_policy.json
```

### Cell 5 — run the same sweep on synthetic policies at increasing scale

This characterizes the dense/sparse kernel crossover and where Clingo's
grounding starts to fall behind as the rule base grows — useful for the
paper's scaling figure. There is no CLI flag for synthetic policies
yet, so run it directly:

```python
from nspe.data.synthetic import make_layered_policy
from nspe.bench.cli import run_sweep, _print_markdown
import json

for num_base, num_rules in [(10, 20), (50, 200), (100, 1000)]:
    policy = make_layered_policy(num_base, num_rules, seed=0)
    rows = run_sweep(policy, "cuda", (1, 64, 1024), warmup=20, reps=200)
    print(f"\n=== base={num_base} rules={num_rules} ===")
    _print_markdown(rows)
    with open(f"bench_results/cuda_synthetic_b{num_base}_r{num_rules}.json", "w") as f:
        json.dump(rows, f, indent=2)
```

### Cell 6 — download the results

```python
from google.colab import files
import shutil
shutil.make_archive("bench_results", "zip", "bench_results")
files.download("bench_results.zip")
```

Send the `bench_results.zip` back and it gets folded into the paper's
benchmark tables/figures.

## Notes

- `bench_results/` is gitignored on purpose (raw benchmark output is
  environment-specific and shouldn't be committed as source); it's
  meant to be generated fresh per machine and attached to the paper
  separately.
- If Colab disconnects mid-sweep, just re-run from Cell 1 — nothing here
  depends on prior state.
- Free-tier Colab GPUs are shared and can vary run-to-run; if the
  numbers look noisy, increase `--reps` or re-run and keep the more
  stable of two runs, and note the GPU model from Cell 2 in the paper.
