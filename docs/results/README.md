# Committed result artifacts

Every number quoted in the paper or in `docs/*_findings.md` should be
traceable to a file here, and every file here should name the command
that produced it and the machine it ran on.

This directory is deliberately **not** `bench_results/` or
`eval_results/`, which stay gitignored: those hold raw, per-machine
sweep output that is regenerated freely. What lands here is the curated
subset that backs a claim, copied in and `git add`ed on purpose. The
curation is the point — do not automate it.

Each JSON already carries its own provenance: `environment` (torch and
clingo versions, platform, device, git commit), `policy_fingerprint`
(content hash of the compiled rule set), and for evaluation runs a
`reasoner_config` block recording the settings that do not live in the
checkpoint.

## H2 — latency (`h2_*.json`)

Schema version 2. Three timed arms per batch size: `reasoner_crisp`
(certified identical to Clingo, the arm speedup claims should lead
with), `reasoner_product` (the deployed configuration), and `clingo`.

| file | command | machine |
|---|---|---|
| `h2_cpu_m5_meta.json` | `python -m nspe.bench.cli --device cpu --batch-sizes 1 8 64 256 --warmup 20 --reps 100 --clingo-budget-s 10` | Apple M5, macOS 26.4.1, torch 2.13.0, clingo 5.8.0 (preliminary local check) |
| `h2_cuda_meta.json` | `python -m nspe.bench.cli --device cuda --batch-sizes 1 8 64 256 1024 8192 --warmup 20 --reps 200 --clingo-budget-s 30` | Cloud T4, Linux x86_64, torch 2.10.0+cu128, clingo 5.8.0 |
| `h2_cpu_meta_cloud.json` | `python -m nspe.bench.cli --device cpu --batch-sizes 1 8 64 256 1024 --warmup 20 --reps 200 --clingo-budget-s 30` | Same cloud instance as above, CPU -- pairs with `h2_cuda_meta.json` for the CPU/GPU crossover comparison |
| `h2_cuda_synthetic_b10_r20.json` | `python -m nspe.bench.cli --device cuda --synthetic 10 20 --batch-sizes 1 64 1024 --warmup 20 --reps 200 --clingo-budget-s 30` | Same cloud T4 |
| `h2_cuda_synthetic_b50_r200.json` | `python -m nspe.bench.cli --device cuda --synthetic 50 200 --batch-sizes 1 64 1024 --warmup 20 --reps 200 --clingo-budget-s 30` | Same cloud T4 |
| `h2_cuda_synthetic_b100_r1000.json` | `python -m nspe.bench.cli --device cuda --synthetic 100 1000 --batch-sizes 1 64 1024 --warmup 20 --reps 200 --clingo-budget-s 30` | Same cloud T4 |

## H1/H3 — consistency and accuracy (`h1_h3/`)

5 seeds x 2 backbones (ViT-L-14, ViT-B-32-quickgelu) plus the Phase 4
ablation sweep. See `docs/h1_h3_findings.md` for the numbers these
back and what they mean; `nspe/ablate/cli.py`'s docstring for the
ablation matrix.

| file | command | machine |
|---|---|---|
| `h1_h3/results_s{0..4}.json` | `python -m nspe.eval.cli --clip-model ViT-L-14 --cache-dir ./emb_cache --reasoner-checkpoint ... --baseline-checkpoint ... --split validation` (one per seed, checkpoints trained via `nspe.train.cli --seed {0..4}`) | Kaggle T4 |
| `h1_h3/results_b32_s{0..4}.json` | Same, `--clip-model ViT-B-32-quickgelu` | Kaggle T4 |
| `h1_h3/ablations.json` | `python -m nspe.ablate.cli --clip-model ViT-L-14 --cache-dir ./emb_cache --device cuda` | Kaggle T4 |
| `h1_h3/results_val_s{0..4}.json` | Held-out-test rerun's validation pass -- fresh checkpoints, same protocol as `results_s*.json` above. Fits the thresholds `results_test_s*.json` applies. | Kaggle T4, commit `464a372` |
| `h1_h3/results_test_s{0..4}.json` | `python -m nspe.eval.cli --clip-model ViT-L-14 --cache-dir ./emb_cache --split test --thresholds-from results_val_s{i}.json` (one per seed; `--split test` refuses to run without `--thresholds-from`) | Kaggle T4, commit `464a372` |

## Reading these

- Compare `per_item_median_ms` across arms, not `median_ms`: one
  reasoner rep is a batched forward, one Clingo rep is `batch_size`
  sequential solves.
- Clingo's `p95_ms`/`p99_ms` are per-batch aggregates, not per-case
  tails, and are not comparable to the reasoner's.
- The CPU and CUDA sweeps exist as a pair. The CPU `batch=1` row holds
  hardware and batching fixed and is the cleanest vectorization-vs-search
  comparison; the rest of the speedup is batching and GPU.
