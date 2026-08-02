# H2: latency of the differentiable reasoner vs. an ASP solver

Snapshot as of commit `74b35dc`. Read this before quoting any speedup
number, and before touching `nspe/bench/`.

H2 is the project's core engineering claim: a GPU-native fuzzy logic
layer should beat an external ASP engine on latency and throughput.
The claim is real but **narrower than a single speedup figure suggests**,
and this document exists so the narrower version is the one that gets
reported.

## What the benchmark times

Three arms per batch size (`nspe/bench/cli.py::run_sweep`), on the same
policy:

| arm | configuration | role |
|---|---|---|
| `reasoner_crisp` | crisp t-norm over the **same fact sets Clingo receives** | The only arm whose verdicts are *certified* identical to Clingo's -- that is what `test/test_clingo_agreement.py` proves. **Speedup claims should lead with this arm.** |
| `reasoner_product` | product t-norm, graded truth degrees | The deployed configuration. Shows the thing you would actually ship is also fast. |
| `clingo` | `ClingoEngine.infer_verdicts` | Baseline. Grounds once at construction; per-case truth is set by solver assumptions, which is the standard multi-shot pattern and the fair way to run it. |

The graded arm draws its own input rather than sharing Clingo's. That
is harmless and deliberate: the reasoner is dense tensor arithmetic
with no value-dependent branching, so its runtime follows tensor
shapes, not values — and the shapes match the crisp arm exactly.

## Headline result: GPU latency is nearly flat, CPU and Clingo are not

Measured on a cloud T4 instance (`Linux-6.12.90+-x86_64`, torch
2.10.0+cu128, clingo 5.8.0, `meta_community_standards` policy,
fingerprint `89eb27db373e6939...`, `--reps 200 --clingo-budget-s 30`).
CPU and CUDA sweeps ran on the **same machine**, so the comparison
between them is apples-to-apples.

| batch | reasoner_crisp (ms) | clingo (ms) | speedup (crisp) | speedup (product) |
|---|---|---|---|---|
| 1 | 2.072 | 0.074 | **0.036x** | 0.024x |
| 8 | 2.139 | 0.578 | 0.270x | 0.191x |
| 64 | 2.142 | 4.667 | 2.18x | 1.52x |
| 256 | 2.128 | 19.428 | 9.13x | 6.27x |
| 1024 | 2.122 | 77.455 | 36.5x | 24.9x |
| 8192 | 2.128 | 617.964 | **290.4x** | 198.1x |

**The reasoner's GPU latency barely moves across four orders of
magnitude of batch size** (2.07 -> 2.13 ms from batch 1 to 8192): at
these sizes the actual tensor compute is negligible next to CUDA kernel
launch/sync overhead, so growing the batch is close to free. Clingo has
no such floor and scales linearly, because it solves one case at a
time. The whole speedup curve is that gap widening.

### The GPU only wins past a batch-size threshold, and CPU beats it below it

Same machine, same policy, CPU vs. CUDA:

| batch | CPU speedup (crisp) | CUDA speedup (crisp) |
|---|---|---|
| 1 | 0.067x | 0.036x |
| 8 | 0.470x | 0.270x |
| 64 | 3.71x | 2.18x |
| 1024 | 22.5x | 36.5x |

**Below roughly batch 64, CPU is the faster reasoner arm, not GPU.**
CUDA kernel launch overhead exceeds a CPU function call's overhead at
small batch sizes; GPU only becomes the better choice once its
near-flat latency starts to dominate a linearly-scaling alternative,
which happens somewhere between batch 64 and 1024 on this hardware. A
deployment doing single-case, low-latency inference should not assume
GPU is faster by default — it depends on the batch size actually
achievable in production.

### Synthetic b10_r20 (10 base predicates, 20 rules): consistent with the meta policy

| batch | speedup (crisp) |
|---|---|
| 1 | 0.042x |
| 64 | 2.67x |
| 1024 | 42.7x |

Same crossover shape as the meta policy. Whether the advantage grows or
shrinks as the rule base scales further (`b50_r200`, `b100_r1000`) is
**pending** — see Open questions below.

## Why the crisp arm is the one to lead with

The defensible form of the H2 claim is about the *formulation*, not
about raw arithmetic being faster: expressing policy inference as
batched differentiable tensor ops is what makes near-flat latency (and
therefore batching) possible at all. An ASP solver has no batch
dimension to exploit — each case is an independent search. The paper
should say precisely that, anchored to the crisp-arm numbers (the
certified-equivalent computation), rather than quoting the product
arm's larger number without the batch axis attached.

## Open questions (pending more T4 runs)

1. **Does the advantage grow or shrink as the rule base scales?** Only
   `b10_r20` is measured. `b50_r200` and `b100_r1000` are needed to see
   whether Clingo's grounding/search cost grows faster or slower than
   the reasoner's (near-constant, but with more predicates the compiled
   rule tensor and hence the per-call GPU work does grow).
2. **Exactly where does the CPU/GPU crossover sit?** Bracketed between
   batch 64 and 1024 above; a finer sweep (`--batch-sizes 64 128 256
   512`) would locate it precisely, which matters if a real deployment's
   batch size falls in that range.
3. **Does the crossover move with policy size?** If a larger rule base
   increases the reasoner's per-call GPU cost, the CPU/GPU crossover
   batch size would shift too.

## Threats to validity

Every one of these should survive into the paper's methods section.

1. **The speedup conflates three advantages: vectorization, batching,
   and GPU hardware.** This is why the CPU sweep exists and is not
   optional. The CPU `batch=1` row isolates vectorization vs. search
   with the other two held fixed — and on that row we lose, on both CPU
   and GPU. The CPU large-batch rows add batching. The CUDA delta at
   large batch adds hardware, but only once batch size clears the
   GPU-launch-overhead threshold described above.
2. **The reasoner's timed region excludes host-to-device transfer** of
   `mu0`, documented in `nspe/bench/cli.py`'s module docstring. Justified
   — production inference would not re-upload per case, and the neural
   extractor's output is already on device — but it is an exclusion.
3. **Clingo's timed region was inflated until commit `009d91b`.**
   `infer()` stringifies every atom in the model, which cost 21% of
   per-case time on the bundled policies and 37.5% on a 200-rule
   synthetic one, growing with the rule base. The benchmark now calls
   `infer_verdicts()`, which runs the identical solve and reads only
   verdict atoms. This *lowered* our reported speedup and any number
   predating that commit is inflated.
4. **Clingo's `p95_ms`/`p99_ms` are not comparable to the reasoner's.**
   One Clingo rep is `batch_size` sequential solves, so its tails are
   per-batch aggregates, not per-case latencies. Compare
   `per_item_median_ms`. Rep counts are budgeted by wall clock
   (`--clingo-budget-s`) rather than a fixed integer, because a fixed
   count costs time linear in batch size; the earlier formula degraded
   to 5 reps at batch 1024 against the reasoner's 200. (At batch 8192,
   `clingo.reps` landed at 47 under a 30s budget — comfortably above the
   floor of 5.)
5. **Only the crisp arm is certified.** `test_clingo_agreement.py`
   proves reasoner-Clingo agreement at `tnorm="crisp"`, and
   `test_bench_cli.py::test_crisp_arm_agrees_with_clingo_batched` proves
   the benchmark's own tensor construction preserves it under batching.
   The product arm is *not* claimed to compute the same function as
   Clingo — it computes graded truth degrees, which is the point of the
   approach, but it means the product speedup is a throughput
   comparison between different computations.
6. **Single-threaded Clingo.** Multi-shot single-solver is the standard
   deployment, so this is the right baseline, but it is worth stating
   that no attempt was made to parallelize it across cores.
7. **Small-to-medium policies, so far.** `meta_community_standards` and
   `synthetic_b10_r20` are both small. Conclusions about large rule
   bases await `b50_r200` and `b100_r1000`.
8. **GPU launch overhead is instance-specific.** The ~2.1ms floor
   reflects this T4 instance's kernel launch and synchronization cost;
   a different GPU (or a warmer/cooler thermal/scheduling state on a
   shared cloud instance) would shift the floor and hence the CPU/GPU
   crossover point. Report the GPU model and note this in the paper.

## Reproducing

`docs/colab_benchmark.md` has the full runbook. Committed artifacts and
the commands that produced them are indexed in `docs/results/README.md`.
`test/test_clingo_agreement.py` passing is the gate that makes any of
these numbers meaningful at all — if the reasoner does not reproduce
Clingo's answers on crisp inputs, the latency comparison compares
nothing.
