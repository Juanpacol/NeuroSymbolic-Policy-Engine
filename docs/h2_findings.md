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

**The crossover, located precisely: between batch 256 and 512, on the
meta policy.** A denser sweep (`--batch-sizes 64 128 256 512`) fills in
the gap the coarse table above left between 64 and 1024:

| batch | CPU per-item (ms) | CUDA per-item (ms) |
|---|---|---|
| 64 | 0.0184 | 0.0330 |
| 128 | 0.0104 | 0.0165 |
| 256 | 0.0067 | 0.0083 |
| 512 | **0.0042** | **0.0041** |

CPU stays ahead through 256, and the two are within rounding error of
each other at 512 (CUDA a hair faster). The crossover is not a wide
band — it is a narrow window right around batch 512 on this policy and
this T4 instance, not "somewhere in 64-1024" as the coarser sweep could
only bracket.

### Rule-base scaling: the advantage has a ceiling, not an open-ended climb

Three synthetic policies, same T4, `--batch-sizes 1 64 1024`:

| policy | speedup @ batch 1 | speedup @ batch 64 | speedup @ batch 1024 | reasoner_crisp @ batch 1024 (ms) | clingo @ batch 1024 (ms) |
|---|---|---|---|---|---|
| b10_r20 (10 base, 20 rules) | 0.042x | 2.67x | 42.7x | 2.249 | 96.02 |
| b50_r200 (50 base, 200 rules) | 0.166x | 11.19x | **177.7x** (peak) | 2.456 | 436.52 |
| b100_r1000 (100 base, 1000 rules) | 0.371x | 23.50x | 151.6x | **8.042** | 1219.48 |

The speedup climbs from `b10` to `b50`, then **falls back** at
`b100_r1000` despite the larger rule base — and the reason is visible
directly in the reasoner's own latency. From `b10_r20` to `b50_r200`,
`reasoner_crisp` at batch 1024 barely moves (2.249 -> 2.456 ms): still
inside the near-flat, launch-overhead-bound regime described above.
From `b50_r200` to `b100_r1000` it jumps to 8.042 ms — a 3.3x increase
— because at 1000 rules and batch 1024 the actual tensor compute
(evaluating every rule against every case) stops being negligible next
to kernel launch overhead. Clingo grows too (436 -> 1219 ms, 2.8x), but
the reasoner's growth outpaces it over that step, so the ratio between
them shrinks.

**The near-flat-latency advantage has a ceiling.** It holds as long as
per-call GPU compute stays small relative to launch/sync overhead; past
some combination of rule count and batch size, the reasoner starts
paying for the work it is actually doing, and the speedup curve bends
down. The reasoner remains overwhelmingly faster (151.6x) at the
largest policy tested, but "the advantage grows without bound as the
rule base grows" is not the claim these three points support — the
honest claim is that it grows, peaks, and then gives some back once
real compute enters the picture.

**The peak sits past `b50_r200`, not at it.** `b70_r500` (70 base
predicates, 500 rules), the point in between the three already
committed, was run to locate the inflection precisely:

| policy | speedup @ batch 1024 | reasoner_crisp @ batch 1024 (ms) | clingo @ batch 1024 (ms) |
|---|---|---|---|
| b10_r20 | 42.7x | 2.249 | 96.02 |
| b50_r200 | 177.7x | 2.456 | 436.52 |
| **b70_r500** | **180.7x (true peak)** | 3.747 | 676.98 |
| b100_r1000 | 151.6x | 8.042 | 1219.48 |

`b70_r500` edges out `b50_r200` (180.7x vs. 177.7x) rather than
sitting between it and `b100_r1000`'s fallen-back 151.6x. Clingo's cost
keeps climbing roughly linearly with the rule count (436 -> 677 ->
1219 ms), while the reasoner's latency is still only creeping up
(2.456 -> 3.747 ms) at 70 rules -- still inside the near-flat,
launch-overhead-bound regime, so the *ratio* keeps improving a little
further before the reasoner's own compute cost catches up and the
curve bends down by `b100_r1000`. The peak is a genuine local maximum
around 500-700 rules on this hardware, not a smooth monotonic decline
starting immediately after `b50_r200`.

**Filling in `b100_r1000`'s batch axis (previously only 1, 64, 1024)
shows the bend is smooth, not a sharp knee:**

| batch | speedup (crisp) |
|---|---|
| 8 | 2.77x |
| 128 | 47.7x |
| 256 | 91.6x |
| 512 | 160.8x |
| 1024 | 151.6x |

Speedup climbs smoothly through 512 and only turns over between 512
and 1024 -- consistent with launch-overhead-bound behavior giving way
gradually to compute-bound behavior as batch size grows, not a discrete
regime switch at one particular batch size.

## Why the crisp arm is the one to lead with

The defensible form of the H2 claim is about the *formulation*, not
about raw arithmetic being faster: expressing policy inference as
batched differentiable tensor ops is what makes near-flat latency (and
therefore batching) possible at all. An ASP solver has no batch
dimension to exploit — each case is an independent search. The paper
should say precisely that, anchored to the crisp-arm numbers (the
certified-equivalent computation), rather than quoting the product
arm's larger number without the batch axis attached.

## Open questions (resolved)

All three were open in the previous version of this document; all
three are now answered by the sweeps in the two sections above.

1. **Exactly where does the CPU/GPU crossover sit?** Between batch 256
   and 512 on the meta policy — CPU still ahead at 256 (0.0067ms vs.
   0.0083ms per-item), the two within rounding error at 512. Resolved,
   not just bracketed.
2. **Exactly where does the rule-base-scaling inflection sit?** The
   true peak is `b70_r500` (180.7x), past `b50_r200` (177.7x) rather
   than at it — the climb continues a little further than the original
   three points suggested before falling back at `b100_r1000`
   (151.6x). Filling in `b100_r1000`'s batch axis (8, 128, 256, 512)
   shows the fall-back is a smooth bend through batch 512-1024, not a
   sharp knee at one specific batch size.
3. **Does the CPU/GPU crossover point move with policy size? Yes,
   substantially.** On `b100_r1000`, CUDA already leads at batch 64
   (0.0499ms vs. CPU's 0.2979ms per-item) and the gap widens to ~40x by
   batch 1024 (0.0079ms vs. 0.3165ms) — nowhere near the meta policy's
   256-512 crossover. A larger rule base pushes the crossover down to
   much smaller batch sizes, because CPU's per-rule cost scales worse
   with rule count than the GPU's parallel-across-rules cost does. A
   deployment with a large policy should default to GPU even at modest
   batch sizes; the "CPU wins below ~512" rule of thumb from the meta
   policy does **not** transfer to a larger rule base.

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
7. **The rule-base scaling sweep now has 4 points, one with 5 batch
   sizes -- still not a dense characterization.** Enough to locate the
   true peak (`b70_r500`) and see the fall-back through `b100_r1000`'s
   batch axis is a smooth bend, but the space between `b70_r500` and
   `b100_r1000` and beyond `b100_r1000` remains unsampled.
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
