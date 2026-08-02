# H2: latency of the differentiable reasoner vs. an ASP solver

Snapshot as of commit `009d91b`. Read this before quoting any speedup
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

## The result is a crossover, not a constant speedup

Measured on an **Apple M5, CPU only** (torch 2.13.0, clingo 5.8.0,
`meta_community_standards` policy, fingerprint `89eb27db373e6939`,
`--reps 100 --clingo-budget-s 10`):

| batch | crisp (ms) | clingo (ms) | speedup (crisp) | crisp per-item (ms) | clingo per-item (ms) |
|---|---|---|---|---|---|
| 1 | 0.102 | 0.016 | **0.15x** | 0.10179 | 0.01575 |
| 8 | 0.108 | 0.125 | 1.16x | 0.01353 | 0.01567 |
| 64 | 0.137 | 0.996 | 7.30x | 0.00213 | 0.01557 |
| 256 | 0.612 | 3.950 | 6.45x | 0.00239 | 0.01543 |

**At batch 1, Clingo is roughly 6x faster than the reasoner.** The
crossover sits between batch 1 and batch 8. This is the honest shape of
the result and it should be reported, not buried: on a single case,
with hardware and batching held fixed, a mature stable-model solver
beats PyTorch's per-op overhead on a policy this small.

What the reasoner has is **amortization**. Its per-item cost falls 48x
from batch 1 to batch 64 (0.10179 -> 0.00213 ms) and then flattens,
while Clingo's is essentially constant across the whole sweep (0.01575
-> 0.01543 ms) because it solves sequentially, one case at a time.

That is the defensible form of the H2 claim, and it is a claim about
the *formulation*, not about raw arithmetic being faster: expressing
policy inference as batched differentiable tensor ops is what makes
batching possible at all. An ASP solver has no batch dimension to
exploit. The paper should say precisely that, rather than quoting
"7.3x" without the batch axis.

### CUDA and rule-base scaling — pending

Requires a T4 run (`docs/colab_benchmark.md`, Cells 4/4b/5). Two
questions to answer there:

1. **Where does the crossover move on GPU?** Expect it to move right
   (higher fixed launch overhead) and the plateau to go lower.
2. **How does the advantage scale with the rule base?** A preliminary
   CPU spot check on a synthetic 50-base/200-rule policy gave 4.75x at
   batch 64 against the meta policy's 7.30x — i.e. the advantage
   *narrowed* as the rule base grew. If that holds on GPU it is a
   limitation worth stating plainly.

## Threats to validity

Every one of these should survive into the paper's methods section.

1. **The speedup conflates three advantages: vectorization, batching,
   and GPU hardware.** This is why the CPU sweep exists and is not
   optional. The CPU `batch=1` row isolates vectorization vs. search
   with the other two held fixed — and on that row we lose. The CPU
   large-batch rows add batching. The CUDA delta adds hardware.
   Reporting only the CUDA large-batch number would be attributing all
   three to the formulation.
2. **The reasoner's timed region excludes host-to-device transfer** of
   `mu0`, documented in `nspe/bench/cli.py`'s module docstring. Justified
   — production inference would not re-upload per case, and the neural
   extractor's output is already on device — but it is an exclusion.
3. **Clingo's timed region was inflated until commit `009d91b`.**
   `infer()` stringifies every atom in the model, which cost 21% of
   per-case time on the bundled policies and 37.5% on a 200-rule
   synthetic one, growing with the rule base. The benchmark now calls
   `infer_verdicts()`, which runs the identical solve and reads only
   verdict atoms. This *lowered* our reported speedup (8.85x -> 7.53x
   at CPU batch 64) and any number predating that commit is inflated.
4. **Clingo's `p95_ms`/`p99_ms` are not comparable to the reasoner's.**
   One Clingo rep is `batch_size` sequential solves, so its tails are
   per-batch aggregates, not per-case latencies. Compare
   `per_item_median_ms`. Rep counts are budgeted by wall clock
   (`--clingo-budget-s`) rather than a fixed integer, because a fixed
   count costs time linear in batch size; the earlier formula degraded
   to 5 reps at batch 1024 against the reasoner's 200.
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
7. **Small policies.** `meta_community_standards` has few rules; the
   synthetic sweep exists to show scaling. Conclusions about large rule
   bases should come from the synthetic rows, not the bundled policy.

## Reproducing

`docs/colab_benchmark.md` has the full runbook. Committed artifacts and
the commands that produced them are indexed in `docs/results/README.md`.
`test/test_clingo_agreement.py` passing is the gate that makes any of
these numbers meaningful at all — if the reasoner does not reproduce
Clingo's answers on crisp inputs, the latency comparison compares
nothing.
