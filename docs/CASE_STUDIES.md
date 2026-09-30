# Measured optimization case studies

Local runs: **2026-09-30 Asia/Seoul** (JSON timestamps use UTC), RTX 4080 SUPER,
Ubuntu WSL2, Python 3.12.13, PyTorch 2.11.0+cu130, Triton 3.6.0. Values are
medians in microseconds. See [all tables](../results/SUMMARY.md),
[raw evidence](../results/) and [methodology](BENCHMARKING.md). These results
replace earlier sets; the [review record](records/REVIEW.md) explains why.

The GPU was shared with desktop applications (a browser, Blender and an editor).
Each report now records how busy the GPU was just before its run: 9-25% for
these runs. The benchmark spreads each case's samples over the run so a burst
spoils only a few ([methodology](BENCHMARKING.md)). The
[stability log](../results/benchmark-stability.log) repeated one suite five times on
2026-09-26: most softmax, RMSNorm and GEMM ratios agreed within 10%, while several
row-sum and cold-cache ratios varied by more than 30%. Treat a difference smaller
than that spread as unresolved. Every comparison below is within one run.

How to read the tables: 1 us (microsecond) is 1/1000 of a millisecond, and lower
is faster. A speedup of 1.75x means PyTorch took 1.75 times as long as Triton;
below 1.00x, Triton was slower. *Warm* means inputs were likely still in the 64 MiB
L2 cache from the previous call; *cold* means L2 was flushed first. More terms are
in the [glossary](../learning/GLOSSARY.md#measuring).

## Specialize what the model fixes

Hypothesis: a size that is a compile-time constant (`tl.constexpr`) triggers a
compile for every new value, while a runtime size can cost speed. A model fixes
its row width and weight shapes, but batch size, token count and row count change
per request. Specializing only the fixed sizes should keep the speed without
compiling per batch.

Three versions of the same kernels, FP16, CUDA-graph replay, warm L2
([log](../results/specialization-sweep.log), reproduced by
`python scripts/specialization_sweep.py`). Brackets show softmax registers per thread.
This sweep is from 2026-09-25: it measures single-block row kernels and the GEMM,
which this review did not change, and the repeat on 2026-09-26 was dominated by
other GPU users (one 5.0 us case measured 52.7 us).

| Shape | Kernel | Every size compile-time (pre-review) | Every size runtime | Current: width and GEMM N/K compile-time |
| --- | --- | ---: | ---: | ---: |
| 512 x 4097 | softmax | 5.66 [48] | 8.47 [128] | 5.60 [48] |
| 512 x 4104 | softmax | 4.39 [56] | 7.92 [128] | 4.71 [56] |
| 512 x 5120 | softmax | 5.01 [54] | 6.55 [85] | 5.05 [54] |
| 512 x 4097 | RMSNorm | 8.77 | 11.06 | 8.43 |
| 512 x 4104 | RMSNorm | 5.86 | 10.96 | 5.94 |
| 1024 x 1024 | softmax | 2.76 [25] | 2.76 [23] | 2.75 [23] |
| 512 x 3072 | softmax | 3.45 [40] | 3.65 [46] | 3.41 [39] |
| 127 x 255 x 65 | GEMM, fixed tile | 2.32 | 2.56 | 2.49 |
| 512 x 512 x 512 | GEMM, fixed tile | 5.96 | 6.59 | 5.99 |

Observations:

- With every size at runtime, widths that do not fill their power-of-two block ran
  up to 1.9x slower (RMSNorm at 4104: 10.96 vs 5.86 us). The 1024-wide and 3072-wide
  cases did not change.
- All three versions issue the same number of loads, exponentials and stores. What
  changes is register use: 48 per thread for softmax at 4097 when the width is
  known, 128 when it is not, which limits how many blocks run at once. Raising the
  warp count did not recover the loss
  ([warp sweep](../results/archive/runtime-args-warp-sweep.log)). An earlier version
  of this page said compile-time sizes let the compiler "fold masks"; the generated
  code does not support that claim, and it is withdrawn.
- The pre-review version's cost was real where a varying size was compile-time:
  every new vector length in `add` and every new GEMM M recompiled, and the local
  cache held 128 builds of the add kernel alone. The current kernels compile once
  per width or weight shape and never per batch size; a GPU test checks that new
  lengths, row counts, strides and GEMM M values within one tuning bucket reuse
  existing kernels.
- The current GEMM uses a one-dimensional grid with grouped tile order. Its extra
  index arithmetic costs a little at the tiny 127 x 255 x 65 shape (2.49 vs 2.32 us)
  and nothing at 512.
- The learning exercises keep every size at runtime. That is always correct, and
  the numbers above show what the portfolio kernels gain by specializing further.

## Cache residency: warm numbers can exceed DRAM

The same current kernels, FP16 graph replay, warm and with L2 flushed before every
call ([warm](../results/rtx4080super-triton-fp16.json),
[cold](../results/rtx4080super-triton-fp16-cold.json)); ratios are `torch / Triton`:

| Case | Warm L2 | Cold L2 |
| --- | ---: | ---: |
| Softmax 1024 x 1024, 4 warps | 1.73x | 1.60x |
| Softmax 512 x 4097, 4 warps | 2.52x | 1.38x |
| Row sum 512 x 4097 | 1.67x | 1.27x |
| RMSNorm 512 x 4097, vs eager composition | 7.20x | 3.60x |
| GEMM 512 x 512 x 512, autotuned | 1.13x | 0.77x |

- Warm, the 1024-wide softmax moved 1,731 GB/s of logical traffic, more than the
  ~736 GB/s DRAM can deliver. Cold, it moved 444 GB/s. Warm results describe
  cache-resident repeated calls, not saved DRAM traffic.
- Cold timing has a floor of about 3.7 us per call (a 257-element add takes
  3.657 us), so small cold cases are dominated by that overhead.
- Cold ratios moved the most between runs ([stability log](../results/benchmark-stability.log)).
  Warm and cold agree that the 1024-wide and 4097-wide softmax win and that fusion
  wins most for RMSNorm.
- At 16384 x 4096, which exceeds L2 even when warm, softmax measured
  1.03x with four warps and 1.04x with eight, and row sum 1.07x; on 2026-09-26 the
  row sum ranged 0.82-1.32x across five runs. Both operators run near the bandwidth
  other GPU users leave, so there is little left to win at that size.

## Wide rows: loop in one program, or split across programs

Rows wider than 8192 no longer fit one block. `_row_plan` in
[triton_kernels.py](../src/kernel_portfolio/triton_kernels.py) picks one of two
ways to cover them:

- **Looped**: one program per row walks the row in chunks. Softmax keeps an online
  maximum and sum per lane (workshop A1's merge rule), then rereads the row to write
  probabilities; RMSNorm sums squares, then rereads to scale.
- **Split**: with fewer rows than SMs, one program per row cannot fill the GPU.
  Each row is split into chunks across programs: the first kernel writes each
  chunk's statistics, and the second has every program merge its row's statistics
  itself and write its chunk. Softmax splits whenever it would loop; RMSNorm only
  above width 16384, where splitting measured faster; row sum gained too little to
  pay for a second launch and always loops.

The online merge needs one guard the single-block kernel does not. A lane or chunk
whose values so far are all `-inf` (a masked prefix in attention scores) has a
maximum of `-inf`, and rescaling by `exp(-inf - (-inf))` is NaN. The first looped
kernel returned NaN for every masked row wider than 8192 while narrower rows were
correct, because the finite-input tests never masked anything. Such a lane now
shifts by 0 and keeps a zero sum; the split kernels and the CUDA online kernel use
the same guard, and `test_masked_softmax` covers every plan.

FP16 graph replay ([warm](../results/rtx4080super-triton-fp16.json),
[cold](../results/rtx4080super-triton-fp16-cold.json),
[compiled](../results/rtx4080super-rmsnorm-compiled.json)); both shapes split
softmax and RMSNorm:

| Shape | Operation | PyTorch | Triton | Warm speedup | Cold speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 x 32769 | row sum (looped) | 2.799 | 2.550 | 1.10x | 1.03x |
| 4 x 32769 | softmax | 7.885 | 2.694 | 2.93x | 1.82x |
| 4 x 32769 | RMSNorm, vs torch.compile | 37.308 | 2.765 | 13.49x | |
| 64 x 131072 | row sum (looped) | 13.790 | 5.858 | 2.35x | 1.22x |
| 64 x 131072 | softmax | 24.439 | 18.091 | 1.35x | 1.06x |
| 64 x 131072 | RMSNorm, vs torch.compile | 178.074 | 27.429 | 6.49x | |

Before the split path, the looped softmax lost at 4 x 32769 (0.83x warm): four
programs cannot fill 80 SMs. The sweep behind the rule
([log](../results/row-chunk-sweep.log), `python scripts/row_chunk_sweep.py`) forces
each plan; all choices for a shape are sampled in shuffled rounds:

| Shape | Operation | PyTorch | Looped 2048/4w | Looped 4096/8w | Looped 8192/16w | Split 2048/4w | Split 4096/8w | Automatic |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 x 32769 | softmax | 8.69 | 16.79 | 11.67 | 10.55 | 2.97 | 3.07 | 2.97 |
| 4 x 32769 | RMSNorm | - | 18.18 | 14.44 | 15.35 | 3.12 | 3.53 | 3.07 |
| 64 x 131072 | softmax | 25.86 | 37.12 | 23.50 | 21.55 | 18.69 | 18.53 | 18.59 |
| 64 x 131072 | RMSNorm | - | 40.81 | 29.13 | 27.75 | 29.27 | 29.03 | 29.13 |
| 128 x 32769 | softmax | 17.82 | 21.45 | 19.81 | 22.17 | 16.74 | 17.41 | 22.02 |
| 128 x 32769 | RMSNorm | - | 24.42 | 24.58 | 27.14 | 18.23 | 19.05 | 27.24 |
| 1024 x 16384 | softmax | 61.18 | 38.18 | 40.38 | 54.22 | 43.88 | 69.38 | 33.74 |
| 1024 x 16384 | RMSNorm | - | 167.83 | 151.65 | 199.11 | 230.55 | 267.52 | 200.86 |

Observations:

- Splitting pays when rows are few and wide: at 4 x 32769 it cut softmax from
  10.55 us (best looped choice) to 2.97 us, and RMSNorm from 14.44 to 3.12 us.
  At 1024 x 16384, with plenty of rows, looping wins, and splitting RMSNorm would
  take 1.5x as long as the best looped choice.
- The rule is conservative. At 128 x 32769, above the SM count, the split path was
  also faster (16.74 vs 19.81 us for softmax), but at 128 rows the prototype
  sweep found looping faster at width 16384 and even at 131072, so the simple
  SM-count rule stays.
- At 64 x 131072, splitting helped softmax (18.53 vs 21.55 us) but not
  RMSNorm (29.03 vs 27.75 us looped): within noise, the rule's choice there
  neither gains nor loses much.
- The automatic column runs the same kernel and configuration as one of the forced
  columns, so the gap between those two measures this run's noise: up to
  20% (RMSNorm at 1024 x 16384).
- `torch.compile` handles these widths poorly: RMSNorm at 64 x 131072 took
  178.074 us compiled against 27.429 us here, and FP32 softmax took
  166.227 us compiled against 48.947 us eager
  ([softmax source](../results/rtx4080super-softmax-compiled.json)). The generated code
  ([log](../results/inductor-wide-rows.log), `python scripts/inspect_inductor.py`)
  shows why. For RMSNorm, Inductor launches 3 kernels: it splits each row into
  only four pieces (256 programs, each looping over 32768 elements), merges them,
  then reads the inputs again in a separate pointwise kernel. For softmax it
  launches 5, because it disables online softmax when it splits the reduction:
  one split pass for the maximum, another for the sum, and a third read to write.
  At 1024 x 1024 both compile to a single kernel.

## Training: backward kernels through autograd

Softmax and residual RMSNorm train. Their backward kernels follow the same row
plans as the forward kernels; `library.py` registers them with autograd, and
`ops.softmax` and `ops.residual_rmsnorm` route gradient-tracking inputs through the
custom ops. Softmax saves its output and computes `y * (dy - sum(y * dy))`. The
RMSNorm backward recomputes each row's inverse RMS instead of saving it, writes the
shared input gradient for `x` and `residual`, and sums the weight gradient over rows
in a separate kernel: programs own blocks of columns and groups of rows, write FP32
partial sums, and one sum adds the groups, so the result is deterministic without
atomics.

A training step here is one forward and one backward pass through autograd, which
is also the only way a CUDA graph can capture a backward pass. FP16 graph replay
([source](../results/rtx4080super-training-fp16.json)); ratios are PyTorch eager
over Triton:

| Shape | Operation | PyTorch | `F.rms_norm` | torch.compile | Triton | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32 x 127 | softmax | 3.514 |  | 2.082 | 2.070 | 1.70x |
| 1024 x 1024 | softmax | 9.523 |  | 10.581 | 4.779 | 1.99x |
| 512 x 4097 | softmax | 23.415 |  | 22.391 | 11.230 | 2.09x |
| 16384 x 4096 | softmax | 2008.644 |  | 1257.165 | 1258.189 | 1.60x |
| 4 x 32769 | softmax | 14.711 |  | 72.943 | 5.086 | 2.89x |
| 64 x 131072 | softmax | 116.292 |  | 416.700 | 68.332 | 1.70x |
| 32 x 127 | RMSNorm | 42.257 | 16.759 | 3.618 | 6.076 | 6.96x |
| 1024 x 1024 | RMSNorm | 87.893 | 41.977 | 21.811 | 14.565 | 6.03x |
| 512 x 4097 | RMSNorm | 217.805 | 86.767 | 49.625 | 27.334 | 7.97x |
| 16384 x 4096 | RMSNorm | 21049.611 | 9595.389 | 2791.595 | 2356.565 | 8.93x |
| 4 x 32769 | RMSNorm | 41.088 | 86.903 | 194.253 | 10.069 | 4.08x |
| 64 x 131072 | RMSNorm | 1860.028 | 1019.051 | 437.385 | 183.125 | 10.16x |

Observations:

- Triton's maximum gradient error against the FP64 gradient was equal to or below
  PyTorch's own in 11 of 12 cases; the rest differed by one FP16 rounding
  step (softmax 16384 x 4096: 6.1e-05 against 3.1e-05) ([JSON](../results/rtx4080super-training-fp16.json)).
- Against `torch.compile`, which fuses forward and backward with AOTAutograd,
  softmax ties at the smallest and the DRAM-bound shape and wins elsewhere, most at
  the wide rows where Inductor's split reductions hurt. RMSNorm loses only at
  32 x 127 (3.618 vs 6.076 us), where launches
  dominate: our step runs five kernels (forward, backward, the weight-gradient
  partials, and PyTorch's sum and cast of those partials).
- Eager training through the custom ops is host-bound for small shapes. With event
  timing ([source](../results/rtx4080super-training-events.json)), a 1024 x 1024
  softmax step took 309.965 us against 106.491 us for PyTorch, although its
  kernels run faster: two Python custom ops and the autograd bookkeeping cost more
  than PyTorch's C++ path. At 16384 x 4096 the kernels dominate and Triton wins
  (2423.125 vs 21159.121 us for RMSNorm). Compile the model, as in the next section,
  to remove the per-call cost.

## torch.compile: pay dispatch once per graph

`kernel_portfolio.library` registers every operator with `torch.library.triton_op`,
so `torch.compile` can place them in one graph and, with `mode="reduce-overhead"`,
replay the graph as a CUDA graph. Host microseconds per call, variants interleaved
in one process ([log](../results/compile-overhead.log),
`python scripts/compile_overhead.py`):

| Workload | Eager `ops.*` | Custom ops, torch.compile | Custom ops, reduce-overhead |
| --- | ---: | ---: | ---: |
| One softmax, 32 x 127 | 22.02 | 29.82 | 46.54 |
| 16 ops: 8 x (residual RMSNorm + softmax) | 406.43 | 155.14 | 51.38 |

Observations:

- For one tiny operator, compiling does not pay: guard checks and graph bookkeeping
  cost more than the launch they replace. Calling through `torch.ops` eagerly also
  adds dispatcher cost (39.93 us), so the plain `kernel_portfolio.*` functions stay
  the eager API.
- Across a 16-op chain, reduce-overhead cut host time per call 7.9x, from 406.43 to
  51.38 us: one graph replay instead of 16 Python launches. That is where the eager
  wrapper's per-call cost, measured below, actually goes away.
- `torch.library.opcheck` validates each registration, including the autograd
  registrations of softmax and RMSNorm, and GPU tests check that compiled graphs,
  including dynamic row counts and a training step, match eager results.

## Softmax: fusion helps device time, dispatch can erase the gain

Hypothesis: keeping a row's max, exponentials and sum in one program removes
intermediates and launch overhead compared with an eager decomposition. Compare
against `torch.softmax` as well, since it is already a specialized operator.

FP16, CUDA graph replay, warm L2; [source JSON](../results/rtx4080super-triton-fp16.json):

| Shape | torch.softmax | Triton 4 warps | Triton 8 warps | Best Triton speedup |
| --- | ---: | ---: | ---: | ---: |
| 32 x 127 | 1.502 | 1.126 | 1.126 | 1.33x |
| 1024 x 1024 | 4.198 | 2.423 | 3.345 | 1.73x |
| 512 x 4097 | 12.732 | 5.045 | 5.654 | 2.52x |
| 16384 x 4096 | 503.533 | 489.199 | 484.489 | 1.04x |

Observation: four warps were best at widths 1024 and 4097; at 32 x 127 and at the
largest shape the two differ by less than this machine's run-to-run spread. The
all-runtime version preferred eight warps at 4097, because its extra registers per
thread made four warps worse; a warp-count conclusion depends on the rest of the
kernel. The inspected 1024-wide softmax reports 23 registers/thread, zero spills
and 16 bytes of shared memory.

An ordinary event-timed wrapper run reverses the mid-size result: for FP16
1024x1024, PyTorch took **8.363 us** and the 4-warp wrapper **21.436 us**
([source](../results/rtx4080super-softmax-events.json)). A host-side breakdown
([log](../results/wrapper-overhead.log), `scripts/wrapper_overhead.py`) puts
Triton's own Python launcher at 11.0 us of the 20.9 us call. Input validation,
output allocation and the device guard add about 4 us together. At 16384 x 4096
the kernel is long enough to hide dispatch
(513.225 vs 526.848 us). Eager calls to a short kernel lose; the compiled
graph above is how to remove that cost. Event timings include host work, so they
move with CPU load more than graph timings do.

The separate FP32 compiled comparison keeps semantics and dtype matched
([source](../results/rtx4080super-softmax-compiled.json)). At 1024x1024, eager
PyTorch took **4.267 us**, compiled PyTorch **5.905 us** and Triton 4 warps
**3.404 us**. At 16384x4096 all three were within 2% (1047.552, 1034.171 and
1026.833 us). Inductor again warned that it split a reduction and disabled online
softmax for the wide shapes; the wide-rows section above shows the kernels it emits.

## Residual RMSNorm: compare with a compiler, not just eager composition

Hypothesis: fusing FP32 residual addition, the row reduction and scaling avoids
writing/reading a full intermediate activation. The exact FP32 residual contract
is shared by all variants.

FP16, CUDA graph replay, warm L2; [source JSON](../results/rtx4080super-rmsnorm-compiled.json):

| Shape | PyTorch composition | `F.rms_norm` | torch.compile | Triton | Compiled / Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 x 127 | 12.698 | 9.216 | 1.049 | 1.024 | 1.02x |
| 1024 x 1024 | 31.812 | 21.367 | 5.257 | 2.697 | 1.95x |
| 512 x 4097 | 53.760 | 46.490 | 11.700 | 7.441 | 1.57x |
| 16384 x 4096 | 7343.992 | 4843.247 | 809.233 | 754.210 | 1.07x |
| 4 x 32769 | 16.828 | 35.942 | 37.308 | 2.765 | 13.49x |
| 64 x 131072 | 494.626 | 442.470 | 178.074 | 27.429 | 6.49x |

Observation: the 7-12x gain against the eager composition mostly counts launches.
The built-in `F.rms_norm` is 1.12-1.52x faster than that composition at most
shapes, but 2.1x slower at 4 x 32769, because the FP32 casts and residual
addition still run as separate kernels. Against `torch.compile`, Triton ties at
the smallest shape (1.02x) and at the DRAM-bound 16384 x 4096 (1.07x), and wins at
the mid-size and wide shapes, by the most where the split path runs. With L2
flushed, the 1024 x 1024 ratio against the composition falls from 11.56x to
3.62x ([cold JSON](../results/rtx4080super-triton-fp16-cold.json)). The evidence supports
fusing casts, residual and normalization over eager PyTorch, and dedicated wide-row
plans. It does not show a hand-written kernel beating a compiler at DRAM-bound
sizes.

## GEMM: grouped tile order and a larger search

The GEMM now launches a one-dimensional grid of output tiles. `GROUP_M=1` is plain
row-major order (the fixed configuration); the autotuned configurations use
`GROUP_M=8`, so neighbouring programs share B columns while those are still in L2.
The autotuner searches eight tile, warp, stage and group configurations, keyed on a
power-of-two bucket of M. FP16 graph replay;
[warm](../results/rtx4080super-triton-fp16.json) and
[cold](../results/rtx4080super-triton-fp16-cold.json) sources:

| M x N x K | torch.mm | Fixed tile | Autotuned | Tuned speedup, warm | Tuned speedup, cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| 127 x 255 x 65 | 3.925 | 2.150 | 2.150 | 1.83x | 1.32x |
| 512 x 512 x 512 | 5.393 | 5.393 | 4.779 | 1.13x | 0.77x |
| 1024 x 1024 x 1024 | 26.385 | 28.501 | 26.010 | 1.01x | 1.34x |
| 4096 x 4096 x 4096 | 1583.435 | 2008.463 | 1461.794 | 1.08x | 1.07x |

At 4096 x 4096 x 4096 the autotuned kernel (BM: 128, BN: 128, BK: 32, GROUP_M: 8, num_warps: 8) reached
94.0 TFLOP/s against 86.8 for `torch.mm` warm, and 94.4 against 88.1 cold.
Across the five runs of 2026-09-26 its lead ranged from 1.03x to 1.12x
([stability log](../results/benchmark-stability.log)).
Isolating the grouping with that same tile ([log](../results/gemm-grouping.log) from
2026-09-25; the GEMM kernel has not changed since, `python scripts/gemm_grouping.py`): `GROUP_M` = 1, 4 and 8 ran within 4% of each
other at 1024, 2048 and 4096 (4096^3: 1281.2, 1288.7 and 1329.9 us). The gain over
`torch.mm` came from the larger tiles the new search offers, not from grouped
order. This GPU's 64 MiB L2 already holds all of B (32 MB at 4096^3 in FP16), so
grouping has little reuse left to recover; a GPU with a smaller L2 could differ.
Two cautions. `torch.mm` runs with reduced-precision reductions disabled so both
sides accumulate in FP32 (see [methodology](BENCHMARKING.md)); PyTorch's default
may pick different kernels. And this is one GPU and one shape family, not a claim
of beating cuBLAS in general. The fixed kernel's saved PTX contains `mma.sync`
instructions and reports 55 registers/thread, zero spills and 6144 bytes of shared
memory.

At 1,048,576 vector elements, PyTorch took **2.287 us**, Triton with block size
256 **2.821 us** and block size 1024 **2.253 us** warm. Cold, all three were
within 1% (15.292-15.497 us), so the warm differences are an L2-resident effect.
At 16,777,216 elements, which exceeds L2, the three measured 153.1-185.0 us here;
their ratios moved by up to 28% between runs on 2026-09-26, more than any difference
between the variants, so they are best read as tied.

## CUDA: parallel reduction, online statistics and a numerical fix

The native Windows CUDA program times four 1024x1024 FP32 softmax kernels, sampled in
rotating order so they share the same conditions
([source](../results/rtx4080super-cuda.json), [three runs](../results/cuda-softmax-runs.log)):

| Kernel | What it does | Median (us) |
| --- | --- | ---: |
| `softmax_serial` | One thread per row; the intentionally weak baseline | 477.348 |
| `softmax_parallel` | One 256-thread block per row; max, sum and write passes (three reads) | 9.814 |
| `softmax_online_scalar` | Online (max, sum) in one pass, then write (two reads) | 9.747 |
| `softmax_online` | The same with 16-byte `float4` loads | 9.827 |

Across three runs the online kernels were 1.00-1.12x faster than the parallel one,
and the `float4` and scalar versions stayed within 3% of each other. At this size
neither the removed read nor wider loads is a dependable gain: the 4 MB input stays
in L2 between passes, so the extra read is cheap. An earlier single run, with kernels
timed one after another, showed `float4` 1.11x faster and credited wider loads; with
interleaved sampling that difference did not hold up. Compare the cold-L2 Triton
softmax above, where a second read of each row does cost time. The serial baseline
demonstrates parallelization; it is not a claim against PyTorch or Triton.

The original serial accumulation produced a row sum around 1.000025 at width
8192 and failed the 2e-5 row-normalization tolerance. Compensation fixed the
failure without loosening the tolerance. Max subtraction alone does not eliminate
all numerical error.

The three kernels share one output buffer in the test program, which at first
hid a class of bugs: an online kernel deliberately broken to skip its last
`float4` still passed, because the parallel kernel's correct values were already
in the buffer. The test now fills the buffer with NaN before every kernel, and the
same broken kernel fails. CUDA memcheck reported zero errors and racecheck zero hazards
on the current binary, which runs all three kernels
([memcheck](../results/cuda-memcheck.log), [racecheck](../results/cuda-racecheck.log)).
