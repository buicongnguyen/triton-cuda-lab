# Measured optimization case studies

Local runs: **2026-09-26 Asia/Seoul** (JSON timestamps use UTC), RTX 4080 SUPER,
Ubuntu WSL2, Python 3.12.13, PyTorch 2.11.0+cu130, Triton 3.6.0. Values are
medians in microseconds. See [all tables](../results/SUMMARY.md),
[raw evidence](../results/) and [methodology](BENCHMARKING.md). These results
replace earlier sets; the [review record](records/REVIEW.md) explains why.

The GPU was shared with desktop applications (a browser, Blender and an editor)
during these runs, and bursts of their work slowed whole cases 2-3x until the
benchmark learned to spread its samples ([methodology](BENCHMARKING.md)). The
[stability log](../results/benchmark-stability.log) repeats the same suite five times:
most softmax, RMSNorm and GEMM ratios agree within 10%, while several row-sum and
cold-cache ratios vary by more than 30% between runs. Treat a difference smaller than
that spread as unresolved. Every comparison below is within one run.

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
| Softmax 1024 x 1024, 4 warps | 1.76x | 1.23x |
| Softmax 512 x 4097, 4 warps | 2.52x | 1.35x |
| Row sum 512 x 4097 | 1.67x | 0.90x |
| RMSNorm 512 x 4097, vs eager composition | 7.11x | 4.20x |
| GEMM 512 x 512 x 512, autotuned | 1.13x | 0.79x |

- Warm, the 1024-wide softmax moved 1,755 GB/s of logical traffic, more than the
  ~736 GB/s DRAM can deliver. Cold, it moved 422 GB/s. Warm results describe
  cache-resident repeated calls, not saved DRAM traffic.
- Cold timing has a floor of about 3.7 us per call (a 257-element add takes
  3.727 us), so small cold cases are dominated by that overhead.
- Cold ratios moved most between runs: the cold row sum at 512 x 4097 measured
  0.90x here and 1.23-1.25x in three other runs
  ([stability log](../results/benchmark-stability.log)). Warm and cold agree that
  the 1024-wide and 4097-wide softmax win and that fusion wins most for RMSNorm.
- At 16384 x 4096, which exceeds L2 even when warm, softmax measured
  1.03x-1.07x (543 GB/s at best) and row sum
  0.83x; across five runs the row sum ranged 0.82-1.32x. Both operators run near the
  bandwidth other GPU users leave, so there is little left to win at that size.

## Wide rows: one looped program per row

Rows wider than 8192 no longer fit one block, so `_row_sum_looped`,
`_softmax_looped` and `_rmsnorm_looped` walk each row in chunks. Softmax keeps an
online maximum and sum per lane (workshop A1's merge rule), then rereads the row
to write probabilities; RMSNorm sums squares, then rereads to scale. One program
still owns one row.

The online merge needs one guard the single-block kernel does not. A lane whose
values so far are all `-inf` (a masked prefix in attention scores) has a running
maximum of `-inf`, and rescaling by `exp(-inf - (-inf))` is NaN. The first looped
kernel returned NaN for every masked row wider than 8192 while narrower rows were
correct, because the finite-input tests never masked anything. Such a lane now
shifts by 0 and keeps a zero sum; `test_masked_softmax` covers both kernels and
both chunk sizes, and the CUDA online kernel has the same guard.

FP16 graph replay ([warm](../results/rtx4080super-triton-fp16.json),
[cold](../results/rtx4080super-triton-fp16-cold.json),
[compiled](../results/rtx4080super-rmsnorm-compiled.json)):

| Shape | Operation | PyTorch | Triton | Warm speedup | Cold speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 64 x 131072 | row sum | 14.124 | 5.871 | 2.41x | 0.86x |
| 64 x 131072 | softmax | 23.415 | 20.378 | 1.15x | 1.04x |
| 64 x 131072 | RMSNorm, vs torch.compile | 151.245 | 26.897 | 5.62x | |
| 4 x 32769 | row sum | 2.799 | 2.526 | 1.11x | 1.02x |
| 4 x 32769 | softmax | 7.851 | 9.455 | 0.83x | 0.93x |
| 4 x 32769 | RMSNorm, vs torch.compile | 33.587 | 13.757 | 2.44x | |

The best chunk depends on the row count
([sweep](../results/row-chunk-sweep.log), `python scripts/row_chunk_sweep.py`; all
choices for a shape are sampled in shuffled rounds):

| Softmax shape | PyTorch | 2048-wide chunks, 4 warps | 8192-wide chunks, 16 warps |
| --- | ---: | ---: | ---: |
| 1024 x 16384 | 91.55 | 34.92 | 48.79 |
| 64 x 131072 | 25.96 | 37.43 | 21.35 |
| 4 x 32769 | 8.69 | 17.10 | 10.85 |

Observations:

- With many rows, small chunks keep many programs resident; with few rows, each
  program must cover more of its row at once. The launcher picks 2048/4 warps when
  there are at least two rows per SM (160 on this GPU) and 8192/16 warps otherwise.
  In this sweep that choice was the best or within 7% of the best for every shape
  and operation (RMSNorm at 4 x 32769 is the 7% case). The sweep's `auto` column runs the same kernel
  and configuration as one of the forced columns, so the gap between those two
  measures this run's noise: up to 32% for RMSNorm at 1024 x 16384 (209.28 vs
  158.16 us).
- With only four rows, four programs cannot fill 80 SMs, and softmax loses to
  PyTorch (0.83x). Splitting each row across programs fixes that: workshop A2's
  three-kernel split softmax ran 4 x 32769 about 2.2x faster than PyTorch
  ([workshop results](../results/workshops/VALIDATION.md)). Routing few-row cases
  to a split path is the natural next step.
- Softmax at 64 x 131072 wins clearly warm (1.15x); cold it measured 1.04x here
  and 0.94-1.08x in other runs. The second pass rereads each row, which is cheap from
  L2 but not from DRAM.
- `torch.compile` handled these widths poorly: RMSNorm at 64 x 131072 took
  151.245 us compiled against 26.897 us for the looped kernel, and FP32 softmax took
  185.822 us compiled against 46.251 us eager
  ([softmax source](../results/rtx4080super-softmax-compiled.json)). The cause was
  not investigated; inspecting Inductor's generated kernels is the next step.

## torch.compile: pay dispatch once per graph

`kernel_portfolio.library` registers every operator with `torch.library.triton_op`,
so `torch.compile` can place them in one graph and, with `mode="reduce-overhead"`,
replay the graph as a CUDA graph. Host microseconds per call, variants interleaved
in one process ([log](../results/compile-overhead.log),
`python scripts/compile_overhead.py`):

| Workload | Eager `ops.*` | Custom ops, torch.compile | Custom ops, reduce-overhead |
| --- | ---: | ---: | ---: |
| One softmax, 32 x 127 | 26.24 | 40.62 | 54.88 |
| 16 ops: 8 x (residual RMSNorm + softmax) | 533.59 | 202.91 | 69.52 |

Observations:

- For one tiny operator, compiling does not pay: guard checks and graph bookkeeping
  cost more than the launch they replace. Calling through `torch.ops` eagerly also
  adds dispatcher cost (51.37 us), so the plain `kernel_portfolio.*` functions stay
  the eager API.
- Across a 16-op chain, reduce-overhead cut host time per call 7.7x, from 533.59 to
  69.52 us: one graph replay instead of 16 Python launches. That is where the eager
  wrapper's per-call cost, measured below, actually goes away.
- `torch.library.opcheck` validates each registration, and GPU tests check that
  compiled graphs, including dynamic row counts, match eager results.

## Softmax: fusion helps device time, dispatch can erase the gain

Hypothesis: keeping a row's max, exponentials and sum in one program removes
intermediates and launch overhead compared with an eager decomposition. Compare
against `torch.softmax` as well, since it is already a specialized operator.

FP16, CUDA graph replay, warm L2; [source JSON](../results/rtx4080super-triton-fp16.json):

| Shape | torch.softmax | Triton 4 warps | Triton 8 warps | Best Triton speedup |
| --- | ---: | ---: | ---: | ---: |
| 32 x 127 | 1.502 | 1.126 | 1.146 | 1.33x |
| 1024 x 1024 | 4.198 | 2.389 | 3.311 | 1.76x |
| 512 x 4097 | 12.730 | 5.050 | 5.598 | 2.52x |
| 16384 x 4096 | 528.009 | 514.901 | 493.909 | 1.07x |

Observation: four warps were best at widths 1024 and 4097; at 32 x 127 and at the
largest shape the two differ by less than this machine's run-to-run spread. The
all-runtime version preferred eight warps at 4097, because its extra registers per
thread made four warps worse; a warp-count conclusion depends on the rest of the
kernel. The inspected 1024-wide softmax reports 23 registers/thread, zero spills
and 16 bytes of shared memory.

An ordinary event-timed wrapper run reverses the mid-size result: for FP16
1024x1024, PyTorch took **11.366 us** and the 4-warp wrapper **33.835 us**
([source](../results/rtx4080super-softmax-events.json)). A host-side breakdown
([log](../results/wrapper-overhead.log), `scripts/wrapper_overhead.py`) puts
Triton's own Python launcher at 16.0 us of the 28.6 us call. Input validation,
output allocation and the device guard add about 5 us together. At 16384 x 4096
the kernel is long enough to hide dispatch, and the wrapper wins
(486.127 vs 516.119 us). Eager calls to a short kernel lose; the compiled
graph above is how to remove that cost. Event timings include host work, so they
move with CPU load more than graph timings do.

The separate FP32 compiled comparison keeps semantics and dtype matched
([source](../results/rtx4080super-softmax-compiled.json)). At 1024x1024, eager
PyTorch took **4.267 us**, compiled PyTorch **5.905 us** and Triton 4 warps
**3.367 us**. At 16384x4096 all three were within 4% (979.213, 991.631 and
1011.746 us). Inductor again warned that it split a reduction and disabled online
softmax for a case in this sweep. Without inspecting each generated kernel, that
warning is not a complete causal explanation of the timing.

## Residual RMSNorm: compare with a compiler, not just eager composition

Hypothesis: fusing FP32 residual addition, the row reduction and scaling avoids
writing/reading a full intermediate activation. The exact FP32 residual contract
is shared by all variants.

FP16, CUDA graph replay, warm L2; [source JSON](../results/rtx4080super-rmsnorm-compiled.json):

| Shape | PyTorch composition | `F.rms_norm` | torch.compile | Triton | Compiled / Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 x 127 | 12.663 | 9.045 | 1.045 | 1.024 | 1.02x |
| 1024 x 1024 | 31.778 | 21.562 | 5.120 | 2.765 | 1.85x |
| 512 x 4097 | 53.692 | 45.943 | 12.390 | 7.430 | 1.67x |
| 16384 x 4096 | 6824.926 | 4514.129 | 706.556 | 731.170 | 0.97x |
| 4 x 32769 | 16.896 | 36.045 | 33.587 | 13.757 | 2.44x |
| 64 x 131072 | 551.595 | 408.098 | 151.245 | 26.897 | 5.62x |

Observation: the 7-12x gain against the eager composition mostly counts launches.
The built-in `F.rms_norm` is 1.17-1.51x faster than that composition at most
shapes, but 2.1x slower at 4 x 32769, because the FP32 casts and residual
addition still run as separate kernels. Against `torch.compile`, Triton ties at
the smallest shape (1.02x), loses slightly at the DRAM-bound 16384 x 4096 (0.97x),
and wins at the mid-size and wide shapes. With L2 flushed, the 1024 x 1024 ratio
against the composition falls from 11.25x to 4.86x
([cold JSON](../results/rtx4080super-triton-fp16-cold.json)). The evidence supports
fusing casts, residual and normalization over eager PyTorch, and a looped kernel
for very wide rows. It does not show a hand-written kernel beating a compiler at
DRAM-bound sizes.

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
| 127 x 255 x 65 | 3.925 | 2.150 | 2.174 | 1.81x | 1.40x |
| 512 x 512 x 512 | 5.359 | 5.390 | 4.745 | 1.13x | 0.79x |
| 1024 x 1024 x 1024 | 26.351 | 28.501 | 26.032 | 1.01x | 1.12x |
| 4096 x 4096 x 4096 | 1570.299 | 2106.470 | 1529.433 | 1.03x | 1.05x |

At 4096 x 4096 x 4096 the autotuned kernel (128 x 128 x 32 tiles, 8 warps, 3 stages,
`GROUP_M=8`) reached 89.9 TFLOP/s against 87.5 for `torch.mm` warm, and 86.9
against 82.7 cold. Across five runs its lead ranged from 1.03x to 1.12x: small, but it
never fell behind ([stability log](../results/benchmark-stability.log)).
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

At 1,048,576 vector elements, PyTorch took **2.309 us**, Triton with block size
256 **2.799 us** and block size 1024 **2.287 us** warm. Cold, all three were
within 2% (15.292-15.565 us), so the warm differences are an L2-resident effect.
At 16,777,216 elements, which exceeds L2, the three measured 166.3-191.0 us here;
their ratios moved by up to 28% between runs, more than any difference between the
variants, so they are best read as tied.

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
