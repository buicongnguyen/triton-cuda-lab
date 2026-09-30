# Measured optimization case studies

Local runs: **2026-09-30 Asia/Seoul** (JSON timestamps use UTC), RTX 4080 SUPER,
Ubuntu WSL2, Python 3.12.13, PyTorch 2.11.0+cu130, Triton 3.6.0. Values are
medians in microseconds. See [all tables](../results/SUMMARY.md),
[raw evidence](../results/) and [methodology](BENCHMARKING.md). These results
replace earlier sets; the [review record](records/REVIEW.md) explains why.

The GPU was shared with desktop applications and other programs (a browser, Blender, an
editor, other test runs). Each report records how busy the GPU was just before its run:
3-40% for these runs. The benchmark spreads each case's samples over
the run so a burst spoils only a few ([methodology](BENCHMARKING.md)), and the final run
waited for a quiet moment before each stage. Load still matters. The
[stability log](../results/benchmark-stability.log) repeated the FP16 suite three times:
29 of 36 Triton/PyTorch comparisons agreed within 10% across the runs
(29 of 36 with L2 flushed, [log](../results/benchmark-stability-cold.log)).
The widest spreads belong to kernels of about a microsecond (row_sum 32 x 127: 123%;
add 257 triton_256: 53%). The
[earlier five-run log](../results/archive/benchmark-stability-2026-09-26.log) told the same story:
most softmax, RMSNorm and GEMM ratios agreed within 10%, while several row-sum and
cold-cache ratios varied by more than 30%. Treat a difference smaller than that spread as
unresolved. Every comparison below is within one run.

Absolute times move between runs of the same suite. PyTorch's own time on the five largest
cases was 9 to 17% shorter in this sweep (other GPU load 18%) than in the
[archived sweep](../results/archive/pre-warmup-fp16.json) taken earlier on the same source (load
10%), yet no Triton/PyTorch ratio on those cases moved by more than
0.24x. Ratios inside one run are the numbers to trust.

How to read the tables: 1 us (microsecond) is 1/1000 of a millisecond, and lower
is faster. A speedup of 1.75x means PyTorch took 1.75 times as long as Triton;
below 1.00x, Triton was slower. *Warm* means inputs were likely still in the 64 MiB
L2 cache from the previous call; *cold* means L2 was flushed first. More terms are
in the [glossary](../learning/GLOSSARY.md#measuring).

## GPU clocks: the first calls after an idle pause

An idle GPU lowers its clocks, and a kernel timed while they climb back looks slower than it
is. [gpu_clock_probe.py](../scripts/gpu_clock_probe.py) times a 2 MiB FP16 softmax, about
2.5 us at boost clocks, right after idle gaps of growing length. The GPU is
warmed before each gap; nvidia-smi's SM clock and utilization are read when the gap ends
([log](../results/gpu-clock-probe.log)). Microseconds per call, by time after the first call:

| Idle gap | SM clock at wake | Load | 0-100 ms | 100-400 ms | 400-800 ms | 800-1200 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 s | 2835 MHz | 6% | 2.92 | 2.92 | 2.75 | 2.75 |
| 5 s | 1410 MHz | 14% | 5.17 | 5.27 | 2.97 | 2.66 |
| 10 s | 2550 MHz | 3% | 8.40 | 2.92 | 2.65 | 2.45 |
| 20 s | 210 MHz | 20% | 25.34 | 2.71 | 2.45 | 2.45 |
| 40 s | 2550 MHz | 3% | 8.39 | 4.81 | 2.86 | 2.65 |

Observations:

- After 5 s or more of idleness the first 100 ms ran at least 1.9x
  slower than the steady state; the worst was 25.34 us against 2.45 us after 20 s,
  when nvidia-smi read the SM clock at 210 MHz. The slowdown lasted 100-800 ms and then
  went away by itself: small kernels do wake the GPU, but not at once. The size does not grow
  steadily with the gap, because the desktop's own work keeps waking the GPU in between; the
  2 s gap left the kernel at full speed.
- A benchmark that starts a case after a pause can therefore time the ramp. Without a warm-up,
  the first timing after a 10 s gap ranged from 2.66 to 122.93 us over six
  trials (medians 9.78 and 17.25 us). A 20-150 ms burst of large GEMMs, then a 500 ms pause, gave 2.92-2.95 us
  (worst trial 4.34 us). With no pause after the burst the first calls were still
  slow (6.40-11.32 us medians), so the benchmark warms the GPU, replays every
  variant once untimed and only then takes samples, and reports medians.
- `kernel-bench` therefore runs a 200 ms burst before each visit to a case unless a burst ended
  less than 250 ms earlier, and records the setting in the report's `warmup` field. The
  standalone CUDA program and the sweeps in `scripts/` do the same.
- It did not change the picture. Against the archived sweep from before the warm-up existed, the
  Triton/PyTorch ratio moved by less than 10% in 31 of 36 variants; the larger
  moves were at kernels of about a microsecond and where PyTorch's own time moved. A sweep
  keeps the GPU busy, so only the first case after a pause was at risk. The warm-up guards
  measurements that start after a pause, such as a single case run from a script; it is not
  what made the numbers steadier, and the stability logs above show that they are not steady
  yet.

## Specialize what the model fixes

Hypothesis: a size that is a compile-time constant (`tl.constexpr`) triggers a
compile for every new value, while a runtime size can cost speed. A model fixes
its row width and weight shapes, but batch size, token count and row count change
per request. Specializing only the fixed sizes should keep the speed without
compiling per batch.

Three versions of the same kernels, FP16, CUDA-graph replay, warm L2
([log](../results/specialization-sweep.log), reproduced by
`python scripts/specialization_sweep.py`). Brackets show softmax registers per thread.
The pre-review and runtime versions are loaded from [results/archive/](../results/archive/).
The sweep times the three versions one after another rather than interleaved, so a burst of
other GPU activity can spoil a single cell; compare rows, not single numbers.

| Shape | Kernel | Every size compile-time (pre-review) | Every size runtime | Current: width and GEMM N/K compile-time |
| --- | --- | ---: | ---: | ---: |
| 512 x 4097 | softmax | 5.12 [48] | 7.65 [128] | 4.95 [48] |
| 512 x 4104 | softmax | 3.89 [56] | 7.17 [128] | 4.12 [56] |
| 512 x 5120 | softmax | 4.70 [54] | 5.94 [85] | 4.51 [54] |
| 512 x 4097 | RMSNorm | 7.91 | 9.90 | 8.32 |
| 512 x 4104 | RMSNorm | 5.36 | 9.93 | 5.38 |
| 1024 x 1024 | softmax | 2.76 [25] | 2.76 [23] | 2.76 [23] |
| 512 x 3072 | softmax | 3.57 [40] | 3.72 [46] | 3.43 [39] |
| 127 x 255 x 65 | GEMM, fixed tile | 2.12 | 2.46 | 2.29 |
| 512 x 512 x 512 | GEMM, fixed tile | 5.56 | 6.08 | 5.56 |

Observations:

- With every size at runtime, widths that do not fill their power-of-two block ran up to
  1.9x slower (RMSNorm at 4104: 9.93 vs 5.36 us). The 1024-wide and 3072-wide
  softmax cases changed by less than 4% between versions.
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
  index arithmetic costs 8% at the tiny 127 x 255 x 65 shape (2.29 vs 2.12 us) and
  0% at 512 (5.56 vs 5.56 us).
- The learning exercises keep every size at runtime. That is always correct, and
  the numbers above show what the portfolio kernels gain by specializing further.

## Cache residency: warm numbers can exceed DRAM

The same current kernels, FP16 graph replay, warm and with L2 flushed before every
call ([warm](../results/rtx4080super-triton-fp16.json),
[cold](../results/rtx4080super-triton-fp16-cold.json)); ratios are `torch / Triton`:

| Case | Warm L2 | Cold L2 |
| --- | ---: | ---: |
| Softmax 1024 x 1024, 4 warps | 1.71x | 1.24x |
| Softmax 512 x 4097, 4 warps | 2.50x | 1.34x |
| Row sum 512 x 4097 | 1.64x | 1.28x |
| RMSNorm 512 x 4097, vs eager composition | 6.88x | 3.77x |
| GEMM 512 x 512 x 512, autotuned | 1.12x | 0.81x |

- Warm, the 1024-wide softmax moved 1,640 GB/s of logical traffic, more than the
  ~736 GB/s DRAM can deliver. Cold, it moved 427 GB/s. Warm results describe
  cache-resident repeated calls, not saved DRAM traffic.
- Cold timing has a floor of about 3.7 us per call (a 257-element add takes
  3.721 us), so small cold cases are dominated by that overhead.
- Repeat runs disagreed by more than 10% on 7 of 36 warm and 7 of 36 cold
  comparisons ([warm](../results/benchmark-stability.log), [cold](../results/benchmark-stability-cold.log)); the
  kernels of about a microsecond moved most in both. Warm and cold agree that the 1024-wide and
  4097-wide softmax win and that fusion wins most for RMSNorm.
- At 16384 x 4096, which exceeds L2 even when warm, softmax measured
  1.01x with four warps and 1.02x with eight, and row sum 0.96x; the row sum
  ranged 1.00-1.08x across the three runs of the stability log and 0.82-1.32x across five
  runs on 2026-09-26 ([archived](../results/archive/benchmark-stability-2026-09-26.log)). Both operators run near the bandwidth
  other GPU users leave, so there is little left to win at that size.

## Wide rows: loop in one program, or split across programs

Rows wider than 8192 no longer fit one block. `_row_plan` in
[triton_kernels.py](../src/kernel_portfolio/triton_kernels.py) picks one of two
ways to cover them:

- **Looped**: one program per row walks the row in chunks. Softmax keeps an online
  maximum and sum per lane (workshop A1's merge rule), then rereads the row to write
  probabilities; RMSNorm sums squares, then rereads to scale. The chunk is 8192 wide
  with 16 warps, except for softmax and RMSNorm rows at most 28672 wide once there are as
  many rows as SMs: those use 4096-wide chunks with 8 warps.
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
| 4 x 32769 | row sum (looped) | 2.901 | 2.662 | 1.09x | 1.02x |
| 4 x 32769 | softmax | 8.011 | 2.799 | 2.86x | 1.78x |
| 4 x 32769 | RMSNorm, vs torch.compile | 33.651 | 2.970 | 11.33x | |
| 64 x 131072 | row sum (looped) | 18.159 | 5.973 | 3.04x | 1.49x |
| 64 x 131072 | softmax | 24.883 | 17.476 | 1.42x | 0.97x |
| 64 x 131072 | RMSNorm, vs torch.compile | 133.622 | 27.536 | 4.85x | |

Before the split path, the looped softmax lost at 4 x 32769 (0.83x warm): four
programs cannot fill 80 SMs. The sweep behind that first choice
([log](../results/row-chunk-sweep.log), `python scripts/row_chunk_sweep.py`) forces
each plan; all choices for a shape are sampled in shuffled rounds:

| Shape | Operation | PyTorch | Looped 2048/4w | Looped 4096/8w | Looped 8192/16w | Split 2048/4w | Split 4096/8w | Automatic |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 x 32769 | softmax | 7.99 | 15.40 | 10.70 | 9.71 | 3.17 | 3.23 | 3.07 |
| 4 x 32769 | RMSNorm | - | 16.68 | 13.20 | 14.12 | 3.02 | 3.42 | 3.01 |
| 64 x 131072 | softmax | 23.50 | 34.34 | 21.70 | 19.30 | 17.15 | 17.09 | 17.10 |
| 64 x 131072 | RMSNorm | - | 37.94 | 27.65 | 25.86 | 27.75 | 26.52 | 26.52 |
| 128 x 32769 | softmax | 16.33 | 19.60 | 18.43 | 20.58 | 15.46 | 16.02 | 20.17 |
| 128 x 32769 | RMSNorm | - | 24.83 | 24.78 | 27.80 | 18.74 | 19.51 | 27.80 |
| 1024 x 16384 | softmax | 48.13 | 35.42 | 40.65 | 53.66 | 66.76 | 51.20 | 47.56 |
| 1024 x 16384 | RMSNorm | - | 155.65 | 150.73 | 152.12 | 272.74 | 235.42 | 151.53 |

Observations:

- Splitting pays when rows are few and wide: at 4 x 32769 it cut softmax from
  9.71 us (best looped choice) to 3.17 us, and RMSNorm from 13.20 to 3.02 us.
  At 1024 x 16384, with plenty of rows, looping wins, and splitting RMSNorm would
  take 1.6x as long.
- At 128 x 32769, above the SM count, the split path was faster
  (15.46 vs 18.43 us for softmax) and the rule still loops. The grid below explains why the
  simple rule stays.
- At 64 x 131072, splitting helped softmax (17.09 vs 19.30 us) but not
  RMSNorm (26.52 vs 25.86 us looped): within noise, the rule's choice there
  neither gains nor loses much.
- The automatic column runs the same kernel and configuration as one of the forced
  columns, so the gap between those two measures this run's noise: up to
  17% (softmax at 1024 x 16384).
- `torch.compile` handles these widths poorly: RMSNorm at 64 x 131072 took
  133.622 us compiled against 27.536 us here, and FP32 softmax took
  157.218 us compiled against 45.773 us eager
  ([softmax source](../results/rtx4080super-softmax-compiled.json)). The generated code
  ([log](../results/inductor-wide-rows.log), `python scripts/inspect_inductor.py`)
  shows why. For RMSNorm, Inductor launches 3 kernels: it splits each row into
  only four pieces (256 programs, each looping over 32768 elements), merges them,
  then reads the inputs again in a separate pointwise kernel. For softmax it
  launches 5, because it disables online softmax when it splits the reduction:
  one split pass for the maximum, another for the sum, and a third read to write.
  At 1024 x 1024 both compile to a single kernel.

### Choosing the plans from a grid

Four shapes are too few to fit a rule to. `python scripts/row_plan_grid.py --full`
([log](../results/row-plan-grid.log)) times every plan (looped with 8192-, 4096-, 2048- and
1024-wide chunks, and both split plans) for softmax and RMSNorm at 18 row counts and 10 widths
from 12288 to 1,048,576 (332 cells), and scores a rule by its regret: the rule's time over the
best plan's time in the same cell, minus 1. A repeat run ([log](../results/row-plan-grid-repeat.log))
differed from the first by 2% in the median plan-and-cell and by
16% at the 90th percentile, so a difference of a few points in a mean is noise.

| Rule | Mean regret, run 1 / run 2 | 90th percentile | Cells over 10% (run 1) |
| --- | ---: | ---: | ---: |
| Current, both operators | 5.4% / 5.2% | 18.5% | 51 of 332 |
| Previous, both operators | 12.3% / 12.8% | 42.8% | 102 of 332 |
| No narrow rule, both operators | 7.0% / 7.2% | 23.1% | 71 of 332 |
| Current, softmax | 5.4% / 4.9% | 21.0% | 26 of 166 |
| Previous, softmax | 11.1% / 12.0% | 34.3% | 58 of 166 |
| No narrow rule, softmax | 8.2% / 8.6% | 27.6% | 45 of 166 |
| Current, RMSNorm | 5.5% / 5.5% | 18.5% | 25 of 166 |
| Previous, RMSNorm | 13.5% / 13.5% | 57.9% | 44 of 166 |
| No narrow rule, RMSNorm | 5.8% / 5.8% | 21.3% | 26 of 166 |

- The previous rule used 2048-wide chunks from two rows per SM. That was the flaw: it
  gave 2.3 times the mean regret and left 102 of 332 cells more than 10% off the best plan,
  against 51 for the current rule. The two rules choose different plans in 156 cells;
  the current plan is at least 10% faster in 62 of them and at least 10% slower in 12.
- The narrow-row rule (4096-wide chunks) is worth 2.8 points of mean regret for softmax and
  0.3 for RMSNorm: it is a softmax optimization that RMSNorm barely uses.
- Split against loop by row count (median of the best split time over the best looped time;
  in brackets, the cells where splitting wins, is within 5%, or loses; widths above 8192 for
  softmax and above 16384 for RMSNorm):

| Rows | Softmax | RMSNorm |
| ---: | ---: | ---: |
| 1 | 0.40 (10 / 0 / 0) | 0.25 (7 / 1 / 0) |
| 4 | 0.43 (10 / 0 / 0) | 0.32 (8 / 0 / 0) |
| 16 | 0.56 (9 / 0 / 1) | 0.66 (8 / 0 / 0) |
| 32 | 0.81 (7 / 2 / 1) | 0.93 (4 / 3 / 1) |
| 64 | 0.99 (3 / 3 / 4) | 1.09 (1 / 3 / 4) |
| 80 | 1.08 (2 / 2 / 6) | 1.10 (1 / 2 / 5) |
| 96 | 0.88 (6 / 2 / 2) | 1.05 (1 / 3 / 4) |
| 128 | 1.02 (2 / 5 / 3) | 1.05 (1 / 3 / 4) |
| 256 | 1.04 (1 / 5 / 3) | 1.13 (0 / 2 / 5) |
| 512 | 1.16 (0 / 1 / 8) | 1.53 (0 / 2 / 5) |
| 1024 | 1.44 (0 / 0 / 8) | 1.61 (0 / 0 / 6) |

  Splitting wins clearly below about 32 rows. RMSNorm loops faster from about 64 rows. For
  softmax the two are within a few percent from 64 to 256 rows (splitting even wins at 96,
  just above the SM count, which the plain "fewer rows than SMs" rule misses) and looping wins
  clearly from 512.
- **Negative result.** `python scripts/row_plan_fit.py` ([log](../results/row-plan-fit.log))
  fits thresholds on one run of the grid and scores them on the other: split thresholds of
  100-160 rows for softmax and 40 for RMSNorm, and a narrow-chunk limit of
  28672-32000 for softmax. On the run they had not seen the fitted rules changed the mean
  regret by -0.2 to +0.3 points, no more than the two runs differ from each other for the
  current rule (up to 0.5 points). The rule stays simple. Regret by row count and width is irregular (the best plan flips between
  50257 and 65536 for RMSNorm at 256 rows), so no threshold explains it.
- The grid times forward softmax and RMSNorm. Row sum and the backward kernels inherit the rule
  without being measured separately.

## Training: backward kernels through autograd

Every operator trains. Softmax, residual RMSNorm and GEMM have Triton backward kernels
that follow the same row plans and tile schedules as the forward kernels; `add` passes its
gradient to both inputs and `row_sum` broadcasts it. `library.py` registers them all with
autograd, and the `ops.*` functions route gradient-tracking inputs through the custom ops.

- Softmax saves its output and computes `y * (dy - sum(y * dy))`.
- The RMSNorm backward recomputes each row's inverse RMS instead of saving it and writes
  the shared input gradient for `x` and `residual`. The weight gradient sums over rows in
  a separate kernel: programs own blocks of columns and groups of rows and write FP32
  partial sums, and a finish kernel adds the groups in a fixed order and rounds once, so the
  result is deterministic without atomics. A training step is four kernels (forward, input
  gradient, weight partials, finish); the first version needed five, because PyTorch's sum and
  cast ran as two more launches.
- The GEMM backward is two more products with transposed operands: `dA = dC B^T` and
  `dB = A^T dC`. One strided kernel serves both: a transposed operand is the same memory read
  with its strides swapped, never a copy. Tokens vary per request and stay a runtime
  argument; the model dimensions are compile-time. Each product autotunes on its own
  buckets, only the gradients that are needed are computed, and there are no atomics.

A training step here is one forward and one backward pass through autograd, which
is also the only way a CUDA graph can capture a backward pass. FP16 graph replay
([source](../results/rtx4080super-training-fp16.json)); ratios are PyTorch eager
over Triton:

| Shape | Operation | PyTorch | `F.rms_norm` | torch.compile | Triton | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32 x 127 | softmax | 3.782 |  | 2.219 | 2.219 | 1.70x |
| 1024 x 1024 | softmax | 10.001 |  | 10.612 | 5.382 | 1.86x |
| 512 x 4097 | softmax | 23.893 |  | 23.211 | 11.871 | 2.01x |
| 16384 x 4096 | softmax | 1850.334 |  | 1186.543 | 1161.216 | 1.59x |
| 4 x 32769 | softmax | 15.257 |  | 71.574 | 5.769 | 2.64x |
| 64 x 131072 | softmax | 117.827 |  | 373.043 | 53.419 | 2.21x |
| 32 x 127 | RMSNorm | 33.241 | 16.828 | 3.823 | 4.301 | 7.73x |
| 1024 x 1024 | RMSNorm | 89.461 | 43.552 | 23.006 | 15.258 | 5.86x |
| 512 x 4097 | RMSNorm | 186.368 | 88.883 | 50.313 | 26.010 | 7.17x |
| 16384 x 4096 | RMSNorm | 19226.282 | 8772.881 | 2525.867 | 2209.382 | 8.70x |
| 4 x 32769 | RMSNorm | 49.732 | 93.491 | 169.165 | 10.131 | 4.91x |
| 64 x 131072 | RMSNorm | 1743.599 | 973.311 | 449.434 | 148.681 | 11.73x |

GEMM training steps (M x N x K; TFLOP/s counts the three products, `6*M*N*K`; the last shape
has many tokens and a small weight, so `dB` is a long reduction):

| M x N x K | PyTorch (cuBLAS) | torch.compile | Triton | Speedup | TFLOP/s, Triton vs PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 127 x 255 x 65 | 10.035 | 9.762 | 10.581 | 0.95x | 1.2 vs 1.3 |
| 512 x 512 x 512 | 16.412 | 16.657 | 15.994 | 1.03x | 50.4 vs 49.1 |
| 1024 x 1024 x 1024 | 88.576 | 84.924 | 96.051 | 0.92x | 67.1 vs 72.7 |
| 4096 x 4096 x 4096 | 4358.202 | 4389.513 | 4295.202 | 1.01x | 96.0 vs 94.6 |
| 16384 x 1024 x 1024 | 1139.533 | 1148.314 | 1172.378 | 0.97x | 87.9 vs 90.5 |

Observations:

- Triton's maximum gradient error against the FP64 gradient was equal to or below
  PyTorch's own in 16 of 17 cases; the rest differed by one FP16 rounding
  step (softmax 16384 x 4096: 6.1e-05 against 3.1e-05) ([JSON](../results/rtx4080super-training-fp16.json)).
  [tolerance-probe.log](../results/tolerance-probe.log) shows how close each gradient test
  runs to its tolerance over 300 random inputs.
- Softmax steps ran 1.59-2.64x faster than PyTorch's; RMSNorm steps
  4.91-11.73x faster than the eager composition and 2.85-9.23x faster than `F.rms_norm`.
- Against `torch.compile`, which fuses forward and backward with AOTAutograd: softmax is
  the same speed (1.00x) at 32 x 127 and within 2% (1.02x) at the DRAM-bound
  16384 x 4096, 1.97x faster at 1024 x 1024, and 6.98x faster at the widest
  rows. RMSNorm is 1.12x slower at 32 x 127 (3.823 vs 4.301 us),
  where launches dominate, 1.14x faster at 16384 x 4096 and 3.02x faster at the widest rows.
  The finish kernel took the 32 x 127 step from 6.076 us ([earlier sweep](../results/archive/pre-warmup-training-fp16.json))
  to 4.301 us.
- The GEMM step is within 10% of cuBLAS at 5 of 5 shapes, from 0.92x at
  1024 x 1024 x 1024 to 1.03x at 512 x 512 x 512; `torch.compile` of `torch.mm` is no different, since it calls
  the same library. Three repeated runs ([log](../results/benchmark-stability-training.log)) kept every
  ratio between 0.93x and 1.09x. Triton reaches
  96.0 TFLOP/s at 4096^3 against 94.6 for cuBLAS.
- Steps of a few microseconds are the noisy ones: 12 of 17 training ratios agreed within 10% across those three
  runs, and the widest spread was the 32 x 127 softmax step (84%).
- Eager training through the custom ops is host-bound for small shapes. With event
  timing ([source](../results/rtx4080super-training-events.json)), a 1024 x 1024
  softmax step took 830.845 us against 373.543 us for PyTorch, although its
  kernels run faster: two Python custom ops and the autograd bookkeeping cost more
  than PyTorch's C++ path, and the same holds for a 1024^3 GEMM step (1128.709 vs 458.859 us).
  At 16384 x 4096 the kernels dominate and Triton wins
  (2210.304 vs 18027.144 us for RMSNorm). Compile the model, as in the next section,
  to remove the per-call cost.
- [examples/train_tiny.py](../examples/train_tiny.py) trains a small classifier that uses all
  five operators beside a plain-PyTorch twin from the same weights and batches
  ([log](../results/train-tiny.log), [compiled](../results/train-tiny-compiled.log)). The loss falls from
  2.6064 to 0.3957 against 0.3952 for the twin over 120 steps (0.3952 with the whole step compiled), and the
  test suite asserts the two curves stay within 0.5% at every step.

## torch.compile: pay dispatch once per graph

`kernel_portfolio.library` registers every operator with `torch.library.triton_op`,
so `torch.compile` can place them in one graph and, with `mode="reduce-overhead"`,
replay the graph as a CUDA graph. Host microseconds per call, variants interleaved
in one process ([log](../results/compile-overhead.log),
`python scripts/compile_overhead.py`):

| Workload | Eager `ops.*` | Custom ops, torch.compile | Custom ops, reduce-overhead |
| --- | ---: | ---: | ---: |
| One softmax, 32 x 127 | 42.80 | 64.44 | 113.61 |
| 16 ops: 8 x (residual RMSNorm + softmax) | 1045.06 | 378.00 | 145.25 |

Observations:

- For one tiny operator, compiling does not pay: guard checks and graph bookkeeping
  cost more than the launch they replace. Calling through `torch.ops` eagerly also
  adds dispatcher cost (84.16 us), so the plain `kernel_portfolio.*` functions stay
  the eager API.
- Across a 16-op chain, reduce-overhead cut host time per call 7.2x, from 1045.06 to
  145.25 us: one graph replay instead of 16 Python launches. That is where the eager
  wrapper's per-call cost, measured below, actually goes away.
- Host times move with CPU load, more than device times do: these were measured while other
  programs shared the CPU.
- `torch.library.opcheck` validates each registration, including the autograd
  registrations of every operator, and GPU tests check that compiled graphs,
  including dynamic row counts and a training step through all five operators, match
  eager results.

## Softmax: fusion helps device time, dispatch can erase the gain

Hypothesis: keeping a row's max, exponentials and sum in one program removes
intermediates and launch overhead compared with an eager decomposition. Compare
against `torch.softmax` as well, since it is already a specialized operator.

FP16, CUDA graph replay, warm L2; [source JSON](../results/rtx4080super-triton-fp16.json):

| Shape | torch.softmax | Triton 4 warps | Triton 8 warps | Best Triton speedup |
| --- | ---: | ---: | ---: | ---: |
| 32 x 127 | 1.638 | 1.259 | 1.263 | 1.30x |
| 1024 x 1024 | 4.368 | 2.558 | 3.447 | 1.71x |
| 512 x 4097 | 12.902 | 5.154 | 5.790 | 2.50x |
| 16384 x 4096 | 459.196 | 456.487 | 450.354 | 1.02x |

Observation: four warps were best at widths 1024 and 4097; at 32 x 127 and at the
largest shape the two differ by 0% and 1%, within the run-to-run spread of the
[stability log](../results/benchmark-stability.log) (14% and 1%). The
all-runtime version preferred eight warps at 4097, because its extra registers per
thread made four warps worse; a warp-count conclusion depends on the rest of the
kernel. The inspected 1024-wide softmax reports 23 registers/thread, zero spills
and 16 bytes of shared memory.

An ordinary event-timed wrapper run reverses the mid-size result: for FP16
1024x1024, PyTorch took **17.583 us** and the 4-warp wrapper **64.098 us**
([source](../results/rtx4080super-softmax-events.json)). A host-side breakdown
([log](../results/wrapper-overhead.log), `scripts/wrapper_overhead.py`; medians of interleaved
rounds) measures Triton's own Python launcher alone at 30.1 us, input validation at 4.0,
output allocation at 6.5 and the device guard at 0.8 us, against 57.4 us for the whole
`ops.softmax` call and 16.8 us for `torch.softmax`. At 16384 x 4096
the kernel is long enough to hide dispatch
(449.195 vs 461.722 us). Eager calls to a short kernel lose; the compiled
graph above is how to remove that cost. Event timings include host work, so they
move with CPU load more than graph timings do.

The separate FP32 compiled comparison keeps semantics and dtype matched
([source](../results/rtx4080super-softmax-compiled.json)). At 1024x1024, eager
PyTorch took **4.403 us**, compiled PyTorch **6.075 us** and Triton 4 warps
**3.550 us**. At 16384x4096 all three were within 1% (895.958, 893.406 and
888.512 us). Inductor again warned that it split a reduction and disabled online
softmax for the wide shapes; the wide-rows section above shows the kernels it emits.

## Residual RMSNorm: compare with a compiler, not just eager composition

Hypothesis: fusing FP32 residual addition, the row reduction and scaling avoids
writing/reading a full intermediate activation. The exact FP32 residual contract
is shared by all variants.

FP16, CUDA graph replay, warm L2; [source JSON](../results/rtx4080super-rmsnorm-compiled.json):

| Shape | PyTorch composition | `F.rms_norm` | torch.compile | Triton | Compiled / Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 x 127 | 12.800 | 9.182 | 2.059 | 1.638 | 1.26x |
| 1024 x 1024 | 32.251 | 22.016 | 5.252 | 2.958 | 1.78x |
| 512 x 4097 | 54.101 | 46.114 | 12.595 | 7.873 | 1.60x |
| 16384 x 4096 | 6204.348 | 4153.685 | 626.790 | 643.652 | 0.97x |
| 4 x 32769 | 16.950 | 36.044 | 33.651 | 2.970 | 11.33x |
| 64 x 131072 | 462.916 | 395.659 | 133.622 | 27.536 | 4.85x |

Observation: the 7-11x gain against the eager composition mostly counts launches.
The built-in `F.rms_norm` is 1.17-1.49x faster than that composition at most
shapes, but 2.1x slower at 4 x 32769, because the FP32 casts and residual
addition still run as separate kernels. Against `torch.compile`, the two are a single launch of a
microsecond or two at the smallest shape, where this machine's noise is as large as the difference
(1.26x here); Triton is 1.78x faster at 1024 x 1024,
within 3% (0.97x) at the DRAM-bound 16384 x 4096 and 4.85x faster at the widest
rows, by the most where the split path runs. With L2
flushed, the 1024 x 1024 ratio against the composition falls from 11.05x to
4.30x ([cold JSON](../results/rtx4080super-triton-fp16-cold.json)). The evidence supports
fusing casts, residual and normalization over eager PyTorch, and dedicated wide-row
plans. It does not show a hand-written kernel beating a compiler at DRAM-bound
sizes or at the smallest shape.

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
| 127 x 255 x 65 | 4.062 | 2.287 | 2.309 | 1.76x | 1.27x |
| 512 x 512 x 512 | 5.526 | 5.557 | 4.915 | 1.12x | 0.81x |
| 1024 x 1024 x 1024 | 26.624 | 28.672 | 26.247 | 1.01x | 1.33x |
| 4096 x 4096 x 4096 | 1445.274 | 1821.969 | 1337.105 | 1.08x | 1.07x |

At 4096 x 4096 x 4096 the autotuned kernel (BM: 128, BN: 128, BK: 32, GROUP_M: 8, num_warps: 8) reached
102.8 TFLOP/s against 95.1 for `torch.mm` warm, and 101.7 against 94.8 cold.
Its ratio ranged 1.08x to 1.10x across the three runs of the
[stability log](../results/benchmark-stability.log) and 1.03x to 1.12x across five earlier runs
([archived](../results/archive/benchmark-stability-2026-09-26.log)).
Isolating the grouping with that same tile ([log](../results/gemm-grouping.log), `python scripts/gemm_grouping.py`):
`GROUP_M` = 1, 4 and 8 ran within 5% of each
other at 1024, 2048 and 4096 (4096^3: 1322.5, 1329.8, 1310.5 us). The gain over
`torch.mm` came from the larger tiles the new search offers, not from grouped
order. This GPU's 64 MiB L2 already holds all of B (32 MB at 4096^3 in FP16), so
grouping has little reuse left to recover; a GPU with a smaller L2 could differ.
Two cautions. `torch.mm` runs with reduced-precision reductions disabled so both
sides accumulate in FP32 (see [methodology](BENCHMARKING.md)); PyTorch's default
may pick different kernels. And this is one GPU and one shape family, not a claim
of beating cuBLAS in general. The fixed kernel's saved PTX contains `mma.sync`
instructions and reports 55 registers/thread, zero spills and 6144 bytes of shared
memory.

At 1,048,576 vector elements, PyTorch took **2.410 us**, Triton with block size
256 **2.901 us** and block size 1024 **2.389 us** warm. Cold, all three were
within 2% (15.155-15.531 us), so the warm differences are an L2-resident effect.
At 16,777,216 elements, which exceeds L2, the three measured 153.7-157.9 us here;
their ratios moved by up to 3% between the runs of the stability log and by up to 28% between runs on
2026-09-26 ([archived](../results/archive/benchmark-stability-2026-09-26.log)), more than any difference
between the variants, so they are best read as tied.

## CUDA: parallel reduction, online statistics and a numerical fix

The native Windows CUDA program times four 1024x1024 FP32 softmax kernels, sampled in
rotating order so they share the same conditions, after a 250 ms compute warm-up
([source](../results/rtx4080super-cuda.json) is the run with the middle parallel-kernel median of
7; [all runs](../results/cuda-softmax-runs.log), `python scripts/cuda_runs.py`):

| Kernel | What it does | Median (us) |
| --- | --- | ---: |
| `softmax_serial` | One thread per row; the intentionally weak baseline | 418.550 |
| `softmax_parallel` | One 256-thread block per row; max, sum and write passes (three reads) | 12.361 |
| `softmax_online_scalar` | Online (max, sum) in one pass, then write (two reads) | 9.165 |
| `softmax_online` | The same with 16-byte `float4` loads | 11.162 |

The GPU was shared with other programs (load 2-9% just before each run) and the fast
kernels moved a lot from run to run: the parallel kernel's median ranged 8.7-20.1 us across the 7
runs, and the `float4` and scalar online kernels 8.8-16.6 and 9.0-12.2 us. Only the
gap to the serial baseline (418-431 us, at least 21x any other kernel in every run) is
dependable. The ordering of the parallel and online kernels changed between runs (parallel over online
0.99x to 1.32x), so neither the removed read nor wider loads is a
demonstrated gain at this size: the 4 MB input stays in L2 between passes, so the extra read is cheap.
An earlier single run, with kernels timed one after another, showed `float4` 1.11x faster and credited
wider loads; with interleaved sampling that difference did not hold up. Compare the cold-L2 Triton
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
