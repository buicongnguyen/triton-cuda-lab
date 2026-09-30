# Benchmark methodology

Terms such as *CUDA graph*, *L2*, *warm/cold cache* and *logical GB/s* are defined in
the [glossary](../learning/GLOSSARY.md#measuring). Run the commands below in the GPU
environment, PowerShell or WSL ([every new terminal](SETUP.md#every-new-terminal)).
Compare timings only within one environment: the recorded results came from WSL.

Run from a quiet machine. Close GPU-heavy applications if you want a clean repeat;
the saved local results came from a desktop GPU under WSL/Windows and can vary. The
benchmark warms the GPU itself (see *GPU clocks* below) but cannot keep other programs
from taking time slices.

```bash
kernel-bench --op all --dtype float16 --output results/local/all.json
# Same sweep with L2 flushed before every timed call: inputs come from DRAM.
kernel-bench --op all --dtype float16 --cache cold --output results/local/all-cold.json
kernel-bench --op softmax --shape 1024 1024 --dtype float32 --compile --output results/local/softmax-compiled.json
kernel-bench --op rmsnorm --compile --output results/local/norm-compiled.json
kernel-bench --op matmul --dtype bfloat16 --output results/local/gemm-bf16.json
# Separate dispatch-sensitive measurement; do not mix it with graph numbers.
kernel-bench --op softmax --timing events --output results/local/softmax-events.json
# Forward + backward through autograd: softmax, residual RMSNorm and GEMM.
kernel-bench --op training --compile --output results/local/training.json
kernel-bench --op matmul_train --shape 16384 1024 1024 --output results/local/gemm-train.json
```

Every variant passes an independent double-precision oracle before timing. Random
inputs and variant order use a recorded seed. Add, reductions and softmax compare
to PyTorch's built-in operator, not just an intentionally slow Python expression.
RMSNorm's exact FP32 residual semantics use a PyTorch composition of about ten
eager kernels, so a large ratio against it mostly counts launches.
`torch_rms_norm` keeps the same FP32 residual semantics but uses the built-in
`F.rms_norm`. `--compile` adds Inductor's fused version and is necessary before
claiming a manual-fusion advantage over a compiler. GEMM compares to `torch.mm` with reduced-precision
reduction disabled. Input/output conversion and tolerances are recorded by code.

The training suites (`--op training`, or `softmax_train`, `rmsnorm_train` and
`matmul_train` alone) time one forward and one backward pass through autograd per call
and check the input gradient (softmax, RMSNorm) or both operand gradients (GEMM, whose
shapes are `M N K`) against an FP64 gradient. The whole step is timed because a CUDA
graph can only replay a backward pass whose forward was captured with it. They run
separately from `--op all`: every prepared case stays in GPU memory until sampling
ends, and training cases also keep activations for backward. With `--compile`, the
PyTorch forward is compiled and its backward comes from AOTAutograd.

The default captures 30 invocations per CUDA graph after correctness, JIT,
autotuning and five warmup calls. Each of nine samples times a graph replay using
CUDA events and divides by 30. All cases are prepared first, then each case is
visited three times (`--visits`) in a random order across the run. A visit replays
each variant once untimed, then takes three shuffled rounds of the case's variants
back to back, so a ratio compares measurements taken moments apart. On a GPU shared
with desktop applications, a burst of activity then spoils one visit of one case,
which the median of nine samples discards. The recorded sweeps use `--samples 15
--visits 5`, so a burst has to spoil three of five visits, not two of three, to move a
case's median. Two simpler designs failed on this
machine: timing each case in one block let a burst slow a whole case 2-3x, and
interleaving every sample of every case separated a case's variants in time and
made tiny kernels pay to reload their graphs. These numbers
describe repeated device execution with warm caches and reused graph addresses.
They exclude Python dispatch and capture cost; graph memory pools retain captured
allocations. They are not application request latency or cold-cache bandwidth.

**Cache residency.** The RTX 4080 SUPER has a 64 MiB L2 and roughly 736 GB/s of
DRAM bandwidth. With the default `--cache warm`, any case whose inputs and output
fit in L2 is replayed from cache: several recorded cases report more than
1,500 GB/s, which DRAM cannot deliver. Treat those as L2-resident results, not
evidence of saved DRAM traffic. `--cache cold` zeroes a scratch buffer of at least
twice the L2 size before each timed call, outside the timed region, as
`triton.testing.do_bench` does. Each sample then averages `--iterations`
single-call timings. The flush keeps the GPU busy while the host submits the next
call, so cold results approximate device time in both timing modes. Each row
operator's `16384 x 4096` shape exceeds L2 even in FP16, so warm mode also includes
one DRAM-bound case per memory-bound operator.

**GPU clocks.** A GPU that has been idle for several seconds starts at reduced clocks, and a
kernel timed while they climb back can run several times slower than it will a moment later.
In [the probe](../results/gpu-clock-probe.log) (`python scripts/gpu_clock_probe.py`) a 2.5 us
softmax took about 25 us in the first 100 ms after a 20 s pause; the
[case study](CASE_STUDIES.md#gpu-clocks-the-first-calls-after-an-idle-pause) has the table. How
long and how bad depends on what else uses the GPU, and a busy desktop can hide it.
`kernel-bench` therefore runs a 200 ms burst of large FP16 GEMMs before each visit to a case,
unless a burst ended within the last 250 ms, outside every timed region and CUDA-graph capture;
it replays every variant once untimed before sampling and records the settings in `warmup`.
Sweeps keep the GPU busy, so this mostly protects the first case and any case that follows a
pause: a sweep taken before the warm-up existed gave the same ratios. Your own timing loops need
the same care.

**Compilation.** Row width and GEMM N/K are `constexpr`, so Triton compiles once per
width or weight shape. Element counts, row counts, strides and GEMM M are runtime
arguments, so batch size never triggers a compile. The first version made every
size compile-time and recompiled for each new batch size; a second made every size
runtime and ran up to 1.9x slower on widths that do not fill their power-of-two
block. The [case study](CASE_STUDIES.md#specialize-what-the-model-fixes) compares
all three. Eager PyTorch kernels are generic, while `torch.compile` also specializes
on the shapes it first sees, so compare against both.

`--timing events` uses the same callables with ordinary Python dispatch and output
allocation. CUDA events measure stream elapsed time, which can include gaps while
the host submits work. It is useful for detecting launch overhead but is not an
isolated kernel duration. Neither mode includes host/device input transfers.

JSON includes all samples, median/min/max, max absolute error, logical I/O bytes,
versions, GPU, seed, timing settings, Git state and Python source hashes. Source
hashes identify the implementation even before a first commit exists.
`gpu_utilization_percent` records nvidia-smi's GPU utilization over about a second
before the run starts and again after it ends, while this process is idle: the
load other programs put on the GPU. The benchmark warns when it is 10% or more.
`warmup` records the burst length, the re-warm gap and how many bursts ran (see *GPU
clocks*). No speedup threshold is asserted; regressions are valid measurements. All displayed ratios
use the PyTorch eager median for that same case and timing mode.

Workshop reports with schema version 2 fingerprint the selected implementation
and its shared dependencies. Reference runs exclude learner exercises; learner
runs include the selected exercise files. Older reports retain their original
hashes, while validation ignores files those runs did not execute. When only the
reporting code changes, `source_snapshots` can point to a preserved copy of the
original measured source: its hash must still match. This preserves old timings
without claiming they were measured using the new reporting code. The maintained
course manifest also excludes editable exercises and journals.

`logical_gbps = minimum_logical_bytes / (median_ms * 1e6)` is an algorithmic
effective bandwidth, **not measured DRAM throughput**. The same useful-work byte
count is used for every variant, including compositions with extra intermediates.
Add counts two reads and one write. Softmax counts one read and one write.
RMSNorm counts two activation reads, one write, and one weight vector, assuming
ideal weight reuse. GEMM's TFLOP/s uses `2*M*N*K`; its I/O count assumes each
matrix is read once. A softmax training step counts five tensor passes (forward:
read x, write y; backward: read y and the gradient, write dx); an RMSNorm step
counts seven, plus three weight-sized vectors. A GEMM training step counts three
GEMMs (the forward product and one gradient per operand: `6*M*N*K` FLOPs) and three
times the forward's matrix traffic. Cache reuse and actual traffic require profiling counters.

Standalone CUDA JSON uses native Windows CUDA events over 50 launches and fifteen
samples, taking one sample of every kernel per round in rotating order, after a
250 ms compute-bound warm-up. It preallocates outputs, tests against CPU references,
and reports FP32. `python scripts/cuda_runs.py` runs it three times and prints the
medians side by side.
Its serial softmax is a pedagogical parallelization baseline. Do not compare its
absolute times directly against WSL Triton CUDA-graph numbers or call that ratio
a framework speedup.

## Companion scripts

Each script prints a table and warms the GPU first; the logs in [results/](../results/)
come from them.

| Script | Question it answers | Log |
| --- | --- | --- |
| `scripts/gpu_clock_probe.py` | How much do idle GPU clocks distort a short timing, and how long does a warm-up burst last? | [gpu-clock-probe.log](../results/gpu-clock-probe.log) |
| `scripts/benchmark_stability.py` | How far do Triton/PyTorch ratios move between full sweeps? | [warm](../results/benchmark-stability.log), [cold](../results/benchmark-stability-cold.log) |
| `scripts/row_plan_grid.py` | Which plan is fastest for each row count and width, and how do the rules score against the best plan? | [row-plan-grid.log](../results/row-plan-grid.log), [repeat](../results/row-plan-grid-repeat.log) |
| `scripts/row_chunk_sweep.py` | The same, at four hand-picked shapes | [row-chunk-sweep.log](../results/row-chunk-sweep.log) |
| `scripts/specialization_sweep.py` | What do compile-time sizes buy and cost? | [specialization-sweep.log](../results/specialization-sweep.log) |
| `scripts/gemm_grouping.py` | Does grouped tile order matter on this GPU? | [gemm-grouping.log](../results/gemm-grouping.log) |
| `scripts/compile_overhead.py`, `scripts/wrapper_overhead.py` | Host time per call: eager wrappers, custom ops, `torch.compile` | [compile](../results/compile-overhead.log), [wrapper](../results/wrapper-overhead.log) |
| `scripts/tolerance_probe.py` | How much of each gradient test tolerance does the worst error use, over many random inputs? | [tolerance-probe.log](../results/tolerance-probe.log) |
| `scripts/cuda_runs.py` | Three runs of the standalone CUDA program, side by side | [cuda-softmax-runs.log](../results/cuda-softmax-runs.log) |

For a defensible claim: repeat on the target hardware, keep an unfavorable shape,
compare against `torch.compile`, note errors/limits, and trace the full model before
claiming end-to-end benefit. A 2x kernel speedup on 10% of runtime gives only
`1 / (0.9 + 0.1/2) = 1.053x` overall before integration overhead.
