# Benchmark methodology

Terms such as *CUDA graph*, *L2*, *warm/cold cache* and *logical GB/s* are defined in
the [glossary](../learning/GLOSSARY.md#measuring). Run the commands below in the GPU
environment, PowerShell or WSL ([every new terminal](SETUP.md#every-new-terminal)).
Compare timings only within one environment: the recorded results came from WSL.

Run from a quiet machine. Close GPU-heavy applications if you want a clean repeat;
the saved local results came from a desktop GPU under WSL/Windows and can vary.

```bash
kernel-bench --op all --dtype float16 --output results/local/all.json
# Same sweep with L2 flushed before every timed call: inputs come from DRAM.
kernel-bench --op all --dtype float16 --cache cold --output results/local/all-cold.json
kernel-bench --op softmax --shape 1024 1024 --dtype float32 --compile --output results/local/softmax-compiled.json
kernel-bench --op rmsnorm --compile --output results/local/norm-compiled.json
kernel-bench --op matmul --dtype bfloat16 --output results/local/gemm-bf16.json
# Separate dispatch-sensitive measurement; do not mix it with graph numbers.
kernel-bench --op softmax --timing events --output results/local/softmax-events.json
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

The default captures 30 invocations per CUDA graph after correctness, JIT,
autotuning and five warmup calls. Each of nine samples times a graph replay using
CUDA events and divides by 30. All cases are prepared first, then each case is
visited three times (`--visits`) in a random order across the run. A visit replays
each variant once untimed, then takes three shuffled rounds of the case's variants
back to back, so a ratio compares measurements taken moments apart. On a GPU shared
with desktop applications, a burst of activity then spoils one visit of one case,
which the median of nine samples discards. Two simpler designs failed on this
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
hashes identify the implementation even before a first commit exists. No speedup
threshold is asserted; regressions are valid measurements. All displayed ratios
use the PyTorch eager median for that same case and timing mode.

`logical_gbps = minimum_logical_bytes / (median_ms * 1e6)` is an algorithmic
effective bandwidth, **not measured DRAM throughput**. The same useful-work byte
count is used for every variant, including compositions with extra intermediates.
Add counts two reads and one write. Softmax counts one read and one write.
RMSNorm counts two activation reads, one write, and one weight vector, assuming
ideal weight reuse. GEMM's TFLOP/s uses `2*M*N*K`; its I/O count assumes each
matrix is read once. Cache reuse and actual traffic require profiling counters.

Standalone CUDA JSON uses native Windows CUDA events over 50 launches and nine
samples. It preallocates outputs, tests against CPU references, and reports FP32.
Its serial softmax is a pedagogical parallelization baseline. Do not compare its
absolute times directly against WSL Triton CUDA-graph numbers or call that ratio
a framework speedup.

For a defensible claim: repeat on the target hardware, keep an unfavorable shape,
compare against `torch.compile`, note errors/limits, and trace the full model before
claiming end-to-end benefit. A 2x kernel speedup on 10% of runtime gives only
`1 / (0.9 + 0.1/2) = 1.053x` overall before integration overhead.

