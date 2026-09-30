# Profiling and compiler inspection

First run correctness tests. Keep profiling runs separate from latency runs because
instrumentation/replay can change timing. Commands assume the corresponding
NVIDIA tools are installed and `results/local` exists.

## Compiler stages (works without hardware counters)

```bash
python scripts/inspect_kernel.py --op softmax --output results/local/ir
python scripts/inspect_kernel.py --op matmul --output results/local/ir
```

The script validates its result, then saves available TTIR, TTGIR, LLVM IR and PTX
and reports registers, spills and shared memory. Find the reduction and masking
in TTIR, distribution/layout choices in TTGIR, and load/store/math instructions
in PTX. PTX is a virtual ISA; it is not the final SASS. The NVIDIA assembler and
driver complete the path to hardware instructions. Exact IR names and structure
depend on the Triton version.

## System timeline, then kernel counters

On Linux/WSL with Nsight tools:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --output=results/local/softmax-timeline \
  python scripts/profile_kernel.py --op softmax
ncu --set full --profile-from-start off --launch-count 1 \
  --target-processes all --export results/local/softmax \
  python scripts/profile_kernel.py --op softmax
```

The script warms compilation/tuning before starting the profiler and marks a
named NVTX range. For GEMM use `--op matmul`. Use the timeline to identify launch
gaps, copies and unexpected extra kernels. Then inspect duration, DRAM throughput,
L2 traffic, active warps, registers, spills, shared memory and stall reasons.
High occupancy or a high cache hit rate alone does not prove a kernel is efficient.

Native CUDA on Windows (Developer PowerShell with tool directories on PATH):

```powershell
compute-sanitizer --tool memcheck --error-exitcode 1 .\build\cuda_portfolio.exe --test-only
compute-sanitizer --tool racecheck --error-exitcode 1 .\build\cuda_portfolio.exe --test-only
ncu --set basic --kernel-name regex:softmax_parallel --launch-skip 6 --launch-count 1 .\build\cuda_portfolio.exe --json results\local\cuda-profiled.json
```

The skip selects
the 1024x1024 validation launch after six small tests. Discard the timing JSON
from profiled runs; it is not comparable to an uninstrumented run.

The Triton kernels deserve the same memory check. It runs natively in PowerShell
with `.venv-win` activated, because the Windows CUDA Toolkit includes Compute
Sanitizer (the WSL environment used here has no toolkit):

```powershell
$env:PYTORCH_NO_CUDA_MEMORY_CACHING = "1"; $env:KERNEL_REQUIRE_GPU = "1"
compute-sanitizer --tool memcheck --error-exitcode 1 python -m unittest tests.test_gpu
Remove-Item Env:PYTORCH_NO_CUDA_MEMORY_CACHING
```

Turn PyTorch's allocator cache off while checking. Normally many tensors share one
large CUDA allocation, so a write past the end of one tensor can land inside
another and go unreported; with caching off, each tensor is its own allocation.
The GPU tests check that inputs are unchanged but cannot see such a write;
memcheck can. CUDA-graph capture needs the cache, so with it off the benchmark
smoke tests switch from graph replay to event timing automatically. As a negative
control, a copy kernel with its store mask removed on purpose was reported as
`Invalid __global__ write` at the unmasked line, while the repository's GPU tests
reported 0 errors ([log](../results/windows-triton-memcheck.log)).

For a fast check of every kernel path, including racecheck, use the smoke script. It
runs each compiled variant once, in every dtype, and compares it with PyTorch,
including every row plan and the backward kernels;
under racecheck it finished in about three minutes here, while the full GPU test
suite takes over an hour:

```powershell
compute-sanitizer --tool racecheck --error-exitcode 1 python scripts/sanitizer_smoke.py
```

On Windows, if the sanitizer hangs after the tests finish, run the base interpreter
instead of the virtual environment's launcher: set `PYTHONPATH` to
`src;.venv-win\Lib\site-packages` and pass the base `python.exe`. The last run
reported 0 hazards ([log](../results/windows-triton-racecheck.log)).

On Linux with the CUDA Toolkit, the equivalent is
`PYTORCH_NO_CUDA_MEMORY_CACHING=1 KERNEL_REQUIRE_GPU=1 compute-sanitizer --tool memcheck --error-exitcode 1 python -m unittest tests.test_gpu`.

## Local limitation

Nsight Compute returned `ERR_NVGPUCTRPERM` on this machine: the driver does not
grant hardware-counter access to this user. The diagnostic is retained in
`results/nsight-cuda.csv`. No bandwidth/occupancy counter results are claimed and
no system-wide driver permission setting was changed. CUDA event timings,
Compute Sanitizer and compiler metadata remain useful independent evidence.

If counters are enabled later by the machine owner, rerun and add one observed
counter-supported bottleneck plus one rejected hypothesis to the case studies.
