# Validation record

Executed locally on 2026-09-26 Asia/Seoul, after the
[fourth review](../docs/records/REVIEW.md#fourth-review-2026-09-26): `-inf` masks in
looped and CUDA online softmax, checkers that reject deliberately broken kernels,
provenance tests, and benchmark sampling that tolerates other GPU users. This is
observed execution evidence; benchmark JSON records UTC timestamps. Earlier result
sets are kept in [archive/](archive/) with the kernel source each one measured.

The GPU was shared with desktop applications (a browser, Blender and an editor)
throughout. Bursts of their work slowed whole benchmark cases 2-3x in two full
sweeps, which led to the sampling change described in
[BENCHMARKING.md](../docs/BENCHMARKING.md). The
[stability log](benchmark-stability.log) compares Triton/PyTorch ratios across five
runs; comparisons that moved by more than about 15% are reported as unresolved.

## Environment

- NVIDIA GeForce RTX 4080 SUPER, 16 GiB, 64 MiB L2, compute capability 8.9, 80 SMs;
  driver 595.97.
- Triton path used for all benchmark results: Ubuntu 22.04 on WSL2, Python 3.12.13,
  PyTorch 2.11.0+cu130, Triton 3.6.0, PyTorch CUDA runtime 13.0. An existing
  isolated Python environment was reused without installing anything into it.
- Native Windows Triton path (PowerShell, option A in setup): Windows 11, Python
  3.14, PyTorch 2.11.0+cu130, `triton-windows` 3.6.0.post26, NumPy 2.3.5, created
  and re-run by `scripts/setup_windows.ps1` from `requirements-tested.txt`.
- CUDA C++ path: native Windows, Visual Studio C++ tools, CUDA Toolkit 13.3,
  `-std=c++17 -O3 -lineinfo -arch=sm_89`. The executable's CUDA runtime/driver
  API version fields are preserved separately in its JSON.

## Executed checks

| Check | Result | Evidence |
| --- | --- | --- |
| Full unittest suite with `KERNEL_REQUIRE_GPU=1`, WSL | **36 test methods ran: 35 passed, 1 skipped**; the interpreter class is skipped as a whole and runs separately | [unittest.log](unittest.log) |
| CPU, contract and provenance tests | 15 passed: quoted timings exist in saved results; every saved result was measured on the current sources; the course manifest matches the course files; archived results match archived kernels | Same log |
| GPU kernel tests | 12 passed: FP32/FP16/BF16, ragged/empty/strided input, `-inf` masks at every width and both looped chunk sizes, 160-row looped cases, rows up to 131072 wide, extreme values, every GEMM autotuning config, streams, parameters under `no_grad`, no recompiles for new batch sizes | Same log |
| Benchmark smoke tests | 3 passed: every timing and cache mode, `torch.compile` variant, a whole interleaved suite, workshop cold-cache mode | Same log |
| `torch.compile` custom ops | 3 passed: `opcheck` for all five ops; compiled graphs (full, reduce-overhead, dynamic row counts) match eager | Same log |
| Checker mutation tests | 2 passed: the vector-add checker rejects an unmasked store, and the fused-GEMM checker rejects two early-rounding kernels | Same log |
| Skipped in the full suite | Two-GPU device guard (one device available); the interpreter test class at setup, because it needs `TRITON_INTERPRET=1` (unittest reports it as a second skip) | Same log |
| Triton CPU interpreter | 4 tests passed, including looped wide rows, `-inf` masks and a grouped GEMM with an incomplete group | [interpreter.log](interpreter.log) |
| Native Windows (PowerShell) suite | Same 36 methods: 35 passed, 1 skipped; interpreter 4 passed | [windows-unittest.log](windows-unittest.log), [windows-interpreter.log](windows-interpreter.log) |
| Learning and workshop reference checks | Learning: 40 CPU + 36 GPU = 76 checks passed. Workshops: 11 CPU + 122 GPU = 133 checks passed | [learning](learning/all-solutions.log), [workshops](workshops/checks.log) |
| Triton kernel memory safety | The 21 GPU test methods under memcheck with PyTorch's allocator cache off: 0 errors (20 passed, 1 skipped). Every kernel path in every dtype via `scripts/sanitizer_smoke.py`: 0 errors | [windows-triton-memcheck.log](windows-triton-memcheck.log), [smoke run](windows-triton-memcheck-smoke.log) |
| Triton kernel races | Every kernel path in every dtype via `scripts/sanitizer_smoke.py` under racecheck: 0 hazards | [windows-triton-racecheck.log](windows-triton-racecheck.log) |
| CUDA build and correctness | Built; 3 add + 18 softmax cases passed (serial, parallel and online kernels, including a masked row), with the output poisoned before each kernel | Source/build command in setup |
| CUDA mutation checks | A copy with the online kernel's last `float4` skipped, and a copy without the `-inf` guard, both failed the test program | Local builds in a scratch directory |
| CUDA memory safety and races | memcheck 0 errors; racecheck 0 hazards, on the final binary | [cuda-memcheck.log](cuda-memcheck.log), [cuda-racecheck.log](cuda-racecheck.log) |
| Install constraints and setup | `scripts/setup_windows.ps1` re-ran on the existing environment and ended with its new GPU check; with `python` removed from PATH it stopped with its own message | Local runs |
| Ruff lint and format | Passed over src, tests, scripts, examples and learning | `ruff check` and `ruff format --check` |
| Course site | `scripts/build_site.py` then `mkdocs build --strict`: all pages built, every page link and anchor resolved | Local build |
| FP16 operation sweep, warm and cold L2 | 25 shapes across five operators, sampled in three visits per case; all correctness gates passed in both files | [warm](rtx4080super-triton-fp16.json), [cold](rtx4080super-triton-fp16-cold.json) |
| Run-to-run stability | Five FP16 runs compared case by case | [benchmark-stability.log](benchmark-stability.log) |
| Compiled PyTorch comparisons | Six RMSNorm FP16 and six softmax FP32 shapes passed | [RMSNorm JSON](rtx4080super-rmsnorm-compiled.json), [softmax JSON](rtx4080super-softmax-compiled.json) |
| Dispatch-sensitive event timings | Six FP16 softmax shapes completed | [JSON](rtx4080super-softmax-events.json) |
| Chunking, compile overhead, wrapper overhead | Looped chunk choices sampled in shuffled rounds; eager vs compiled host cost; launcher, validation, allocation and guard measured separately | [chunks](row-chunk-sweep.log), [compile](compile-overhead.log), [wrapper](wrapper-overhead.log) |
| Specialization and grouping sweeps | From 2026-09-25. They measure single-block row kernels and the GEMM, which this review did not change; a repeat on 2026-09-26 was dominated by other GPU users | [specialization](specialization-sweep.log), [grouping](gemm-grouping.log) |
| Standalone CUDA timing | Vector add and four softmax variants, sampled in rotating order; three runs recorded | [JSON](rtx4080super-cuda.json), [three runs](cuda-softmax-runs.log) |
| Compiler inspection | Both operators validated; TTIR/TTGIR/LLVM IR/PTX saved | [softmax metadata](compiler/softmax-metadata.json), [GEMM metadata](compiler/matmul-metadata.json) |
| Nsight Systems timeline | From 2026-09-23, before later CUDA rebuilds | [CSV](nsight-kernels_cuda_gpu_kern_sum.csv) |
| Nsight Compute counters | Unavailable: `ERR_NVGPUCTRPERM` | [diagnostic](nsight-cuda.csv) |

The Python benchmark files record source SHA-256 identities, and
`tests/test_docs.py` fails if any of them no longer matches the current sources.
Hashes read CRLF as LF, so Windows working copies and Git checkouts agree. CUDA,
Triton and profiler workloads from this project ran sequentially, but the desktop
applications sharing the GPU were not controlled.

## Commands to reproduce

From a configured Linux environment at the repository root:

```bash
KERNEL_REQUIRE_GPU=1 python -m unittest discover -s tests -v
TRITON_INTERPRET=1 KERNEL_REQUIRE_INTERPRETER=1 python -m unittest discover -s tests -p 'test_interpreter.py' -v
python -m kernel_portfolio.benchmark --op all --dtype float16 --output results/local/all.json
python -m kernel_portfolio.benchmark --op all --dtype float16 --cache cold --output results/local/all-cold.json
python -m kernel_portfolio.benchmark --op rmsnorm --compile --output results/local/norm.json
python -m kernel_portfolio.benchmark --op softmax --dtype float32 --compile --output results/local/softmax.json
python -m kernel_portfolio.benchmark --op softmax --timing events --output results/local/events.json
python scripts/specialization_sweep.py
python scripts/row_chunk_sweep.py
python scripts/gemm_grouping.py
python scripts/compile_overhead.py
python scripts/wrapper_overhead.py
python scripts/inspect_kernel.py --op softmax
python scripts/inspect_kernel.py --op matmul
python scripts/course_manifest.py
python scripts/report_results.py
```

See [SETUP.md](../docs/SETUP.md) for CUDA build commands and
[PROFILING.md](../docs/PROFILING.md) for sanitizer/profiler commands.

## Not established by these checks

The CI and Pages workflows run only on GitHub; their runs are recorded there, not
here. The full GPU suite has no racecheck summary; the smoke script covers every
kernel path instead, and memcheck covers only the inputs the tests exercise. The
course checkers still cannot observe a missing load mask (memcheck can), a launch
on the wrong stream, or an FP32 conversion that Triton performs anyway. A
clean-machine CUDA dependency installation, the CPU-only-PyTorch repair path in
the setup script, Linux CUDA Toolkit/CMake build, other GPUs, two-device behavior,
backward gradients, non-NVIDIA accelerators, distributed collectives and production
inference were not validated. No hardware-counter bottleneck claim or end-to-end
model speedup is made. Forward numerical checks cover finite inputs and `-inf`
masks, not all possible floating-point inputs.
