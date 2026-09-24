# Validation record

Executed locally on 2026-09-25 Asia/Seoul, after the
[improvements](../docs/records/REVIEW.md#improvements-2026-09-25): looped kernels for
wide rows, grouped GEMM order with a larger search, `torch.compile` custom ops and an
online CUDA softmax. This is observed execution evidence; benchmark JSON records UTC
timestamps. Earlier result sets are kept in [archive/](archive/) with the kernel
source each one measured.

Other desktop applications were using the GPU during this run (about 40%
utilization at idle clocks), so host-side timings are higher than in a quiet run.
Every reported comparison is within one run.

## Environment

- NVIDIA GeForce RTX 4080 SUPER, 16 GiB, 64 MiB L2, compute capability 8.9, 80 SMs;
  driver 595.97.
- Triton path used for all benchmark results: Ubuntu 22.04 on WSL2, Python 3.12.13,
  PyTorch 2.11.0+cu130, Triton 3.6.0, PyTorch CUDA runtime 13.0. An existing
  isolated Python environment was reused without installing anything into it.
- Native Windows Triton path (PowerShell, option A in setup): Windows 11, Python
  3.14, PyTorch 2.11.0+cu130, `triton-windows` 3.6.0.post26, NumPy 2.3.5, created
  by `scripts/setup_windows.ps1` from `requirements-tested.txt`.
- CUDA C++ path: native Windows, Visual Studio C++ tools, CUDA Toolkit 13.3,
  `-std=c++17 -O3 -lineinfo -arch=sm_89`. The executable's CUDA runtime/driver
  API version fields are preserved separately in its JSON.

## Executed checks

| Check | Result | Evidence |
| --- | --- | --- |
| Full unittest suite with `KERNEL_REQUIRE_GPU=1`, WSL | **32 test methods: 28 passed, 4 skipped** | [unittest.log](unittest.log) |
| CPU, contract and docs tests | 12 passed, including benchmark argument errors and quoted-timing consistency | Same log |
| GPU kernel tests | 11 passed: FP32/FP16/BF16, ragged/empty/strided input, rows up to 131072 wide, extreme values, every GEMM autotuning config, streams, parameters under `no_grad`, no recompiles for new batch sizes | Same log |
| Benchmark smoke tests | 2 passed: every timing and cache mode, `torch.compile` variant, workshop cold-cache mode | Same log |
| `torch.compile` custom ops | 3 passed: `opcheck` for all five ops; compiled graphs (full, reduce-overhead, dynamic row counts) match eager | Same log |
| Skipped in the full suite | Two-GPU device guard (one device available); three interpreter tests that need `TRITON_INTERPRET=1` | Same log |
| Triton CPU interpreter | 3 tests passed, including looped wide rows and a grouped GEMM with an incomplete group | [interpreter.log](interpreter.log) |
| Native Windows (PowerShell) suite | Same 32 methods: 28 passed, 4 skipped; interpreter passed | [windows-unittest.log](windows-unittest.log), [windows-interpreter.log](windows-interpreter.log) |
| Learning and workshop reference checks | Learning: 40 CPU + 34 GPU = 74 checks passed. Workshops: 11 CPU + 120 GPU = 131 checks passed | [learning](learning/all-solutions.log), [workshops](workshops/checks.log) |
| Triton kernel memory safety | All GPU tests under memcheck with PyTorch's allocator cache off: 0 errors. Every kernel path via `scripts/sanitizer_smoke.py`: 0 errors. A deliberately unmasked kernel was reported as a negative control | [windows-triton-memcheck.log](windows-triton-memcheck.log), [smoke run](windows-triton-memcheck-smoke.log) |
| Triton kernel races | Every kernel path via `scripts/sanitizer_smoke.py` under racecheck: 0 hazards, 17 seconds. The full GPU suite under racecheck ran over an hour without finishing and was stopped | [windows-triton-racecheck.log](windows-triton-racecheck.log) |
| CUDA build and correctness | Built; 3 add + 18 softmax cases passed (serial, parallel and online kernels) | Source/build command in setup |
| CUDA memory safety and races | memcheck 0 errors; racecheck 0 hazards | [cuda-memcheck.log](cuda-memcheck.log), [cuda-racecheck.log](cuda-racecheck.log) |
| Interpreter NumPy compatibility | GEMM interpreter test fails with NumPy 2.4.6 and 2.5.3, passes with 2.3.5; pinned below 2.4 | Local runs in both environments |
| Install constraints | `pip install -c requirements-tested.txt -e .[gpu,dev]` resolves the validated versions on Windows; the setup script ran with it | Local dry run and setup run |
| Ruff lint and format | Passed over src, tests, scripts, examples and learning | `ruff check` and `ruff format --check` |
| Course site | `scripts/build_site.py` then `mkdocs build --strict`: all pages built, every page link and anchor resolved | Local build |
| Package build | Wheel carries the current kernels, `library.py` and the MIT license; source archive includes scripts, archive and records | Local `dist/` (ignored generated artifacts) |
| FP16 operation sweep, warm and cold L2 | 25 shapes across five operators, including two row widths above 8192 and a 4096^3 GEMM; all correctness gates passed in both files | [warm](rtx4080super-triton-fp16.json), [cold](rtx4080super-triton-fp16-cold.json) |
| Compiled PyTorch comparisons | Six RMSNorm FP16 and six softmax FP32 shapes passed | [RMSNorm JSON](rtx4080super-rmsnorm-compiled.json), [softmax JSON](rtx4080super-softmax-compiled.json) |
| Dispatch-sensitive event timings | Six FP16 softmax shapes completed | [JSON](rtx4080super-softmax-events.json) |
| Specialization, chunking, grouping, compile overhead | Three kernel versions; looped chunk choices; `GROUP_M` 1/4/8 on one tile; eager vs compiled host cost | [specialization](specialization-sweep.log), [chunks](row-chunk-sweep.log), [grouping](gemm-grouping.log), [compile](compile-overhead.log) |
| Wrapper host overhead | Launcher, validation, allocation and guard measured separately | [log](wrapper-overhead.log) |
| Standalone CUDA timing | Vector add and four softmax variants completed | [JSON](rtx4080super-cuda.json) |
| Compiler inspection | Both operators validated; TTIR/TTGIR/LLVM IR/PTX saved | [softmax metadata](compiler/softmax-metadata.json), [GEMM metadata](compiler/matmul-metadata.json) |
| Nsight Systems timeline | From 2026-09-23, before later CUDA rebuilds | [CSV](nsight-kernels_cuda_gpu_kern_sum.csv) |
| Nsight Compute counters | Unavailable: `ERR_NVGPUCTRPERM` | [diagnostic](nsight-cuda.csv) |

The Python benchmark files include source SHA-256 identities that match the
sources they measured. CUDA, Triton and profiler workloads from this project ran
sequentially, but unrelated desktop activity was not controlled.

## Commands to reproduce

From a configured Linux environment at the repository root:

```bash
KERNEL_REQUIRE_GPU=1 python -m unittest discover -s tests -v
TRITON_INTERPRET=1 python -m unittest discover -s tests -p 'test_interpreter.py' -v
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
python scripts/report_results.py
```

See [SETUP.md](../docs/SETUP.md) for CUDA build commands and
[PROFILING.md](../docs/PROFILING.md) for sanitizer/profiler commands.

## Not established by these checks

The CI and Pages workflows run only on GitHub; their first runs are recorded there,
not here. The full GPU suite has no racecheck summary; the smoke script covers every
kernel path instead, and memcheck covers only the inputs the tests exercise. A
clean-machine CUDA dependency installation, Linux CUDA Toolkit/CMake build, other
GPUs, two-device behavior, backward gradients, non-NVIDIA accelerators, distributed
collectives and production inference were not validated. No hardware-counter
bottleneck claim or end-to-end model speedup is made. Forward numerical checks
cover the documented finite-input scope, not all possible floating-point inputs.
