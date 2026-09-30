# Validation record

Executed locally on 2026-09-30 Asia/Seoul, after the
[improvements of 2026-09-30](../docs/records/REVIEW.md#improvements-2026-09-30):
Triton backward kernels for softmax and residual RMSNorm registered with autograd,
a split path for few wide rows, GPU-load recording in every benchmark report, and a
stricter docs test. This is observed execution evidence; benchmark JSON records UTC
timestamps. Earlier result sets are kept in [archive/](archive/) with the kernel
source each one measured.

The GPU was shared with desktop applications (a browser, Blender and an editor)
throughout; each benchmark report now records how busy the GPU was before and after
its run (9-25% before). The benchmark spreads each case's samples over the run as
described in [BENCHMARKING.md](../docs/BENCHMARKING.md), and the
[stability log](benchmark-stability.log) from 2026-09-26 compares Triton/PyTorch
ratios across five runs; comparisons that moved by more than about 15% are reported
as unresolved.

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
  `-std=c++17 -O3 -lineinfo -arch=sm_89`. The CUDA program did not change in this
  round; its results and sanitizer logs are from 2026-09-26.

## Executed checks

| Check | Result | Evidence |
| --- | --- | --- |
| Full unittest suite with `KERNEL_REQUIRE_GPU=1`, WSL | **49 test methods ran: 48 passed, 1 skipped**; the interpreter class is skipped as a whole and runs separately | [unittest.log](unittest.log) |
| CPU, contract, docs and workflow tests | 23 passed: quoted timings match current results (archived ones only on lines that link the archive); saved results match the current sources; the course manifest matches the course files; site rebuilds and provenance workflows; benchmark `git` calls take no locks | Same log |
| GPU kernel tests | 14 passed: FP32/FP16/BF16, ragged/empty/strided input, every row plan (one block, split, both looped chunkings), `-inf` masks on every plan, rows up to 131072 wide, softmax and RMSNorm gradients against FP64 autograd with dense, transposed and broadcast upstream gradients, a deterministic weight gradient, every GEMM autotuning config, streams, a trained parameter, no recompiles for new batch sizes | Same log |
| Benchmark smoke tests | 4 passed: every timing and cache mode, `torch.compile` variant, a whole interleaved suite, the training suites in both timing modes, workshop cold-cache mode | Same log |
| `torch.compile` custom ops | 5 passed: `opcheck` for all seven ops, including autograd registration and AOT dispatch for softmax and RMSNorm; compiled training steps (static and dynamic shapes) and compiled forward graphs match eager; the other three ops stay forward-only | Same log |
| Checker mutation tests | 2 passed: the vector-add checker rejects an unmasked store, and the fused-GEMM checker rejects two early-rounding kernels | Same log |
| Skipped in the full suite | Two-GPU device guard (one device available); the interpreter test class at setup, because it needs `TRITON_INTERPRET=1` (unittest reports it as a second skip) | Same log |
| Triton CPU interpreter | 5 tests passed, including looped and split wide rows, `-inf` masks, every backward plan and a grouped GEMM with an incomplete group | [interpreter.log](interpreter.log) |
| Native Windows (PowerShell) suite | Same 49 methods: 48 passed, 1 skipped; interpreter 5 passed | [windows-unittest.log](windows-unittest.log), [windows-interpreter.log](windows-interpreter.log) |
| Learning and workshop reference checks | Learning: 40 CPU + 36 GPU = 76 checks passed. Workshops: 11 CPU + 122 GPU = 133 checks passed | [learning](learning/all-solutions.log), [workshops](workshops/checks.log) |
| Triton kernel memory safety | The 26 GPU test methods under memcheck with PyTorch's allocator cache off: 0 errors (25 passed, 1 skipped). Every kernel path in every dtype via `scripts/sanitizer_smoke.py`, now including the split and backward kernels: 0 errors | [windows-triton-memcheck.log](windows-triton-memcheck.log), [smoke run](windows-triton-memcheck-smoke.log) |
| Triton kernel races | Every kernel path in every dtype via `scripts/sanitizer_smoke.py` under racecheck: 0 hazards | [windows-triton-racecheck.log](windows-triton-racecheck.log) |
| CUDA build, correctness, memory safety and races | Unchanged since 2026-09-26: 3 add + 18 softmax cases passed; memcheck 0 errors; racecheck 0 hazards | [cuda-memcheck.log](cuda-memcheck.log), [cuda-racecheck.log](cuda-racecheck.log) |
| Ruff lint and format | Passed over src, tests, scripts, examples and learning | `ruff check` and `ruff format --check` |
| Course site | `scripts/build_site.py` then `mkdocs build --strict`: all pages built, every page link and anchor resolved | Local build |
| FP16 operation sweep, warm and cold L2 | 25 shapes across five operators, sampled in three visits per case; all correctness gates passed in both files | [warm](rtx4080super-triton-fp16.json), [cold](rtx4080super-triton-fp16-cold.json) |
| Training steps | Forward + backward for six softmax and six RMSNorm shapes against PyTorch, `F.rms_norm` and `torch.compile`, graph and event timing; every gradient passed its FP64 check | [graph JSON](rtx4080super-training-fp16.json), [event JSON](rtx4080super-training-events.json) |
| Compiled PyTorch comparisons | Six RMSNorm FP16 and six softmax FP32 shapes passed | [RMSNorm JSON](rtx4080super-rmsnorm-compiled.json), [softmax JSON](rtx4080super-softmax-compiled.json) |
| Dispatch-sensitive event timings | Six FP16 softmax shapes completed | [JSON](rtx4080super-softmax-events.json) |
| Row plans, compile and wrapper overhead | Looped and split choices sampled in shuffled rounds; eager vs compiled host cost; launcher, validation, allocation and guard measured separately | [plans](row-chunk-sweep.log), [compile](compile-overhead.log), [wrapper](wrapper-overhead.log) |
| Inductor's generated kernels | Kernel list and launch sizes for RMSNorm and softmax at 1024 x 1024 and 64 x 131072 | [log](inductor-wide-rows.log) |
| Specialization and grouping sweeps | From 2026-09-25. They measure single-block row kernels and the GEMM, which no later change touched | [specialization](specialization-sweep.log), [grouping](gemm-grouping.log) |
| Standalone CUDA timing | From 2026-09-26: vector add and four softmax variants, sampled in rotating order; three runs recorded | [JSON](rtx4080super-cuda.json), [three runs](cuda-softmax-runs.log) |
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
python -m kernel_portfolio.benchmark --op training --compile --output results/local/training.json
python -m kernel_portfolio.benchmark --op training --timing events --output results/local/training-events.json
python -m kernel_portfolio.benchmark --op rmsnorm --compile --output results/local/norm.json
python -m kernel_portfolio.benchmark --op softmax --dtype float32 --compile --output results/local/softmax.json
python -m kernel_portfolio.benchmark --op softmax --timing events --output results/local/events.json
python scripts/specialization_sweep.py
python scripts/row_chunk_sweep.py
python scripts/gemm_grouping.py
python scripts/compile_overhead.py
python scripts/wrapper_overhead.py
python scripts/inspect_inductor.py
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
on the wrong stream, or an FP32 conversion that Triton performs anyway. Gradients
are validated for softmax and residual RMSNorm only; `add`, `row_sum` and `matmul`
remain forward-only, double backward is unsupported, and no model was trained end
to end. A clean-machine CUDA dependency installation, the CPU-only-PyTorch repair
path in the setup script, Linux CUDA Toolkit/CMake build, other GPUs, two-device
behavior, non-NVIDIA accelerators, distributed collectives and production inference
were not validated. No hardware-counter bottleneck claim or end-to-end model
speedup is made. Numerical checks cover finite inputs and `-inf` masks, not all
possible floating-point inputs.
