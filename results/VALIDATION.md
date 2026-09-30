# Validation record

Executed locally on 2026-09-30 Asia/Seoul, after the
[improvements of 2026-09-30](../docs/records/REVIEW.md#improvements-2026-09-30) and the
[follow-up round that evening](../docs/records/REVIEW.md#follow-up-improvements-2026-09-30-evening):
Triton backward kernels for every operator, a width-aware chunk rule for looped rows, GPU
warm-up and load recording in every benchmark report, and stricter tests. This is observed
execution evidence; benchmark JSON records UTC timestamps. Earlier result sets are kept in
[archive/](archive/) with the kernel source each one measured.

The GPU was shared with desktop applications and other programs (a browser, Blender, an editor,
other test runs) throughout; each benchmark report records how busy the GPU was before and after its
run (3-40% before). The benchmark spreads each case's samples over the run
as described in [BENCHMARKING.md](../docs/BENCHMARKING.md), and the stability logs repeat the
suites three times each: 29 of 36 warm and 29 of 36 cold
Triton/PyTorch ratios agree within 10% ([warm](benchmark-stability.log),
[cold](benchmark-stability-cold.log)), and 12 of 17 training-step ratios
([training](benchmark-stability-training.log)); comparisons that moved by more than about 15% are
reported as unresolved.

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
  `-std=c++17 -O3 -lineinfo -arch=sm_89`. The CUDA program gained a GPU warm-up in this
  round; it was rebuilt, and its correctness, sanitizer and timing results are from 2026-09-30.

## Executed checks

| Check | Result | Evidence |
| --- | --- | --- |
| Full unittest suite with `KERNEL_REQUIRE_GPU=1`, WSL | **69 test methods ran: 68 passed, 1 skipped**; the interpreter class is skipped as a whole and runs separately (unittest reports it as a second skip) | [unittest.log](unittest.log) |
| CPU, contract, docs and workflow tests | 23 passed: quoted timings match current results (archived ones only on lines that link the archive); saved results match the current sources; the course manifest matches the course files; site rebuilds and provenance workflows; benchmark `git` calls take no locks and run in the package directory; the benchmark CLI rejects FP32 GEMM suites; index arithmetic keeps 65536 of headroom below 2^31 | Same log |
| Row-plan rules | 6 passed: which plan every kind of shape reaches on an 80-SM GPU (pure functions, no GPU needed), and the grid script's scoring of the current and previous rules | Same log |
| GPU kernel tests | 24 passed: FP32/FP16/BF16, ragged/empty/strided input, every row plan, `-inf` masks on every plan, rows up to 131072 wide; gradients of every operator against FP64 autograd with dense, transposed, broadcast (rows, columns and scalar) and large-mean upstream gradients, a gradient that follows the RMSNorm input, a deterministic weight gradient, every backward GEMM configuration, frozen operands, saved tensors, gradients of one add or norm that must not alias; every GEMM autotuning config, streams, a trained parameter, no recompiles for new batch sizes | Same log |
| Benchmark smoke tests | 6 passed: every timing and cache mode, `torch.compile` variant, a whole interleaved suite, the training suites (softmax, RMSNorm and GEMM) in both timing modes, the GPU warm-up in and out of inference mode, per-run warm-up counts, workshop cold-cache mode | Same log |
| `torch.compile` custom ops | 6 passed: `opcheck` for all nine ops, including autograd registration and AOT dispatch; compiled training steps (static and dynamic shapes) through all five operators and compiled forward graphs match eager | Same log |
| Training example | 2 passed: a small model trained with all five operators tracks its plain-PyTorch twin within 0.5% at every step, eager and with the whole step compiled | Same log |
| Checker mutation tests | 2 passed: the vector-add checker rejects an unmasked store, and the fused-GEMM checker rejects two early-rounding kernels | Same log |
| Skipped in the full suite | Two-GPU device guard (one device available); the interpreter test class at setup, because it needs `TRITON_INTERPRET=1` | Same log |
| Triton CPU interpreter | 12 tests passed, including looped and split wide rows, `-inf` masks, every backward plan and both backward GEMM products | [interpreter.log](interpreter.log) |
| Native Windows (PowerShell) suite | 69 test methods ran: 68 passed, 1 skipped (the same 2 skips as above); interpreter 12 passed | [windows-unittest.log](windows-unittest.log), [windows-interpreter.log](windows-interpreter.log) |
| Learning and workshop reference checks | Learning: 40 CPU + 36 GPU = 76 checks passed. Workshops: 11 CPU + 122 GPU = 133 checks passed | [learning](learning/all-solutions.log), [workshops](workshops/checks.log) |
| Triton kernel memory safety | Every kernel path in every dtype via `scripts/sanitizer_smoke.py` (163 paths, including the new chunk plans, the weight-gradient kernels and every backward GEMM configuration): ERROR SUMMARY: 0 errors. The full GPU suite under memcheck was not re-run this round; its last run, from before this round's kernels, is kept | [smoke run](windows-triton-memcheck-smoke.log); [earlier full-suite run](windows-triton-memcheck.log) |
| Triton kernel races | Every kernel path in every dtype via `scripts/sanitizer_smoke.py` under racecheck: RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) | [windows-triton-racecheck.log](windows-triton-racecheck.log) |
| CUDA build, correctness, memory safety and races | Rebuilt this round: 3 add + 18 softmax cases passed; memcheck ERROR SUMMARY: 0 errors; racecheck RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) | [cuda-memcheck.log](cuda-memcheck.log), [cuda-racecheck.log](cuda-racecheck.log) |
| Ruff lint and format | Passed over src, tests, scripts, examples and learning | `ruff check` and `ruff format --check` |
| Course site | `scripts/build_site.py` then `mkdocs build --strict`: all pages built, every page link and anchor resolved | Local build |
| FP16 operation sweep, warm and cold L2 | 25 shapes across five operators, 15 samples in five visits per case; all correctness gates passed in both files | [warm](rtx4080super-triton-fp16.json), [cold](rtx4080super-triton-fp16-cold.json) |
| Training steps | Forward + backward for six softmax, six RMSNorm and five GEMM shapes against PyTorch, `F.rms_norm` and `torch.compile`, graph and event timing; every gradient passed its FP64 check | [graph JSON](rtx4080super-training-fp16.json), [event JSON](rtx4080super-training-events.json) |
| Repeat runs | The FP16 warm and cold sweeps and the training suites, three runs each, compared ratio by ratio | [warm](benchmark-stability.log), [cold](benchmark-stability-cold.log), [training](benchmark-stability-training.log) |
| Compiled PyTorch comparisons | Six RMSNorm FP16 and six softmax FP32 shapes passed | [RMSNorm JSON](rtx4080super-rmsnorm-compiled.json), [softmax JSON](rtx4080super-softmax-compiled.json) |
| Dispatch-sensitive event timings | Six FP16 softmax shapes completed | [JSON](rtx4080super-softmax-events.json) |
| GPU clocks | Idle-gap probe with the SM clock and load read at each wake-up, and the effect of a warm-up burst; a sweep from before the warm-up is archived for comparison | [log](gpu-clock-probe.log), [archive](archive/) |
| Row plans | Every plan on 332 shape cells twice, the rules scored against the best plan, and thresholds fitted on one run and scored on the other; four hand-picked shapes; compile and wrapper overhead | [grid](row-plan-grid.log), [repeat](row-plan-grid-repeat.log), [fit](row-plan-fit.log), [plans](row-chunk-sweep.log), [compile](compile-overhead.log), [wrapper](wrapper-overhead.log) |
| Gradient tolerances | Worst error over the tolerance for every backward kernel over 300 random inputs each | [tolerance-probe.log](tolerance-probe.log) |
| Small model | Loss curves of the Triton model and its PyTorch twin, eager and compiled | [eager](train-tiny.log), [compiled](train-tiny-compiled.log) |
| Inductor's generated kernels | Kernel list and launch sizes for RMSNorm and softmax at 1024 x 1024 and 64 x 131072 | [log](inductor-wide-rows.log) |
| Specialization and grouping sweeps | Re-run this round, with the GPU warm-up | [specialization](specialization-sweep.log), [grouping](gemm-grouping.log) |
| Standalone CUDA timing | Vector add and four softmax variants, sampled in rotating order after a warm-up; seven runs recorded, the middle one saved | [JSON](rtx4080super-cuda.json), [runs](cuda-softmax-runs.log) |
| Compiler inspection | Both operators validated; TTIR/TTGIR/LLVM IR/PTX saved | [softmax metadata](compiler/softmax-metadata.json), [GEMM metadata](compiler/matmul-metadata.json) |
| Nsight Systems timeline | From 2026-09-23, before later CUDA rebuilds | [CSV](nsight-kernels_cuda_gpu_kern_sum.csv) |
| Nsight Compute counters | Unavailable: `ERR_NVGPUCTRPERM` | [diagnostic](nsight-cuda.csv) |

The Python benchmark files record source SHA-256 identities, and
`tests/test_docs.py` fails if any of them no longer matches the current sources.
Hashes read CRLF as LF, so Windows working copies and Git checkouts agree. CUDA,
Triton and profiler workloads from this project ran sequentially, but the desktop
applications and other programs sharing the GPU were not controlled.

## Commands to reproduce

From a configured Linux environment at the repository root:

```bash
KERNEL_REQUIRE_GPU=1 python -m unittest discover -s tests -v
TRITON_INTERPRET=1 KERNEL_REQUIRE_INTERPRETER=1 python -m unittest discover -s tests -p 'test_interpreter.py' -v
python -m kernel_portfolio.benchmark --op all --dtype float16 --samples 15 --visits 5 --output results/local/all.json
python -m kernel_portfolio.benchmark --op all --dtype float16 --cache cold --samples 15 --visits 5 --output results/local/all-cold.json
python -m kernel_portfolio.benchmark --op training --compile --samples 15 --visits 5 --output results/local/training.json
python -m kernel_portfolio.benchmark --op training --timing events --samples 15 --visits 5 --output results/local/training-events.json
python -m kernel_portfolio.benchmark --op rmsnorm --compile --samples 15 --visits 5 --output results/local/norm.json
python -m kernel_portfolio.benchmark --op softmax --dtype float32 --compile --samples 15 --visits 5 --output results/local/softmax.json
python -m kernel_portfolio.benchmark --op softmax --timing events --samples 15 --visits 5 --output results/local/events.json
python scripts/benchmark_stability.py --runs 3
python scripts/benchmark_stability.py --runs 3 --cache cold
python scripts/benchmark_stability.py --runs 3 --op training
python scripts/gpu_clock_probe.py
python scripts/row_plan_grid.py --full
python scripts/row_plan_fit.py results/row-plan-grid.log results/row-plan-grid-repeat.log
python scripts/tolerance_probe.py --trials 300
python examples/train_tiny.py
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

On Windows, `python scripts/cuda_runs.py` times the CUDA program.

See [SETUP.md](../docs/SETUP.md) for CUDA build commands and
[PROFILING.md](../docs/PROFILING.md) for sanitizer/profiler commands.

## Not established by these checks

The CI and Pages workflows run only on GitHub; their runs are recorded there, not
here. The full GPU suite has no racecheck summary and was not re-run under memcheck
this round; the smoke script covers every kernel path instead, and memcheck covers
only the inputs the tests exercise. The course checkers still cannot observe a
missing load mask (memcheck can), a launch on the wrong stream, or an FP32
conversion that Triton performs anyway. Gradients are validated for every operator
against FP64 autograd and a small model was trained beside its PyTorch twin, but
double backward through the Triton kernels is unsupported, an RMSNorm weight that is
frozen still gets its gradient computed, and no model was trained at scale. The row
plan grid times forward softmax and RMSNorm only. A clean-machine CUDA dependency
installation, the CPU-only-PyTorch repair path in the setup script, Linux CUDA
Toolkit/CMake build, other GPUs, two-device behavior, non-NVIDIA accelerators,
distributed collectives and production inference were not validated. No
hardware-counter bottleneck claim or end-to-end model speedup is made. Numerical
checks cover finite inputs and `-inf` masks, not all possible floating-point inputs.
