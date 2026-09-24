# Logic and code review record

> **Project record, not a learning step.** This page documents how the repo was reviewed and what each review changed.
> You can skip it while learning; start from the
> [README](../../README.md#the-path-step-by-step) instead.

The later beginner-course revision has its own [plan and completed review](BEGINNER_UPGRADE.md)
and [validation evidence](../../results/learning/VALIDATION.md). The original kernel
implementation review below remains applicable; its measured source files are unchanged.

Reviewed 2026-09-23. This is an assistant self-review, not an independent human
review. Scope: implementation, API contracts, GPU math, CUDA synchronization,
benchmark fairness, packaging and documentation. The design was reviewed before
implementation; the evidence below concerns the completed code.

## Logic review

| Finding | Resolution | Status |
| --- | --- | --- |
| A dedicated Triton book could be mistaken for a finished publication | Record Manning MEAP status and public-excerpt-only access; pair it with official docs | Resolved |
| A pure tutorial collection lacks experimental reasoning | Add several variants, independent oracles, raw measurements and cases where custom code loses | Resolved |
| Fusion gains against eager Python alone overstate custom-kernel value | Include compiled PyTorch comparisons for softmax and residual RMSNorm | Resolved |
| A learning schedule could be mistaken for evidence of personal mastery | Implement working material now; document the learner's own reproduction/reimplementation work | Resolved |

## Code review findings and fixes

| Priority | Finding | Fix and evidence |
| --- | --- | --- |
| High | Softmax padding with zero could alter negative-logit rows | Negative-infinity loads; extreme negative and constant-row tests |
| High | Forward kernels would silently disconnect autograd | Reject `requires_grad` inputs before launch |
| High | Contiguous-only addressing could read the wrong values from row views | Use actual row strides; reject noncontiguous columns and overlapping rows; strided tests |
| High | Large logical tensors or strided views could overflow signed 32-bit offsets | Validate physical address spans and GEMM output size; metadata-only large-view tests |
| High | A 2D GEMM launch could exceed CUDA's second grid dimension | Explicit N bound using the smallest candidate BN |
| High | Reusing CUDA shared scratch could race between two reductions | Barrier after all readers consume the broadcast; racecheck reports zero hazards |
| Medium | Serial FP32 softmax accumulated enough drift at width 8192 to fail normalization | Compensated FP32 summation; retained strict row-sum tolerance; CUDA tests pass |
| Medium | CMake's compiler initialization could bypass the intended architecture default | Set the default before `project()`; native Windows build explicitly targets sm_89 |
| Medium | Softmax tests initially used overly loose generic elementwise absolute tolerances | Add softmax-specific tolerances and row-normalization checks |
| Medium | Shape autotuning, compile cost or host gaps could contaminate a performance claim | Warmup/JIT outside timing, graph and event modes labeled separately, raw repeated samples |
| Medium | Missing Triton could break CPU checks | Lazy backend import; explicit GPU gate; CPU-only CI |
| Medium | Current-device mismatch could send launches to the wrong device | Device context around launches; two-GPU regression test present but skipped locally |
| Low | Ambiguous pointer names and inconsistent imports reduce readability | Name output pointers `OUT`, format sources, pass Ruff |

Some rows describe invariants verified during review rather than bugs discovered
in a failing implementation. The observed failures were serial CUDA normalization
drift and initial lint findings; the large-index/grid issue was found by inspection.

## Remaining limitations

- One local NVIDIA Ada GPU; the second-device test is unexecuted.
- Linux CMake build instructions and clean-machine package installation were not
  exercised with a Linux CUDA Toolkit. Native Windows CUDA compilation was tested.
- No gradients, all-infinite/NaN semantics, broad arbitrary-stride support, dynamic
  dispatch beyond the stated bounds, or accuracy guarantee for overflowing FP32 intermediates.
- Measurements use a desktop/WSL environment; small differences can be noise.
  CUDA graph timings are a warm repeated workload, not end-to-end inference latency.
- Nsight Compute counter permissions are unavailable; no counter-derived
  bandwidth or occupancy claim is made.

No known correctness failure remains within the documented and locally tested
scope. See [VALIDATION.md](../../results/VALIDATION.md) for exact executed checks;
this statement is not a production-readiness claim.

## Second review, 2026-09-24

Also an assistant review, not an independent human review. It re-read all code
against the recorded results. It found no correctness failure within the
documented contracts. It did find design and methodology issues that changed
the measurements, so all results were regenerated with the fixed code. The
pre-fix FP16 sweep from the same session is kept in
[results/archive](../../results/archive/) for comparison.

| Priority | Finding | Fix and evidence |
| --- | --- | --- |
| High | Sizes, strides and `eps` were `tl.constexpr`, so every new shape triggered a JIT compile (128 cached `_add` builds) and Triton ran shape-specialized code against PyTorch's generic kernels | Only tile sizes stay `constexpr` in all kernels, exercises and workshop solutions; GPU test asserts that new widths reuse the compiled kernel; case study quantifies the cost |
| High | Default timing replays L2-resident inputs; several cases reported more than 1,500 GB/s against ~736 GB/s DRAM | `--cache cold` flushes L2 before each timed call; one larger-than-L2 shape per row operator; methodology documents residency |
| Medium | RMSNorm's large ratio was measured against a ~10-kernel eager composition | `torch_rms_norm` baseline (built-in `F.rms_norm`, same FP32 semantics) added to every sweep |
| Medium | `requires_grad` check rejected `nn.Parameter` weights under `no_grad`/`inference_mode` | Rejected only while autograd is enabled; CPU and GPU tests |
| Medium | CI never executed kernel code and tested one Python version | Triton interpreter job (`TRITON_INTERPRET=1`), Python 3.10/3.12 matrix, `ruff format --check` |
| Low | Eager wrapper cost ~20 us per call | Measured: ~12 us is Triton's Python launcher, ~4 us validation/allocation/guard; guard now skipped when already on the device |
| Low | Width-1 column views rejected; `revision()` used the caller's cwd; checker stopped at the first TODO; workshop CPU checks not keyed by name; dead negative-stride check; CUDA reduction's 256-thread assumption implicit; CMake without default build type; exercise lesson numbers | Each fixed, with tests where behavior changed |
| Low | `results/learning/source-sha256.json` no longer matched `learning/README.md` before this review | Manifest regenerated |

Not changed deliberately: the 2D GEMM grid (kept for explanation; grouping is
taught in workshop I3), skipping future tiles in causal attention (Capstone A in
the workshop experiments), and a license (the owner's decision).

## Third review, 2026-09-24

Also an assistant review, not an independent human review. It re-read the code,
including the second review's changes, and measured one of those changes instead
of assuming it was free.

| Priority | Finding | Fix and evidence |
| --- | --- | --- |
| High | The second review made every size a runtime argument. Measured afterwards, that ran up to 1.9x slower on row widths that do not fill their power-of-two block (RMSNorm 512x4104: 10.79 vs 5.73 us) and up to 17% slower on fixed-tile GEMM. Softmax at width 4097 used 128 registers per thread instead of 48. Its benefit, no recompiles, mattered only for sizes that vary per request | Row width and GEMM N/K are `constexpr` again; element counts, row counts, strides, GEMM M and eps stay runtime. `scripts/specialization_sweep.py` compares all three versions ([log](../../results/specialization-sweep.log)); both older kernel files are kept in [results/archive](../../results/archive/) |
| High | The case study credited the specialization gain to "folded masks" without evidence | Generated PTX has identical load, exponential and store counts in all versions; the measured difference is register use. Claim withdrawn and replaced |
| Medium | GEMM autotuning re-ran for every distinct M | Autotune key uses a power-of-two bucket of M; a GPU test checks that M=33 and M=40 share one tuning and M=100 starts another |
| Medium | CI would install the newest Triton (3.8.0), not the validated 3.6.0; Windows was missing from the install metadata | `requirements-tested.txt` is a constraints file for CI and `setup_windows.ps1` and lists the Windows, NumPy and Ruff pins; the `gpu` extra installs `triton-windows` on Windows |
| Medium | The benchmark's warm/cold, graph/events and compile paths had no automated test; timings in the docs were copied by hand | GPU smoke tests for every mode and for the workshop benchmark's new `--cache cold`; a CPU test for argument errors; `tests/test_docs.py` fails when a quoted timing matches no saved result. It caught 36 stale numbers after this review's re-run |
| Medium | The new smoke tests broke the sanitizer recipes. Under memcheck, CUDA-graph capture fails with PyTorch's allocator cache off, and the failed capture made every later test error. Under racecheck, their repeated calls and 256 MB L2 flushes ran for over 30 minutes without finishing | With the cache off, the smoke tests use event timing; memcheck then reported 0 errors across all GPU tests. The smoke tests moved to a `BenchmarkSmokeTests` class, and racecheck targets the kernel tests in `GpuTests` |
| Low | Stale `dist/` wheel built from the pre-review kernels; 78 MB of stale compiled kernels | Deleted; wheel and source archive rebuilt from current sources |
| Low | Missing-Triton message named only Linux/WSL; setup script did not check the Python version; project records mixed with guides | Message names both packages; version check added; records moved to `docs/records/` |
| Info | The Triton kernels had never run under Compute Sanitizer | Native Windows: memcheck (allocator cache off) 0 errors, and a deliberately unmasked kernel was reported as a negative control. Racecheck ran every kernel test without reporting a hazard but hung at exit, so the current kernels have no summary line; on the intermediate kernels it completed with 0 hazards |

Not changed deliberately: the exercises and workshop solutions keep every size at
runtime. That is always correct, and the case study uses the contrast to teach the
trade-off. Also unchanged: the 2D GEMM grid, causal tile skipping (workshop
Capstone A), and the license and first commit, which are the owner's decisions.

## Improvements, 2026-09-25

The four larger improvements suggested by the third review, each measured before
being claimed. Details and numbers are in the [case studies](../CASE_STUDIES.md).

| Improvement | What changed | Measured result and remaining limit |
| --- | --- | --- |
| Rows wider than 8192 | `_row_sum_looped`, `_softmax_looped` (online max/sum per lane) and `_rmsnorm_looped` walk a row in chunks; widths up to 2^20. Chunk size and warps follow the row count relative to the SM count, chosen from a sweep | 64 x 131072: row sum 2.91x and softmax 1.20x faster than PyTorch warm, RMSNorm 5.15x faster than `torch.compile`. With only 4 rows softmax loses (0.85x); workshop A2's split design wins there. Cold, the second pass costs softmax its lead at 64 x 131072 |
| Grouped GEMM order and a larger search | One-dimensional tile grid with `GROUP_M` (1 = row-major for the fixed config); eight autotuning configs up to 128 x 256 tiles; every config tested on ragged shapes | 4096^3: 108.3 TFLOP/s against 97.0 for `torch.mm` (1.12x warm, 1.09x cold); 1024^3 now 1.01x. A controlled test showed the gain comes from larger tiles, not grouping: `GROUP_M` 1, 4 and 8 were within 4% on this GPU's 64 MiB L2. The extra index arithmetic costs a little on the tiny 127 x 255 x 65 case |
| `torch.compile` integration | `kernel_portfolio.library` registers every op with `torch.library.triton_op`; launchers route through `wrap_triton`; `opcheck` and compile tests | A 16-op chain's host time per call fell 9.3x with reduce-overhead. A single tiny op does not benefit, and eager calls through `torch.ops` add dispatcher cost, so the plain functions remain the eager API. The earlier suggestion that registration alone removes the eager launch cost was wrong |
| CUDA online softmax | `softmax_online` merges (max, sum) pairs in one pass and uses `float4` loads when the width allows | 1.11x faster than the three-pass kernel at 1024 x 1024; a scalar variant shows the gain comes from `float4` loads, not from the removed read, at this L2-resident size |

Also added: an MIT license, and `scripts/build_site.py` with a Pages workflow for
the course site.
