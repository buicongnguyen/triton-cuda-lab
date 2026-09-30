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
- No double backward, all-infinite/NaN semantics, broad arbitrary-stride support, dynamic
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

## Fourth review, 2026-09-26

An assistant review again, this time with two helper reviews (course code; scripts
and CI) whose findings were confirmed by running them before anything was changed.
Most findings share one pattern: a check that could not fail. Several were shown by
deliberately breaking the code and watching the check still pass.

| Priority | Finding | Fix and evidence |
| --- | --- | --- |
| High | Softmax on rows wider than 8192 returned NaN for every row containing `-inf` (masked attention scores). The looped kernel's online merge computed `exp(-inf - (-inf))`; the single-block kernel was correct, and every test used finite inputs | A lane that has seen only `-inf` shifts by 0. `test_masked_softmax` covers both kernels, both chunk sizes, three dtypes and fully masked rows (NaN, as in PyTorch); an interpreter test runs it in CI. The CUDA online kernel had the same gap and the same guard |
| High | The CUDA test program ran three softmax kernels into one output buffer, so an online kernel that skipped its last `float4` still passed on the previous kernel's values | The buffer is filled with NaN before each kernel, and that broken kernel now fails. Failures name the kernel, row and column |
| High | Two course checkers accepted broken kernels. The vector-add checker could not see stores past N; the fused-GEMM checker accepted an FP16 accumulator (tolerance 0.02, K at most 33) | A guard-buffer check, an exact FP16 case (2048 + 1 + 1 = 2050 only with one rounding) and a K = 1024 case. GPU tests assert that the checkers reject these broken kernels |
| Medium | The validation record said benchmark source hashes match the sources, but nothing checked it. One workshop file had been hashed from a CRLF working copy, so no fresh clone could reproduce its hash | Hashes read CRLF as LF. `tests/test_docs.py` checks every saved result against the current sources, the course manifest against the course files, and archived results against archived kernels |
| Medium | On a GPU shared with desktop applications, a burst of activity spoiled every sample of whichever case was being timed. Two consecutive full sweeps each had one section 2-3x slow for PyTorch and Triton alike (row sum 1024 x 1024 in one, RMSNorm 512 x 4097 in the other); re-running the same code gave the usual numbers | The benchmark prepares all cases, then visits each case three times in random order; a burst costs one visit (three of nine samples). The chunk sweep interleaves its variants too. A first version interleaved every sample of every case; that separated a case's variants in time and made tiny kernels pay to reload their CUDA graphs, so each case is now visited three times and its variants are timed back to back within a visit. The recorded results come from a sweep run after this change |
| Medium | The case study credited a 1.11x CUDA online-softmax gain to `float4` loads, from one run that timed the kernels one after another | The CUDA program now samples all kernels in rotating order. Over three runs the online kernels were 1.00-1.12x faster than the three-pass kernel and `float4` stayed within 3% of scalar loads; the claim is withdrawn ([runs](../../results/cuda-softmax-runs.log)) |
| Medium | The saved CUDA sanitizer logs predated the online kernel (12 softmax cases), although the docs cited them for it | Re-run on the current binary: 18 cases, 0 errors, 0 hazards |
| Medium | The 2048-wide chunk path (two or more rows per SM) ran only in the sanitizer smoke script | GPU tests add 160-row looped cases for all three row kernels; the smoke script now covers every dtype |
| Medium | CI's interpreter job would pass if its tests were skipped | `KERNEL_REQUIRE_INTERPRETER=1` turns that skip into a failure |
| Medium | `setup_windows.ps1` kept a CPU-only PyTorch when re-run, printed "Done" without a usable GPU, and could not show its own "python was not found" message | It checks `torch.version.cuda` and reinstalls, ends with a GPU check, and looks for python before calling it |
| Low | Pages workflow: deploy permissions on the build job, a new push could cancel a running deployment, and actions on the deprecated Node 20 | Permissions on the deploy job only, no cancellation, current action versions; pull requests build the site without deploying |
| Low | Benchmarks listed 4-warp and 8-warp softmax for looped rows, where `num_warps` does not apply | One `triton` variant for rows wider than 8192 |
| Low | `build_site.py --out` could delete a source folder; `report_results.py` depended on the working directory; the CUDA build script needed the repo root; the workshop benchmark lost all results on one unfinished exercise | Guards, repo-relative paths, a `CUDA_ARCH` override, and unfinished exercises recorded as skipped |
| Low | Lesson text: the `<= N` bug example used N=256, which exposes nothing; a 2 MiB input was called 4 MB; the workshop README said graph timing includes allocation | Corrected. The FP32-conversion and non-default-stream notes now say what the checker can and cannot observe |

Not fixed: `tests/test_docs.py` accepts any saved median, archived ones included,
and checks only three-decimal microsecond figures. The non-default-stream check
cannot prove which stream a kernel used, and an FP32 conversion before softmax
arithmetic is not observable because Triton already promotes FP16 there.

## Improvements, 2026-09-30

An evaluation of the repository after the fourth review and the follow-up fixes,
then the gaps the repository itself recorded: operators that could not train, the
one operator that lost to PyTorch, and measurements that did not say how busy the
GPU was. Details and numbers are in the [case studies](../CASE_STUDIES.md).

| Improvement | What changed | Measured result and remaining limit |
| --- | --- | --- |
| Training support | Softmax and residual RMSNorm gained Triton backward kernels for every row plan; `library.py` registers them with autograd, and `ops.softmax` and `ops.residual_rmsnorm` route gradient-tracking inputs through those custom ops. The RMSNorm backward recomputes each row's inverse RMS and sums the weight gradient in FP32 in a fixed order, without atomics | Gradients match FP64 autograd in all dtypes, with transposed and broadcast upstream gradients and strided inputs; `opcheck` passes its autograd tests and a compiled training step matches eager. A forward+backward step ran 1.6-2.9x faster than PyTorch for softmax and 4.1-10.2x faster than the RMSNorm composition (2.8-8.6x faster than `F.rms_norm`). Eager training through the Python custom ops costs a few hundred microseconds of host time per step; compile the model to remove it. `add`, `row_sum` and `matmul` stay forward-only, and double backward is unsupported |
| Few wide rows | `_row_plan` splits softmax (and RMSNorm above width 16384) across programs when there are fewer rows than SMs: one kernel writes per-chunk statistics, the next merges them and writes its chunk | Softmax 4 x 32769 went from 0.83x to 2.93x of PyTorch's speed; RMSNorm 4 x 32769 from 1.21x to 5.95x against the eager composition. The rule is conservative: at 128 x 32769, above the SM count, splitting also measured faster but still loops |
| Contention on record | Each benchmark report stores nvidia-smi's GPU utilization before and after the run and warns above 10%; SUMMARY shows it | Desktop applications kept the GPU 9-25% busy before these runs |
| Git lock left behind | `revision()` ran `git status`, which takes `.git/index.lock` to refresh the index; on a busy machine a status killed by its timeout left the lock, and the next commit failed. It now passes `--no-optional-locks` (read-only), with a unit test | Found when a test run left a stale lock |
| Evidence for torch.compile's wide-row results | `scripts/inspect_inductor.py` lists the kernels Inductor generates | At 64 x 131072, Inductor splits each RMSNorm row into four pieces across three launches and softmax into five launches with two separate passes for maximum and sum, having disabled online softmax |
| Stricter docs test | A quoted timing must match a current result; an archived median counts only on a line that links to the archive | No quoted number relied on the archive |

## Follow-up improvements, 2026-09-30 (evening)

The four next steps the previous round recorded: a better rule for few wide rows, a cheaper RMSNorm
weight gradient, backward for the remaining operators, and cleaner numbers. One of them ended as a
negative result, and one finding did not explain what it was expected to. Details and numbers are in the
[case studies](../CASE_STUDIES.md).

| Improvement | What changed | Measured result and remaining limit |
| --- | --- | --- |
| Backward for every operator | `add`, `row_sum` and `matmul` gained backward passes (`matmul_backward` is public). GEMM's two gradients are one strided kernel that reads the transposed operand through strides, with its own autotuning buckets, computing only the gradients that are needed and using no atomics. The RMSNorm weight gradient's final sum and cast became a Triton kernel, so a training step is four kernels instead of five. `examples/train_tiny.py` trains a small model with all five operators beside a plain-PyTorch twin | Gradients match FP64 autograd for dense, transposed, broadcast and large-mean upstream gradients, every GEMM configuration and ragged or empty shapes; a compiled training step through all five operators matches eager. The GEMM training step ran at 0.92-1.03x of cuBLAS; the RMSNorm step at 32 x 127 went from 6.076 to 4.301 us. The tiny model's loss falls from 2.6064 to 0.3957 against 0.3952 for the twin. Double backward through the kernels is unsupported, and the RMSNorm weight gradient is computed even when the weight is frozen |
| Chunk rule for looped rows | Wide rows that loop used 2048-wide chunks whenever there were two or more rows per SM. A grid of 332 row-count and width cells (`scripts/row_plan_grid.py --full`, run twice) showed that was wrong for wide rows. The rule is now width-aware: 4096-wide chunks for softmax and RMSNorm rows at most 28672 wide from one row per SM, 8192-wide chunks otherwise | Mean regret against the best plan per shape fell from 12.3% to 5.4%, and cells more than 10% off the best from 102 to 51 of 332. The split rule stayed as it was: thresholds fitted on one run of the grid gained -0.2-0.3 points on the other, no more than the runs differ from each other (`scripts/row_plan_fit.py`). The grid times forward softmax and RMSNorm only; row sum and the backward kernels inherit the rule |
| GPU clocks and cleaner numbers | A kernel timed in the first few hundred milliseconds after an idle pause can run several times slower (`scripts/gpu_clock_probe.py`). The benchmark, the workshop benchmark, the sweeps and the CUDA program now warm the GPU first; reports record the warm-up and the load; the regeneration waited for a quiet GPU before each stage; standalone CUDA timing has its own multi-run script | The probe recorded 25.34 us for a kernel that takes 2.45 us once warm, after a 20 s pause. Against an archived sweep from before the warm-up, 31 of 36 Triton/PyTorch ratios moved by less than 10%, so the sweeps were not affected and clocks do not explain their run-to-run spread. That spread remains: 29 of 36 comparisons agree within 10% over three runs, and kernels of about a microsecond move most. Load from other programs is the larger effect; closing them still helps |
| Independent review of this round | A read-only review of the new code found defects in the harness, the CLI and the tests, fixed with regression tests: `warm_gpu` cached inference tensors and failed when called outside inference mode; the first cuBLAS call consumed the warm-up burst; `--op training --dtype float32` crashed mid-run once the GEMM suite joined the group; the burst counter accumulated across runs; `matmul_backward` accepted mismatched inner dimensions; `matmul` saved both operands although each gradient reads one; and 32-bit index arithmetic could wrap a few elements below 2^31 (the limit now keeps 65536 of headroom). Test weaknesses fixed: an assertion that could not fail; softmax-backward tests that could not see a dropped row dot product when the upstream gradient had zero mean; a compiled-versus-eager tolerance 50 times looser than needed; and a test that passed if only one GEMM configuration ran | A copy of the softmax backward kernels with a lost partial dot product now fails the tests (the constant, broadcast and large-mean upstream gradients catch it in FP16, the FP32 tolerance in FP32). Gradient tolerances came from a probe over 300 random inputs (`scripts/tolerance_probe.py`): the worst error uses 39% of any tolerance |
| Evidence layout | The sweeps taken before the warm-up and the five-run stability log are archived beside the kernel each measured. `RowPlanTests` pins the plan rules, and a test checks the grid script's scoring | Every quoted timing still matches a current or archived median, and every result matches the source it measured. The full suite ran 69 tests: OK (skipped=2). |

Not fixed: double backward through the Triton kernels; a frozen RMSNorm weight still gets its gradient;
the grid does not time the backward kernels; and the shared GPU's noise. The standalone CUDA program's
medians moved by more than a factor of two between runs while other programs used the GPU, so the
CUDA case study quotes only the gap to the serial baseline.
