# Intermediate and advanced workshop validation

Executed **2026-09-23** and last re-executed **2026-09-25** on the NVIDIA RTX 4080 SUPER
under Ubuntu WSL2. The [second review](../../docs/records/REVIEW.md#second-review-2026-09-24)
made kernel sizes and strides runtime arguments in every workshop solution; the 2026-09-25
run repeats every check and measurement after the portfolio kernels gained wide-row support.
Python 3.12.13, PyTorch 2.11.0+cu130, Triton 3.6.0, driver 595.97.
This records execution of the separate reference solutions, not learner completion.

## Correctness and review

| Check | Observed result | Evidence |
| --- | --- | --- |
| Online normalizer, standard library only | 11 named checks passed under `python -S` | CPU execution; also repeated at the start of [checks.log](checks.log) |
| Five GPU reference implementations | 120 named checks passed, actual GPU execution | [checks.log](checks.log), 131 including the CPU workshop |
| Learning checker regression suite | 15 tests passed under `python -S`, including six workshop-runner tests | [checker-tests.log](../learning/checker-tests.log) |
| Portfolio regressions | 28 of 32 passed; the two-GPU test and three interpreter-only tests skipped | [portfolio-regression.log](portfolio-regression.log) |
| All benchmark variants | Correctness gate passed before timing | Errors and raw samples in the JSON below |
| Static checks | Ruff passed across src, tests, scripts, examples and learning | Local Ruff execution |
| Provenance | Benchmark source hashes match the current sources; all measurements re-run on 2026-09-25 | `source_sha256` in the raw reports |

Checks cover ragged dimensions, empty outputs, all supported dtypes, selected
strided/broadcast layouts, option variants, invalid metadata, input preservation,
fresh outputs, and non-default streams. Backward is checked against FP64 autograd
and a central finite difference. Attention is checked against dense FP64 math,
including future-value perturbations that must not affect earlier causal outputs.

The [pre-implementation review](../../docs/records/INTERMEDIATE_ADVANCED_PLAN.md) was
followed by this assistant self-review; it was not an independent human review.

| Review concern | Resolution / remaining scope |
| --- | --- |
| Learners could accidentally validate answers as their own | Starter paths remain separate; solution mode is explicitly labeled; TODO/unavailable outcomes return nonzero |
| Intermediate-only selection could pass without running anything | Empty CPU selection returns 3 with instructions to include GPU |
| Both operands in backward may have different strides | Separate y/g strides; both-strided and 8-warp case added |
| Fused GEMM can round too early or mis-handle K=0 | Epilogue acts on FP32 accumulator; FP64 oracle; zero-K and ragged final-group cases |
| Empty online states can produce NaN | Explicit identity handling, including empty+empty |
| Split softmax could race across stages | Ordered launches on the current stream; private per-call scratch; no global spin barriers |
| Causal attention can read future values or mishandle a changing maximum | Key mask includes diagonal; both numerator and denominator rescaled; perturbation and large-logit cases |
| Timing could reward a numerically different baseline | Baseline contracts recorded explicitly, particularly the FP32 GEMM epilogue comparison |
| Simplified attention could be mistaken for production FlashAttention | Single-head bounded scope, approximate PV precision, no backward/dropout/batching, full future-tile traversal documented |

No new Compute Sanitizer or hardware-counter trace was collected for these
workshops. The original native CUDA sanitizer results apply only to that earlier
implementation. Correctness comparisons and non-default-stream checks are not
a proof that every possible execution is race-free. The available single GPU
does not validate multi-GPU behavior.

## Measured examples

Raw data: [graph-fp16.json](graph-fp16.json), [events-fp16.json](events-fp16.json).
Readable command output: [graph.log](graph.log), [events.log](events.log).
Each run used FP16 inputs, **7 samples of 20 calls**, five warmup calls per variant,
and randomized variant order within each sample. Compilation happened before
timing. Both runs covered ten operation/shape cases and 38 variant measurements
each (76 total), with every individual sample preserved.

The table selects the **same candidate configuration in both timing modes**.
Latency is the median in microseconds; speedup is baseline/candidate. GEMM shape
order is M,N,K; attention is N,D and causal=True. Strided inputs are transposes.

| Operator / shape | Candidate | Graph baseline / candidate (us) | Graph ratio | Event ratio |
| --- | --- | ---: | ---: | ---: |
| Strided softmax, 256x257 | direct_4w | 3.942 / 2.509 | 1.57x | 0.61x |
| Strided softmax, 1024x1024 | direct_4w | 19.098 / 15.974 | 1.20x | 0.67x |
| Softmax VJP, 256x1025 | fused_vjp | 13.722 / 1.994 | 6.88x | 2.28x |
| Softmax VJP, 1024x4097 | fused_vjp | 126.618 / 14.778 | 8.57x | 4.32x |
| GEMM+bias+ReLU, 127x65x33 | group_4 | 12.237 / 2.509 | 4.88x | 2.13x |
| GEMM+bias+ReLU, 512x512x512 | group_4 | 30.821 / 7.117 | 4.33x | 1.81x |
| Split softmax, 4x32769 | chunk_1024 | 8.858 / 3.994 | 2.22x | 0.23x |
| Split softmax, 64x131072 | chunk_4096 | 25.907 / 19.915 | 1.30x | 0.48x |
| Attention, 129x32 | stream_bn64 | 6.912 / 3.320 | 2.08x | 0.32x |
| Attention, 512x64 | stream_bn64 | 8.704 / 8.960 | 0.97x | 1.00x |

Interpretation matters:

- **Strides:** direct loads beat copy-then-kernel in the graph runs on these two
  shapes. The copy is included in timing. This does not prove a universal layout rule.
- **Backward:** the baseline is an eager FP32 VJP expression including casts and
  several launches. It is not PyTorch's complete training step or a compiled VJP.
- **GEMM:** the baseline performs FP32 GEMM/bias/ReLU and then casts, with TF32
  disabled. The candidate uses half-precision dot inputs and an FP32 epilogue.
  These ratios **do not establish a win over fused half-precision cuBLASLt**.
  GROUP=1/4/8 were nearly tied at 512-cubed; the dispersion does not support a
  cache-efficiency claim or a convincing winner among those group sizes.
- **Split softmax:** chunk_256 at 64x131072 took 54.016 us in graph mode, **0.48x**
  the PyTorch baseline. More programs were not automatically better. In event
  timing all three split candidates lost on both shapes; Python dispatch and
  three launches change the result materially.
- **Attention:** the primary baseline is PyTorch SDPA with automatic backend
  selection. BN=64 won the small graph case but did not establish a win on the
  larger case; its 0.97x result is close enough to treat as roughly tied. The small
  event case lost (0.32x) and the larger one tied (1.00x). BN=32 was clearly slower
  in the larger graph case at 0.71x.

These are local microbenchmarks on a desktop GPU, not isolated-server guarantees
or model-level latency results. Min/max dispersion is recorded. No clock lock,
peak-memory measurement, DRAM-traffic counter or SDPA-backend trace is claimed.
The attention score-storage figures in the lessons are analytic calculations.

## Reproduce

From the repo root in the configured environment (or with `PYTHONPATH=src`):

```bash
python -S -m learning.workshops.check online_normalizer --solution
python -S -m unittest discover -s learning/tests -v
python -m learning.workshops.check all --include-gpu --solution
KERNEL_REQUIRE_GPU=1 python -m unittest discover -s tests -v
python -m learning.workshops.benchmark --solution --timing graph --output results/local/workshops-graph.json
python -m learning.workshops.benchmark --solution --timing events --output results/local/workshops-events.json
```

Remove `--solution` to check or benchmark your own exercise files. Keep new results
under `results/local/` until you intentionally add your own documented experiment.
