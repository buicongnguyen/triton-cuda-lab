# Intermediate and advanced workshop validation

Executed **2026-09-23** and last re-executed **2026-09-30** on the NVIDIA RTX 4080 SUPER
under Ubuntu WSL2. The [second review](../../docs/records/REVIEW.md#second-review-2026-09-24)
made kernel sizes and strides runtime arguments in every workshop solution, and the
[fourth review](../../docs/records/REVIEW.md#fourth-review-2026-09-26) strengthened the
fused-GEMM checker. The 2026-09-30 run repeats every check and measurement after the
[improvements](../../docs/records/REVIEW.md#improvements-2026-09-30) to the portfolio
kernels, which the workshop runner shares.
Python 3.12.13, PyTorch 2.11.0+cu130, Triton 3.6.0, driver 595.97.
This records execution of the separate reference solutions, not learner completion.

## Correctness and review

| Check | Observed result | Evidence |
| --- | --- | --- |
| Online normalizer, standard library only | 11 named checks passed under `python -S` | CPU execution; also repeated at the start of [checks.log](checks.log) |
| Five GPU reference implementations | 122 named checks passed, actual GPU execution | [checks.log](checks.log), 133 including the CPU workshop |
| Learning checker regression suite | 15 tests passed under `python -S`, including six workshop-runner tests | [checker-tests.log](../learning/checker-tests.log) |
| Portfolio regressions | 48 of 49 passed; the two-GPU test skipped, and the interpreter-only class, skipped at setup, runs separately (5 passed) | [portfolio-regression.log](portfolio-regression.log) |
| All benchmark variants | Correctness gate passed before timing | Errors and raw samples in the JSON below |
| Static checks | Ruff passed across src, tests, scripts, examples and learning | Local Ruff execution |
| Provenance | Reports use schema version 2: the measured solutions and their shared dependencies match the current sources, checked by `tests/test_docs.py`; unused learner exercises are not fingerprinted. All measurements re-run on 2026-09-30 | `source_sha256` in the raw reports |

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
| Fused GEMM can round too early or mis-handle K=0 | Epilogue acts on FP32 accumulator; FP64 oracle; zero-K and ragged final-group cases. The checker first accepted an FP16 accumulator; an exact FP16 case (2048 + 1 + 1 = 2050 only with one rounding) and a K = 1024 case now reject it, and a GPU test checks that |
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
timing, and the GPU was warmed before each case. Both runs covered ten operation/shape
cases and 38 variant measurements each (76 total), with every individual sample preserved.

The table selects the **same candidate configuration in both timing modes**.
Latency is the median in microseconds; speedup is baseline/candidate. GEMM shape
order is M,N,K; attention is N,D and causal=True. Strided inputs are transposes.

| Operator / shape | Candidate | Graph baseline / candidate (us) | Graph ratio | Event ratio |
| --- | --- | ---: | ---: | ---: |
| Strided softmax, 256x257 | direct_4w | 3.774 / 2.304 | 1.64x | 0.62x |
| Strided softmax, 1024x1024 | direct_4w | 17.357 / 14.434 | 1.20x | 0.86x |
| Softmax VJP, 256x1025 | fused_vjp | 12.698 / 1.792 | 7.09x | 2.19x |
| Softmax VJP, 1024x4097 | fused_vjp | 123.187 / 13.414 | 9.18x | 2.21x |
| GEMM+bias+ReLU, 127x65x33 | group_4 | 11.264 / 2.304 | 4.89x | 2.74x |
| GEMM+bias+ReLU, 512x512x512 | group_4 | 28.774 / 6.502 | 4.43x | 2.06x |
| Split softmax, 4x32769 | chunk_1024 | 8.074 / 4.915 | 1.64x | 0.14x |
| Split softmax, 64x131072 | chunk_4096 | 23.501 / 18.586 | 1.26x | 0.27x |
| Attention, 129x32 | stream_bn64 | 6.758 / 3.366 | 2.01x | 0.72x |
| Attention, 512x64 | stream_bn64 | 8.499 / 8.235 | 1.03x | 1.01x |

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
- **Split softmax:** chunk_256 at 64x131072 took 49.152 us in graph mode, **0.48x**
  the PyTorch baseline. More programs were not automatically better. In event
  timing all three split candidates lost on both shapes; Python dispatch and
  three launches change the result materially.
- **Attention:** the primary baseline is PyTorch SDPA with automatic backend
  selection. BN=64 won the small graph case (2.01x) but did not establish a win on
  the larger case; its 1.03x result is close enough to treat as roughly tied. BN=32
  was clearly slower in the larger graph case at 0.76x. In event timing BN=64
  measured 0.72x and 1.01x here; earlier runs measured 0.32x and 1.00x
  (2026-09-25), then 1.09x and 1.37x, and 0.65x and 0.83x (2026-09-26), and 0.77x and
  0.94x earlier on 2026-09-30, before the GPU warm-up existed. Event ratios
  that include Python dispatch of several launches move with host load and do not
  rank these implementations.

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
