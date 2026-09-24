# Intermediate and advanced kernel workshops

You can now write a masked row reduction. The next step is to support real tensor
layouts, derive a backward pass, fuse a useful operation, and handle computations
too large to keep in one program. These six workshops provide that progression.

Each has a lesson, a starter, a separately implemented answer and named checks.
Every GPU starter begins with a commented outline: the kernel signature, which
arguments are compile-time constants, and the wrapper steps. Use it as scaffolding
or delete it for more challenge. Unfamiliar terms are in the
[glossary](../GLOSSARY.md#algorithms-in-the-workshops).
The checked answers are working code. Your starters are deliberately unfinished.
The [validation and measured results](../../results/workshops/VALIDATION.md) show
what was actually executed, including slow cases.

## Choose your level

| Level | Workshop | Prerequisite | Deliverable |
| --- | --- | --- | --- |
| Intermediate I1 | [Strided softmax](lessons/01_strided_softmax.md) | Beginner strides + Triton softmax | Correct transposes/slices; compare direct access with copying |
| Intermediate I2 | [Softmax backward](lessons/02_softmax_backward.md) | Softmax + basic derivatives | Explicit vector-Jacobian product, checked against autograd |
| Intermediate I3 | [GEMM with bias and ReLU](lessons/03_fused_gemm.md) | Beginner GEMM + reductions | Fused epilogue and a grouped tile sweep |
| Advanced A1 | [Online normalization](lessons/04_online_normalizer.md) | Stable softmax | CPU merge state that survives a changing maximum |
| Advanced A2 | [Wide-row split softmax](lessons/05_split_softmax.md) | A1 + masked Triton reductions | Three ordered kernels with per-call scratch |
| Advanced A3 | [Streaming attention](lessons/06_streaming_attention.md) | I3 + A1; A2 helpful | Tiled causal/noncausal attention without a full score tensor |

Budget 2-4 focused sessions for each intermediate workshop and 3-5 for each
advanced workshop. A session is 60-90 minutes. These are suggested study times,
not a deadline or a claim that completing six exercises makes a production expert.

## Run from the repo root

The CPU workshop needs only Python's standard library:

```bash
python -S -m learning.workshops.check --list
python -S -m learning.workshops.check online_normalizer --hint
# Edit learning/workshops/exercises/online_normalizer.py first:
python -S -m learning.workshops.check online_normalizer
```

The CPU workshop runs in any PowerShell or WSL terminal. GPU workshops run in the
GPU environment from the [setup guide](../../docs/SETUP.md), PowerShell or WSL;
activate it as described in [every new terminal](../../docs/SETUP.md#every-new-terminal). Check the
environment with the reference answer first; it should pass:

```bash
python -m learning.workshops.check strided_softmax --solution
```

Then work on your own file:

```bash
python -m learning.workshops.check strided_softmax --hint
python -m learning.workshops.check strided_softmax
python -m learning.workshops.check all --level intermediate --include-gpu
python -m learning.workshops.check all --level advanced --include-gpu
# Opt in to checking the supplied answers instead of your work:
python -m learning.workshops.check all --include-gpu --solution
```

`all` runs CPU workshops unless `--include-gpu` is given. A specifically named
GPU workshop always attempts GPU execution. No selected work or unavailable
dependencies return exit code 3; TODO returns 2; wrong answers return 1. The
runner never marks skipped work as complete and never overwrites starters.

## Measure after correctness passes

```bash
# Defaults to YOUR implementation; --solution explicitly selects reference code.
python -m learning.workshops.benchmark --op strided_softmax --output results/local/i1.json
python -m learning.workshops.benchmark --op all --solution --timing graph --output results/local/workshops-graph.json
python -m learning.workshops.benchmark --op all --solution --timing events --output results/local/workshops-events.json
```

Each operation has two fixed reproducible shapes. JSON includes raw samples,
errors against a double-precision oracle, configuration labels, environment and
source hashes. The main softmax backward baseline is an eager FP32 expression;
the attention baseline is PyTorch SDPA with automatic backend selection. GEMM
uses an FP32 expression to preserve the one-rounding epilogue contract. Read the
individual lesson before interpreting a speedup: these baselines differ.

Graph timing measures warmed captured execution. Event timing surrounds repeated
Python calls and can include GPU idle time caused by host dispatch. Neither is
end-to-end model latency. Allocation/copy costs inside a callable belong to that
variant; compilation is warmed up before measurement.

Use [the experiment worksheet and capstones](EXPERIMENTS.md) to record your work,
then compare with [the answer explanations](ANSWER_KEY.md). Reference results
validate the course; they are not evidence that you personally completed it.

## Scope and contracts

All GPU functions allocate fresh contiguous outputs on the input device. Inputs
must be NVIDIA CUDA tensors with supported floating dtypes and finite values, and
must not require gradients while autograd is enabled. Values must keep the stated FP32 intermediates finite. Wrappers
validate metadata but do not scan values, which would add extra GPU work. Tensor
address spans and allocated output element counts must fit signed 32-bit indexing.

| Function | Input and options | Output and limits |
| --- | --- | --- |
| `strided_softmax.softmax` | 2D FP32/FP16/BF16; nonnegative strides, including zero strides; `num_warps=4/8` | Same shape/dtype; 1-8192 columns; empty row count allowed |
| `softmax_backward.backward` | Matching 2D y/g; same dtype/device; each may have different strides; `num_warps=4/8` | VJP using the supplied y; same limits as I1; no autograd registration |
| `fused_gemm.matmul_bias_relu` | Contiguous A[M,K], B[K,N], bias[N]; FP16/BF16; same dtype/device; `group_m=1/4/8`, `tile_m=16/32/64` | FP32 accumulation + bias + ReLU, then one output cast; all zero dimensions allowed |
| `split_softmax.softmax` | Contiguous 2D FP32/FP16/BF16; `chunk_size=256/1024/4096` | Same shape/dtype; 1-131072 columns; empty row count allowed |
| `streaming_attention.attention` | Contiguous matching Q/K/V[N,D], FP16/BF16; `causal` bool, `block_n=32/64` | Same shape/dtype; N=0-2048, D=16/32/64; scale=1/sqrt(D), diagonal included when causal |

Attention is single-head self-attention, forward only, without dropout, padding
lengths, arbitrary masks, cross-attention or batching. P is converted to input
precision for the PV dot product. Causal masking currently traverses every key
tile, including future tiles: it teaches correctness, not an optimal causal
schedule. These limitations are explicit extension opportunities, not implemented features.

Back to the [beginner course](../README.md) or [portfolio](../../README.md).
