# Eighteen sessions with concrete outputs

Plan for 45–90 minutes per session, three sessions per week. If a prerequisite is
new, repeat a session; the exit task matters more than the calendar. Keep your
answers and measurements in a copy of [JOURNAL.md](JOURNAL.md).

| Week/session | Work | Command or file | Produce before advancing |
| --- | --- | --- | --- |
| 1/1 | Read orientation and draw CPU→GPU flow | Lesson 0; `python -m learning.check --list` | Definitions of host, device, tensor, launch |
| 1/2 | Trace a partial block, then implement offsets | `python -m learning.check offsets` | Passing checks and N=17/BLOCK=8 diagram |
| 1/3 | Map slices and transposes to storage | `python -m learning.check strides` | Six physical indices for the exit question |
| 2/1 | Build the sum tree | `python -m learning.check reduction` | All levels for width 5 and an explanation of padding |
| 2/2 | Work softmax with three numbers | `python -m learning.explain softmax` | Hand-calculated max, exponentials, denominator |
| 2/3 | Implement and debug stable softmax | `python -m learning.check softmax` | Passing extreme-logit and shift-invariance checks |
| 3/1 | Work residual RMSNorm by hand | `python -m learning.check rmsnorm` | Per-row/per-column/per-element labels |
| 3/2 | Implement rectangular matrix multiplication | `python -m learning.check matmul` | Correct 2x3 @ 3x2 product, no aliased output rows |
| 3/3 | Trace K tiles and output edge masks | `python -m learning.explain matmul` | Diagram for M=5, N=7, K=5 and toy tiles |
| 4/1 | Check GPU environment; complete Triton add | `python -m learning.check triton_add` | Passing tails in FP32 and FP16 |
| 4/2 | Complete Triton row sum | `python -m learning.check triton_sum` | Passing strided-row case and dtype explanation |
| 4/3 | Complete Triton softmax | `python -m learning.check triton_softmax` | Passing full-row output and stability cases |
| 5/1 | Calculate timing units and Amdahl's law | `python -m learning.check measurement` | Correct bandwidth and overall-speedup example |
| 5/2 | Predict and benchmark softmax warp counts | `kernel-bench --op softmax --shape 1024 1024 --dtype float32` | Prediction, JSON path, observed medians |
| 5/3 | Compare graph replay and ordinary dispatch | Same command with `--timing events` and a different output file | Explanation of the changed measurement boundary |
| 6/1 | Compare eager and compiled RMSNorm | `kernel-bench --op rmsnorm --compile` | One carefully scoped result sentence |
| 6/2 | Read CUDA softmax and inspect GEMM PTX | `cuda/kernels.cu`; `python scripts/inspect_kernel.py --op matmul` | CUDA cooperation diagram and one identified PTX instruction |
| 6/3 | Reimplement one kernel from a blank file and explain it | [Journal](JOURNAL.md) | Ten-minute walkthrough and one honest unresolved question |

Notes on the order:

- **Lesson 7 comes in week 5, not between lessons 6 and 8.** Its CPU exercise is
  paired with the GPU benchmarks that use it (sessions 5/2 and 5/3), so you
  calculate speedups and GB/s right before measuring your own.
- **Session 4/1 starts with the one-time [GPU setup](../docs/SETUP.md)** (on Windows,
  `.\scripts\setup_windows.ps1`).
  If installation takes the whole session, that is normal. Finish `triton_add` in
  the next session and let the rest of the schedule slide by one.
- **Weeks 1–3 need no GPU.** The optional "GPU step" commands in lessons 5–7 belong
  to weeks 5–6.

## What each part of the schedule demonstrates

| Question | Session evidence |
| --- | --- |
| Can you reason about memory correctly? | Offset ownership, tail masks, row gaps, strides |
| Can you derive an operation? | Stable softmax and RMSNorm handwritten calculations |
| Can you write and debug a kernel? | Your three edited Triton exercises and named failing/passing checks |
| Can you explain hardware tradeoffs? | CUDA reduction cooperation, GEMM input reuse, measured warp-count experiment |
| Can you make a credible performance claim? | Same-shape/dtype baselines, graph versus dispatch, saved raw results |

The beginner path prepares foundations; compiler implementation and distributed
collectives remain additional topics.
