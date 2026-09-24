# Reading references

Checked 2026-09-23. Books supply a learning framework; executable API details are
checked against upstream documentation. No paid book text or examples are vendored.
Only public publisher descriptions, contents and available excerpts were reviewed,
not the full paid books. The code in this repository is a compact teaching
implementation of the cited algorithms, with its own contracts, tests and experiments.

## Books

**[Open the downloaded previews and book readers](../references/books/README.md).**
The local files contain Manning's short Chapter 1 brief and O'Reilly's 14-page
table of contents. Full ebooks are not included; publisher access links are
provided in the index.

1. **Harshwardhan Fartale, _GPU Programming with Triton_**, Manning,
   ISBN 9781633434233.
   [Publisher](https://www.manning.com/books/gpu-programming-with-triton) ·
   [public welcome excerpt](https://livebook.manning.com/book/gpu-programming-with-triton).
   The publisher currently lists MEAP early access, **4 of 11 chapters available**,
   begun August 2026, with final publication estimated for spring 2027. Treat its
   scope as a developing roadmap, not a finished reference. Its block-programming,
   memory access, reductions, fusion and tiling topics guide labs 1–5. Advanced
   attention topics are follow-on reading; no unavailable chapter content is assumed.
2. **Chris Fregly, _AI Systems Performance Engineering_**, O'Reilly, 2025,
   ISBN 9798341627772.
   [Publisher and contents](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/).
   This is a wider systems book with a substantial Triton component. Read the
   profiling and benchmarking topics in chapter 13, and chapter 14,
   “PyTorch Compiler, OpenAI Triton, and XLA Backends,” for the compiler path,
   kernel launch tuning and autotuning. Use it alongside labs 5–6 and the
   benchmark methodology. It motivates measuring an entire pipeline as well as
   one kernel; this small repository only measures isolated forward operators.

## Free primary technical references

| Reference | Used for |
| --- | --- |
| [Triton installation](https://triton-lang.org/main/getting-started/installation.html) and [supported platforms](https://github.com/triton-lang/triton#compatibility) | Linux setup and hardware expectations |
| [Vector addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) | Program IDs, block offsets, masked tails |
| [Fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | Stable row reductions, fusion and memory traffic reasoning |
| [Matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | Tile reuse, FP32 accumulation and configuration search |
| [Fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | Tiled attention and online numerator/denominator rescaling in advanced workshop A3 |
| [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) | Mergeable max/sum state in workshops A1-A2 |
| [Layer normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) | Row normalization patterns; our RMSNorm omits mean subtraction and adds a residual |
| [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | Coalescing, memory hierarchy, occupancy and measurement |
| [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | Blocks, synchronization and warp primitives |
| [Triton debugging](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html) | Interpreter and device-side diagnostics |
| [Nsight Compute guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Interpreting counters and measurement limitations |

Reading order: block programming → memory and masks → reductions → fusion →
tiling → measurement and compiler inspection. Run a small experiment after each
topic, then describe the result without looking at the source.
