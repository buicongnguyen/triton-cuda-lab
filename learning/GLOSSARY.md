# Glossary

Short definitions for words used across the lessons, labs and case studies. Each
entry says where you first need it, so you can skip terms you have not reached yet.
Lesson numbers refer to the [beginner course](README.md).

## Execution model

| Term | Meaning | First needed |
| --- | --- | --- |
| Host / device | The CPU running Python / the GPU running kernels | Lesson 0 |
| Kernel | A function compiled to run on the GPU (`@triton.jit` or CUDA `__global__`) | Lesson 0 |
| Launch | Submitting a kernel with a grid: `kernel[grid](args)` in Triton, `kernel<<<blocks, threads>>>(args)` in CUDA | Lesson 0 |
| Grid | How many kernel instances run: Triton *programs* or CUDA *thread blocks* | Lesson 1 |
| Program (Triton) | One kernel instance. It processes a whole block of values; `tl.program_id(0)` says which one | Lesson 1 |
| Block (Triton) | The vector of values one program handles, e.g. `tl.arange(0, BLOCK)`. Not the same as a CUDA thread block | Lesson 1 |
| Thread / thread block (CUDA) | One scalar worker / a group of threads that can share memory and use `__syncthreads()` | Lesson 9 |
| Warp / lane | 32 NVIDIA threads that execute together / one thread's position (0-31) inside its warp | Lesson 9 |
| Asynchronous launch | Python returns before the GPU finishes, so a plain stopwatch measures submission, not work | Lesson 0 |
| Stream | An ordered GPU work queue. Kernels on one stream run in submission order | Workshop A2 |

## Memory and layout

| Term | Meaning | First needed |
| --- | --- | --- |
| Offset | Distance in **elements** (not bytes) from a tensor's first element | Lesson 1 |
| Mask / tail | Per-lane true/false saying whether a load or store may touch memory / the final, partly filled block | Lesson 1 |
| Stride | Elements to skip to move one step along a dimension. Shape alone cannot locate a row | Lesson 2 |
| Contiguous | Rows stored back to back with column stride 1, so the strides follow from the shape | Lesson 2 |
| View | A tensor sharing another tensor's storage with different shape/strides, e.g. a slice or transpose | Lesson 2 |
| Global memory (DRAM) | The GPU's main memory, 16 GiB on the RTX 4080 SUPER; about 736 GB/s peak | Lesson 4 |
| L2 cache | 64 MiB on-chip cache in front of DRAM. Data used again soon may come from L2, much faster than DRAM | Lesson 7 |
| Shared memory | Fast memory shared by the threads of one CUDA block, e.g. `scratch` in the CUDA softmax | Lesson 9 |
| Registers / spill | Each thread's fastest private storage / values moved to slower memory when registers run out | Lesson 9 |
| Coalescing | Neighboring threads reading neighboring addresses, so the hardware combines their accesses | Lesson 2 |
| Occupancy | How many warps an SM (streaming multiprocessor) keeps resident. Higher is not automatically faster | Lab 2 |

## Numbers and precision

| Term | Meaning | First needed |
| --- | --- | --- |
| dtype | Element format: FP32 (4 bytes), FP16 and BF16 (2 bytes; BF16 keeps FP32's range with less precision) | Lesson 0 |
| Accumulation dtype | The precision used while combining values. Here reductions use FP32 even for FP16 inputs | Lesson 3 |
| Reduction / identity | Combining many values into one (sum, max) / the padding value that changes nothing (0 for sum, -inf for max) | Lesson 3 |
| Stable softmax | Subtracting the row maximum before `exp` so large inputs cannot overflow | Lesson 4 |
| Oracle / reference | An independent, simpler computation (often in FP64) that a kernel's output is compared against | Lesson 4 |
| Tolerance (`atol`, `rtol`) | Allowed absolute and relative difference from the oracle; rounding makes exact equality unrealistic | Lesson 8 |
| Epsilon (`eps`) | Small positive number added before a square root so an all-zero row does not divide by zero | Lesson 5 |

## Compiling Triton

| Term | Meaning | First needed |
| --- | --- | --- |
| JIT compilation | Triton compiles a kernel the first time it is called with new settings. That first call can take a second or more | Lesson 8 |
| `tl.constexpr` | A value fixed at compile time. Each new value compiles a new kernel. Use it for tile sizes like `BLOCK`, and for sizes a model fixes, such as row width | Lesson 8 |
| Runtime argument | An ordinary integer or pointer argument. It can change per call without recompiling (sizes, strides) | Lesson 8 |
| Specialization | Compiling a version tuned to known values. Triton also specializes runtime ints equal to 1 or divisible by 16 | Lesson 8 |
| Autotuning | Timing several tile configurations on the first call for a shape and keeping the fastest | Lab 5 |
| Fusion | Doing several steps in one kernel so intermediates never go through global memory | Lesson 4 |
| Tensor Core / `mma.sync` | Matrix-multiply hardware units / the PTX instruction that uses them | Lesson 6 |
| TTIR, TTGIR, LLVM IR, PTX, SASS | Successive compiler stages, from Triton's own IR down to NVIDIA's final machine code | Lab 6 |

## Measuring

| Term | Meaning | First needed |
| --- | --- | --- |
| Speedup | Baseline time / candidate time. Above 1 means the candidate is faster | Lesson 7 |
| Median | Middle sample; less sensitive to one slow outlier than the mean | Lesson 7 |
| CUDA events | GPU-side timestamps; the elapsed time between two events measures device work | Lesson 7 |
| CUDA graph / graph replay | A recorded sequence of launches replayed with almost no Python cost. The default benchmark mode | Lesson 7 |
| Event timing mode | `--timing events`: ordinary Python calls, so host dispatch gaps are included | Lesson 7 |
| Warm vs cold cache | Warm: inputs likely already in L2 from the previous call. Cold (`--cache cold`): L2 flushed first, so data comes from DRAM | Lesson 7 |
| Logical GB/s | Minimum useful bytes / time. A model, not a hardware counter; above DRAM peak means the data came from cache | Lesson 7 |
| TFLOP/s | `2*M*N*K / seconds` for GEMM: trillions of floating-point operations per second | Lab 5 |
| Arithmetic intensity | Operations per byte moved. Low intensity means memory-bound | Lesson 6 |
| Amdahl's law | Speeding up part of a program only helps in proportion to that part's share of the runtime | Lesson 7 |
| Eager mode | Normal PyTorch: each operation launches its own kernel(s) immediately | Lesson 4 |
| `torch.compile` / Inductor | PyTorch's compiler; Inductor generates fused (often Triton) kernels. A strong baseline for fusion claims | Lesson 5 |
| cuBLAS / SDPA | NVIDIA's tuned matrix-multiply library / PyTorch's `scaled_dot_product_attention`, which selects a fused backend | Lesson 6 / Workshop A3 |

## Algorithms in the workshops

| Term | Meaning | First needed |
| --- | --- | --- |
| Jacobian / VJP | Matrix of all partial derivatives / vector-Jacobian product: the gradient a backward pass actually needs | Workshop I2 |
| Autograd | PyTorch's automatic differentiation. The portfolio's softmax and residual RMSNorm register Triton backward kernels with it; the other operators are forward-only | Workshop I2 |
| Epilogue | Work applied to a GEMM result before storing it, e.g. bias and ReLU | Workshop I3 |
| Grouped tile ordering | Launching output tiles so neighbors reuse cached inputs; `GROUP` in the fused GEMM | Workshop I3 |
| Online normalizer | Mergeable `(max, sum)` state that lets softmax statistics be computed chunk by chunk | Workshop A1 |
| Scratch | Temporary GPU buffers a multi-kernel operation allocates for intermediate results | Workshop A2 |
| Custom op | A function registered with PyTorch (`torch.library`) so `torch.compile` can trace it; see `kernel_portfolio/library.py` | [Setup](../docs/SETUP.md#use-the-operators-from-python) |
| Causal mask | Attention rule that a query may only see keys at the same or earlier positions | Workshop A3 |

## Tools

| Term | Meaning | First needed |
| --- | --- | --- |
| WSL | Windows Subsystem for Linux: Ubuntu running inside Windows. One of the two GPU options here | [Setup](../docs/SETUP.md) |
| `triton-windows` | Community build of Triton for native Windows; the PowerShell GPU option uses it | [Setup](../docs/SETUP.md) |
| Virtual environment | An isolated Python install with its own packages (`.venv-win` in PowerShell, `.venv` in WSL); activate it in each new terminal | [Setup](../docs/SETUP.md) |
| Triton interpreter | `TRITON_INTERPRET=1` runs Triton kernels on the CPU with NumPy, for debugging and CI | [Setup](../docs/SETUP.md) |
| Compute Sanitizer | NVIDIA tool: `memcheck` finds out-of-bounds accesses, `racecheck` finds shared-memory races | Lesson 9 |
| Nsight Systems / Compute | NVIDIA profilers: a timeline of launches / per-kernel hardware counters | [Profiling](../docs/PROFILING.md) |
