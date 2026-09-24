# 9. Read the CUDA version without getting lost

**Time:** 60–90 minutes. **Prerequisite:** your three Triton exercises.
**Read:** [cuda/kernels.cu](../../cuda/kernels.cu). This lesson uses the existing,
tested CUDA examples; you do not need to build a second framework integration.

## Start with vector addition

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) out[i] = x[i] + y[i];
```

Each CUDA thread computes one scalar index. `threadIdx.x` is the position within
its block; `blockDim.x` is the number of threads per block; `blockIdx.x` chooses
the block. The launch `<<<ceil(n/256), 256>>>` selects the number of blocks and
256 threads per block. For n=257, only thread 0 of block 1 accesses valid data.

Compare this to Triton's block of offsets. The address formula has the same
purpose, but CUDA explicitly gives the individual thread its scalar work.

## Memory spaces in the softmax example

| Location | Who uses it? | Example in the CUDA source |
| --- | --- | --- |
| Global memory | Accessible by all blocks | Input `x` and output `out` |
| Registers/local thread state | A thread's own working values | Its running maximum and sum |
| Shared memory | Threads of one block cooperate through it | `scratch[kWarps]`: one partial per warp, eight for 256 threads |

“Local thread state” describes ownership; the compiler may spill values to local
memory if there are not enough registers. Declaring a C++ scalar alone does not
prove it never spills. The profiler/compiler output is the evidence.

## Follow one row through the parallel softmax

1. A 256-thread block owns one row. Thread t processes columns t, t+256,
   t+512, and so on until the row ends. Neighboring threads therefore access
   neighboring columns at each loop iteration.
2. Each thread forms a local maximum over its columns. A thread with no columns
   starts with negative infinity, the neutral maximum value.
3. The 256 threads form eight 32-thread warps. Warp shuffles combine each warp's
   values. Lane 0 of each warp writes one partial value to `scratch[warp]`.
4. `__syncthreads()` ensures the block has written all eight partials before the
   first warp reads them. That warp reduces eight values into the row maximum.
5. The result is shared with the block. Another barrier ensures all readers are
   done before the scratch array is reused for the sum reduction.
6. Threads compute exponentials, reduce their sums by the same pattern, then
   write normalized outputs for the columns they own.

For thread ID 67: warp=67//32=2 and lane=67%32=3. It does not write the warp's
partial into shared memory because it is not lane 0.

## Why no thread returns early

Even when a row has only 33 columns, the kernel still launches all 256 threads.
Threads with no valid columns contribute neutral values and participate in every
block barrier. Returning early from some threads before a block-wide barrier
would break the intended synchronization. The code masks *work* rather than
removing participants from this cooperative algorithm.

The shuffle mask assumes every warp is fully present. The block size is one
constant, `kThreads = 256`; `kWarps`, the scratch size and both launch sites derive
from it, and a `static_assert` rejects a size that is not a whole number of warps.
Changing the block size therefore means changing that one line, then re-running
the correctness and sanitizer checks.

## Run and investigate

Build and run it from **PowerShell** at the repo root (details in
[SETUP.md](../../docs/SETUP.md#standalone-cuda-c)):

```powershell
.\scripts\build_cuda_windows.cmd
.\build\cuda_portfolio.exe --test-only
```

The second command should print `CUDA correctness passed (3 vector + 18 softmax cases)`. Use the sanitizer
commands in [PROFILING.md](../../docs/PROFILING.md) to check memory accesses and races.

The serial baseline assigns one thread to an entire row. It is easy to understand
but poorly parallelized across columns. Its long-row sum uses compensation after
testing exposed numerical drift. The parallel tree has a different summation
order. Read the [case study](../../docs/CASE_STUDIES.md#cuda-parallel-reduction-online-statistics-and-a-numerical-fix)
without treating a gain over this teaching baseline as a gain over a vendor library.

**Optional: the online variant.** `softmax_online` computes the row maximum and the
sum of exponentials in a single pass, merging `(max, sum)` pairs across threads and
warps with the rule from workshop A1. It reads each row twice instead of three
times, and rows whose width is a multiple of four use 16-byte `float4` loads. Compare
its `block_merge` with `block_reduce`: the structure is identical; only the combined
value is a pair.

**Exit questions:** Why eight shared partials? Why are barriers needed even though
each thread owns different output elements? Which memory accesses are contiguous?
What must change if you launch 128 threads instead of 256?
