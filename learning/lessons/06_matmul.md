# 6. Matrix multiplication: reuse the inputs

**Time:** 90 minutes, split into two sessions if needed.
**Prerequisite:** loops, multiplication, reductions and matrix shapes.
**Edit:** [matmul.py](../exercises/matmul.py).

## The three dimensions have different jobs

```text
A has shape [M,K]
B has shape [K,N]
C has shape [M,N]
C[row,col] = sum_inner(A[row,inner] * B[inner,col])
```

M chooses an output row. N chooses an output column. K is reduced away. You
cannot multiply arbitrary matrices: A's column count must match B's row count.

For `A=[[1,2,3],[4,5,6]]` and `B=[[7,8],[9,10],[11,12]]`:

```text
C[0,0] = 1*7 + 2*9  + 3*11 = 58
C[0,1] = 1*8 + 2*10 + 3*12 = 64
C[1,0] = 4*7 + 5*9  + 6*11 = 139
C[1,1] = 4*8 + 5*10 + 6*12 = 154
```

Implement the three-loop CPU version first. Allocate separate output rows:
`[[0.0] * N for _ in range(M)]`. Avoid `[[0.0] * N] * M`, which repeats references
to the same mutable list; changing one row would change every row.

## Tiles are small rectangles of work

One Triton program owns BM rows by BN columns of C. It walks through the
reduction dimension in chunks of BK. At each step:

```text
load A tile [BM,BK]
load B tile [BK,BN]
accumulator[BM,BN] += dot(A_tile, B_tile)
```

Use the toy example above with BK=2. C[0,0] receives 25 from the first K tile
(`1*7+2*9`) and 33 from the second (`3*11`). The accumulator must survive both
steps. Moving its initialization inside the K loop would keep only the last tile.

```bash
python -m learning.explain matmul
python -m learning.check matmul
```

`tile_ranges(7,3)` gives `(0,3)`, `(3,6)`, `(6,7)`. These are half-open intervals:
start is included; stop is excluded. A GPU block still uses its full tile shape,
but masks invalid rows, columns or K entries. Zero padding is neutral for dot products.

## Why tiling can improve throughput

Within one tile, each A element is reused across BN output columns; each B element
is reused across BM output rows. For BM=32, BN=64, BK=32 and two-byte inputs:

```text
input elements = 32*32 + 32*64 = 3072
input bytes    = 3072*2 = 6144
arithmetic     = 2*32*64*32 = 131072 FLOPs
ratio          = 131072/6144 ≈ 21.33 FLOPs per input byte
```

This is a tile-level reuse model, excluding output traffic and assuming ideal
reuse of the loaded tiles. It is not a measured GPU roofline point. Larger tiles
can improve reuse while consuming more registers/shared memory per program.

## Connect to the real implementation

Read `_matmul` in [triton_kernels.py](../../src/kernel_portfolio/triton_kernels.py):

- The grid is one-dimensional. With `GROUP_M=1` (the fixed configuration), program
  `p` owns tile `(p // tiles_n, p % tiles_n)`: plain row-major order. The autotuned
  configurations use `GROUP_M=8`, which walks down eight tile rows before moving
  right; workshop I3 derives that mapping.
- `mi` chooses the output rows; `nj` chooses output columns.
- `kk` represents positions within a K tile; `tile * BK + kk` moves along K.
- A uses address `row*K+inner`; B uses `inner*N+col` because inputs are contiguous.
- `acc` is FP32; inputs and final output are FP16/BF16.
- The final store masks both output dimensions.

`tl.dot` can lower to Tensor Core matrix instructions on supported inputs and
hardware. The saved [PTX](../../results/compiler/matmul.ptx) contains `mma.sync`.
That confirms this instruction path for the inspected configuration, not that
the implementation fully utilizes the hardware.

**GPU step (optional now).** Run this after lesson 8's setup:

```bash
kernel-bench --op matmul --dtype float16 --output results/local/my-gemm.json
```

The benchmark's custom `--shape` order is **M N K**, for example `--shape 127 255 65`.
Keep a shape where `torch.mm` is faster. Your first goal is understanding ownership
and reuse; beating a tuned vendor library is a later challenge.

**Exit check:** draw a 5-by-7 C tiled by BM=2, BN=3, with K=5 and BK=2. Count
programs and K steps and identify which dimensions need masks. Next: [measurement](07_measurement.md).
