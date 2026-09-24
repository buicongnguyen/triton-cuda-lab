# I3. Fuse a GEMM epilogue and change tile scheduling

**Prerequisite:** beginner GEMM and masked tensor loads. **Time:** 3-4 sessions.
**Edit:** [fused_gemm.py](../exercises/fused_gemm.py).
**Goal:** implement `ReLU(A @ B + bias)` with a precisely defined rounding point.

An *epilogue* is the work after the dot product: here, adding one bias per output
column and replacing negative values with zero. Keep the accumulator in FP32,
do the epilogue, then convert to the input dtype only at the output store.

For A=`[[1,2],[3,4]]`, B=`[[1,-1],[2,1]]`, bias=`[-6,1]`:

```text
A@B        = [[5,1], [11,1]]
+ bias     = [[-1,2], [5,2]]
ReLU       = [[0,2], [5,2]]
```

The bias broadcasts over M, not K. Load it by the output column indices.

## Tiles and grouped ownership

Use BM in `{16,32,64}`, BN=64 and BK=32. A program owns BM-by-BN outputs and
loops over K. Mask A's M/K edges, B's K/N edges, and output M/N edges separately.

Use a one-dimensional grid. Suppose there are 3 M-tiles and 2 N-tiles, grouped
two M-tiles at a time. The `(m_tile,n_tile)` sequence is:

```text
pid:    0      1      2      3      4      5
tile: (0,0) (1,0) (0,1) (1,1) (2,0) (2,1)
```

The final group has only one M-tile. Compute `actual_group=min(GROUP, tiles_m-first_m)`
before taking the within-group remainder and quotient. Using GROUP unconditionally
duplicates or loses outputs at the end. GROUP=1 corresponds to row-major tiles.

Grouping changes which tiles may reuse cached B data. It does not change the
mathematical product and does not guarantee the GPU executes blocks in order.

## Implement in stages

1. Write a fixed-tile GEMM with GROUP=1. Verify ragged M/N/K dimensions.
2. Fuse the FP32 bias/ReLU epilogue. For K=0, the zero accumulator still needs the
   epilogue: output is `ReLU(bias)` repeated over M, not necessarily all zeros.
3. Add grouped PID mapping. Test an incomplete final group.
4. Sweep GROUP=1/4/8 and BM=16/32/64 while keeping BN/BK fixed.

```bash
python -m learning.workshops.check fused_gemm
python -m learning.workshops.benchmark --op fused_gemm --output results/local/i3.json
```

## Why the baseline uses FP32

`(A @ B + bias).relu()` with half inputs normally rounds the matmul output before
bias addition. That differs from the fused contract. The correctness reference is
FP64 GEMM+bias+ReLU rounded at the end. The timed PyTorch expression converts
inputs to FP32, disables TF32, runs the epilogue and finally casts the output.

This matches the intended rounding stage within tolerance, but has different
compute/cast costs from a tuned half-precision fused cuBLASLt operation. A large
ratio against this expression is evidence of fusion/precision choices, not proof
that this small kernel beats the best library implementation.

**Experiment:** explain why a tiny ragged GEMM can respond differently to grouping
than a 512-cubed GEMM. Record a losing configuration too; choose the best measured
configuration per shape only when you also report the search space.

**Done when:** derive the six-program mapping above, pass all tails/zero-K cases,
identify the rounding stage, and present the grouping experiment without a cache
hit-rate claim unless counters were actually measured.

Reference: [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
Next: [online normalization](04_online_normalizer.md).
