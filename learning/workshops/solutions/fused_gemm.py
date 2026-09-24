"""I3: tiled GEMM with grouped scheduling and a bias/ReLU epilogue."""

import torch
import triton
import triton.language as tl

from kernel_portfolio import contracts as c


@triton.jit
def _kernel(
    A,
    B,
    BIAS,
    OUT,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_m, tiles_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = pid // (GROUP * tiles_n)
    first_m = group * GROUP
    actual_group = tl.minimum(tiles_m - first_m, GROUP)
    within = pid % (GROUP * tiles_n)
    tile_m = first_m + within % actual_group
    tile_n = within // actual_group
    rows = tile_m * BM + tl.arange(0, BM)
    cols = tile_n * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + inner
        a = tl.load(
            A + rows[:, None] * K + kk[None, :], (rows[:, None] < M) & (kk[None, :] < K), other=0
        )
        b = tl.load(
            B + kk[:, None] * N + cols[None, :], (kk[:, None] < K) & (cols[None, :] < N), other=0
        )
        acc = tl.dot(a, b, acc)
    bias = tl.load(BIAS + cols, cols < N, other=0).to(tl.float32)
    # Bias and ReLU happen before the one output conversion.
    out = tl.maximum(acc + bias[None, :], 0)
    tl.store(
        OUT + rows[:, None] * N + cols[None, :], out, (rows[:, None] < M) & (cols[None, :] < N)
    )


def matmul_bias_relu(a, b, bias, *, group_m=4, tile_m=32):
    for value, name, ndim in ((a, "a", 2), (b, "b", 2), (bias, "bias", 1)):
        c.tensor(value, name, ndim=ndim)
        c.contiguous(value)
    c.same(a, b, shape=False)
    c.same(a, bias, shape=False)
    if a.dtype == torch.float32:
        raise TypeError("fused GEMM requires float16 or bfloat16")
    if a.shape[1] != b.shape[0] or bias.shape[0] != b.shape[1]:
        raise ValueError("Expected A[M,K], B[K,N], bias[N]")
    if group_m not in (1, 4, 8) or tile_m not in (16, 32, 64):
        raise ValueError("group_m must be 1/4/8 and tile_m must be 16/32/64")
    m, k = a.shape
    n = b.shape[1]
    if m * n >= 2**31:
        raise ValueError("Output exceeds signed 32-bit indexing")
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if m and n:
        with torch.cuda.device(a.device):
            _kernel[(triton.cdiv(m, tile_m) * triton.cdiv(n, 64),)](
                a,
                b,
                bias,
                out,
                m,
                n,
                k,
                tile_m,
                64,
                32,
                group_m,
                num_warps=4,
                num_stages=2,
            )
    return out
