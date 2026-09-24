"""Teaching kernels; algorithm references are listed in docs/REFERENCES.md.

Each program owns its output region. All reductions accumulate in FP32.
"""

import triton
import triton.language as tl


@triton.jit
def _add(X, Y, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset, offset < N, other=0).to(tl.float32)
    y = tl.load(Y + offset, offset < N, other=0).to(tl.float32)
    tl.store(OUT + offset, x + y, offset < N)


@triton.jit
def _row_sum(X, OUT, STRIDE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * STRIDE + col, col < N, other=0).to(tl.float32)
    tl.store(OUT + row, tl.sum(x, axis=0))


@triton.jit
def _softmax(X, OUT, STRIDE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    # Negative infinity is neutral for max and yields zero after exp.
    x = tl.load(X + row * STRIDE + col, col < N, other=-float("inf")).to(tl.float32)
    numerator = tl.exp(x - tl.max(x, axis=0))
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(OUT + row * N + col, result, col < N)


@triton.jit
def _rmsnorm(
    X,
    R,
    W,
    OUT,
    SX: tl.constexpr,
    SR: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
    r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    z = x + r
    inv = tl.rsqrt(tl.sum(z * z, axis=0) / N + EPS)
    tl.store(OUT + row * N + col, z * inv * w, col < N)


@triton.jit
def _matmul(
    A,
    B,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # A simple 2D grid makes tile ownership easy to explain in an interview.
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    nj = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for tile in range(triton.cdiv(K, BK)):
        k = tile * BK + kk
        a = tl.load(A + mi[:, None] * K + k[None, :], (mi[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + k[:, None] * N + nj[None, :], (k[:, None] < K) & (nj[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(OUT + mi[:, None] * N + nj[None, :], acc, (mi[:, None] < M) & (nj[None, :] < N))


_matmul_tuned = triton.autotune(
    configs=[
        triton.Config({"BM": 32, "BN": 64, "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)(_matmul)


def launch_add(x, y, out, block_size):
    return _add[(triton.cdiv(x.numel(), block_size),)](
        x, y, out, x.numel(), block_size, num_warps=4
    )


def launch_row_sum(x, out):
    return _row_sum[(x.shape[0],)](
        x, out, x.stride(0), x.shape[1], triton.next_power_of_2(x.shape[1]), num_warps=4
    )


def launch_softmax(x, out, num_warps):
    return _softmax[(x.shape[0],)](
        x, out, x.stride(0), x.shape[1], triton.next_power_of_2(x.shape[1]), num_warps=num_warps
    )


def launch_rmsnorm(x, residual, weight, out, eps):
    return _rmsnorm[(x.shape[0],)](
        x,
        residual,
        weight,
        out,
        x.stride(0),
        residual.stride(0),
        x.shape[1],
        eps,
        triton.next_power_of_2(x.shape[1]),
        num_warps=4,
    )


def launch_matmul(a, b, out, autotune):
    m, k = a.shape
    n = b.shape[1]

    def grid(meta):
        return triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"])

    if autotune:
        return _matmul_tuned[grid](a, b, out, m, n, k)
    return _matmul[grid](a, b, out, m, n, k, BM=32, BN=64, BK=32, num_warps=4, num_stages=2)
