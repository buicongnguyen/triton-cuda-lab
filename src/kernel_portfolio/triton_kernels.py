"""Teaching kernels; algorithm references are listed in docs/REFERENCES.md.

Each program owns its output region. All reductions accumulate in FP32.

Specialize what a model fixes, keep runtime what a request varies. Row width and
GEMM N/K (hidden sizes, weight shapes) are tl.constexpr: one compile per value, and
measurably faster when a width does not fill its power-of-two block. Element counts,
row counts, row strides, GEMM M (tokens) and eps are runtime arguments, so a new
batch size reuses the compiled kernel. See docs/CASE_STUDIES.md for the measurements.

Every launcher takes an optional `wrap`. kernel_portfolio.library passes
torch.library.wrap_triton so torch.compile can trace the same launches.
"""

import functools

import torch
import triton
import triton.language as tl

# Rows up to this width load in one block; wider rows loop over chunks (see _chunking).
SINGLE_BLOCK_MAX = 8192


@functools.lru_cache(maxsize=None)
def _sm_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _chunking(rows, device):
    """(chunk width, warps) for looped row kernels, one program per row.

    With fewer rows than about two per SM, each program must cover more of its row
    at once to use the GPU; with many rows, smaller chunks keep more programs resident.
    Measured on an RTX 4080 SUPER; see docs/CASE_STUDIES.md. Never above
    SINGLE_BLOCK_MAX, so a looped row always fills its first chunk.
    """
    if device.type != "cuda":  # Triton interpreter on CPU tensors
        return 2048, 4
    if rows >= 2 * _sm_count(device.index):
        return 2048, 4
    return 8192, 16


@triton.jit
def _add(X, Y, OUT, N, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset, offset < N, other=0).to(tl.float32)
    y = tl.load(Y + offset, offset < N, other=0).to(tl.float32)
    tl.store(OUT + offset, x + y, offset < N)


@triton.jit
def _row_sum(X, OUT, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * STRIDE + col, col < N, other=0).to(tl.float32)
    tl.store(OUT + row, tl.sum(x, axis=0))


@triton.jit
def _row_sum_looped(X, OUT, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    # One program per row walks the row in BLOCK-wide chunks; lanes keep partial sums.
    row = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, N, BLOCK):
        col = start + tl.arange(0, BLOCK)
        acc += tl.load(X + row * STRIDE + col, col < N, other=0).to(tl.float32)
    tl.store(OUT + row, tl.sum(acc, axis=0))


@triton.jit
def _softmax(X, OUT, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    # Negative infinity is neutral for max and yields zero after exp.
    x = tl.load(X + row * STRIDE + col, col < N, other=-float("inf")).to(tl.float32)
    numerator = tl.exp(x - tl.max(x, axis=0))
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(OUT + row * N + col, result, col < N)


@triton.jit
def _softmax_looped(X, OUT, STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    # Pass 1 keeps an online (max, sum) per lane (workshop A1's merge rule); pass 2
    # rereads the row and writes probabilities. A lane that has seen only -inf (tail
    # padding or a masked score) shifts by 0 instead of -inf, so it keeps sum 0 rather
    # than computing (-inf) - (-inf) = NaN. A fully masked row still yields NaN, as in
    # torch.softmax.
    row = tl.program_id(0)
    lane_max = tl.full((BLOCK,), -float("inf"), tl.float32)
    lane_sum = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, N, BLOCK):
        col = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * STRIDE + col, col < N, other=-float("inf")).to(tl.float32)
        new_max = tl.maximum(lane_max, x)
        shift = tl.where(new_max == -float("inf"), 0.0, new_max)
        lane_sum = lane_sum * tl.exp(lane_max - shift) + tl.exp(x - shift)
        lane_max = new_max
    row_max = tl.max(lane_max, axis=0)
    row_sum = tl.sum(lane_sum * tl.exp(lane_max - row_max), axis=0)
    for start in range(0, N, BLOCK):
        col = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * STRIDE + col, col < N, other=-float("inf")).to(tl.float32)
        tl.store(OUT + row * N + col, tl.exp(x - row_max) / row_sum, col < N)


@triton.jit
def _rmsnorm(
    X,
    R,
    W,
    OUT,
    SX,
    SR,
    N: tl.constexpr,
    EPS,
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
def _rmsnorm_looped(
    X,
    R,
    W,
    OUT,
    SX,
    SR,
    N: tl.constexpr,
    EPS,
    BLOCK: tl.constexpr,
):
    # Pass 1 accumulates the FP32 residual sum of squares; pass 2 recomputes z and scales it.
    row = tl.program_id(0)
    squares = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, N, BLOCK):
        col = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
        r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
        squares += (x + r) * (x + r)
    inv = tl.rsqrt(tl.sum(squares, axis=0) / N + EPS)
    for start in range(0, N, BLOCK):
        col = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
        r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
        w = tl.load(W + col, col < N, other=0).to(tl.float32)
        tl.store(OUT + row * N + col, (x + r) * inv * w, col < N)


# M_BUCKET only feeds the autotuning key; do_not_specialize avoids extra compiles for it.
@triton.jit(do_not_specialize=["M_BUCKET"])
def _matmul(
    A,
    B,
    OUT,
    M,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # A 1D grid of output tiles. GROUP_M=1 is plain row-major order: program p owns
    # tile (p // tiles_n, p % tiles_n). Larger groups walk GROUP_M tile rows down each
    # column before moving right, so neighbouring programs reuse the same B columns
    # while they are still in L2 (workshop I3 derives the mapping).
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    group_size = GROUP_M * tiles_n
    first_m = (pid // group_size) * GROUP_M
    rows_in_group = tl.minimum(tiles_m - first_m, GROUP_M)
    tile_m = first_m + (pid % group_size) % rows_in_group
    tile_n = (pid % group_size) // rows_in_group
    mi = tile_m * BM + tl.arange(0, BM)
    nj = tile_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        k = tile * BK + kk
        a = tl.load(A + mi[:, None] * K + k[None, :], (mi[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + k[:, None] * N + nj[None, :], (k[:, None] < K) & (nj[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(OUT + mi[:, None] * N + nj[None, :], acc, (mi[:, None] < M) & (nj[None, :] < N))


def _config(bm, bn, bk, group, warps, stages):
    return triton.Config(
        {"BM": bm, "BN": bn, "BK": bk, "GROUP_M": group}, num_warps=warps, num_stages=stages
    )


# Tune once per power-of-two bucket of M, not once per exact token count. Configs that
# exceed the GPU's shared memory are skipped by the autotuner.
_matmul_tuned = triton.autotune(
    configs=[
        _config(32, 64, 32, 1, 4, 2),
        _config(64, 64, 32, 8, 4, 3),
        _config(64, 128, 32, 8, 4, 4),
        _config(128, 64, 32, 8, 4, 4),
        _config(64, 64, 64, 8, 4, 3),
        _config(128, 128, 32, 8, 8, 3),
        _config(128, 128, 64, 8, 8, 2),
        _config(128, 256, 32, 8, 8, 3),
    ],
    key=["M_BUCKET", "N", "K"],
)(_matmul)

FIXED_MATMUL_CONFIG = {"BM": 32, "BN": 64, "BK": 32, "GROUP_M": 1}


def m_bucket(m):
    return triton.next_power_of_2(max(m, 16))


def _kernel(fn, wrap):
    return fn if wrap is None else wrap(fn)


def launch_add(x, y, out, block_size, wrap=None):
    return _kernel(_add, wrap)[(triton.cdiv(x.numel(), block_size),)](
        x, y, out, x.numel(), block_size, num_warps=4
    )


def launch_row_sum(x, out, wrap=None):
    rows, n = x.shape
    if n <= SINGLE_BLOCK_MAX:
        return _kernel(_row_sum, wrap)[(rows,)](
            x, out, x.stride(0), n, triton.next_power_of_2(n), num_warps=4
        )
    chunk, warps = _chunking(rows, x.device)
    return _kernel(_row_sum_looped, wrap)[(rows,)](x, out, x.stride(0), n, chunk, num_warps=warps)


def launch_softmax(x, out, num_warps, wrap=None):
    rows, n = x.shape
    if n <= SINGLE_BLOCK_MAX:
        return _kernel(_softmax, wrap)[(rows,)](
            x, out, x.stride(0), n, triton.next_power_of_2(n), num_warps=num_warps
        )
    # num_warps is the single-block knob; looped rows use the measured chunking instead.
    chunk, warps = _chunking(rows, x.device)
    return _kernel(_softmax_looped, wrap)[(rows,)](x, out, x.stride(0), n, chunk, num_warps=warps)


def launch_rmsnorm(x, residual, weight, out, eps, wrap=None):
    rows, n = x.shape
    looped = n > SINGLE_BLOCK_MAX
    kernel = _kernel(_rmsnorm_looped if looped else _rmsnorm, wrap)
    chunk, warps = _chunking(rows, x.device) if looped else (triton.next_power_of_2(n), 4)
    return kernel[(rows,)](
        x,
        residual,
        weight,
        out,
        x.stride(0),
        residual.stride(0),
        n,
        eps,
        chunk,
        num_warps=warps,
    )


def launch_matmul(a, b, out, autotune, wrap=None):
    m, k = a.shape
    n = b.shape[1]

    def grid(meta):
        return (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)

    if autotune:
        return _kernel(_matmul_tuned, wrap)[grid](a, b, out, m, m_bucket(m), n, k)
    return _kernel(_matmul, wrap)[grid](
        a, b, out, m, m_bucket(m), n, k, **FIXED_MATMUL_CONFIG, num_warps=4, num_stages=2
    )
