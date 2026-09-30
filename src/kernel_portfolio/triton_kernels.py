"""Teaching kernels; algorithm references are listed in docs/REFERENCES.md.

Each program owns its output region. All reductions accumulate in FP32.

Specialize what a model fixes, keep runtime what a request varies. Row width and
GEMM N/K (hidden sizes, weight shapes) are tl.constexpr: one compile per value, and
measurably faster when a width does not fill its power-of-two block. Element counts,
row counts, row strides, GEMM M (tokens) and eps are runtime arguments, so a new
batch size reuses the compiled kernel. See docs/CASE_STUDIES.md for the measurements.

A row kernel covers a row in one of three ways (see _row_plan): one block, a loop
over chunks in one program, or chunks split across programs when there are too few
rows to fill the GPU. Softmax and residual RMSNorm also have backward kernels, which
kernel_portfolio.library registers with autograd.

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


# Rows at most this wide loop in 4096-wide chunks once there are as many rows as SMs.
# 28672 = 7 x 4096: the grid in scripts/row_plan_grid.py cannot tell it from 32000, and
# from 50257 on the widest chunk wins. The evidence is in docs/CASE_STUDIES.md.
NARROW_ROW_WIDTH = 28672


def _chunking(rows, n, device):
    """(chunk width, warps) for looped row kernels, one program per row.

    A program that walks a long row wants many bytes in flight, which the widest chunk
    and the most warps give. Smaller chunks paid off only for narrow rows with enough rows
    to fill the GPU with smaller programs. The previous rule used small chunks for every
    shape with many rows, which cost wide rows dearly. scripts/row_plan_grid.py times every
    plan over a grid of row counts and widths and scores both rules against the best plan
    per shape; docs/CASE_STUDIES.md reports the result.
    Never above SINGLE_BLOCK_MAX, so a looped row always fills its first chunk.
    """
    if device.type != "cuda":  # Triton interpreter on CPU tensors
        return 2048, 4
    if n <= NARROW_ROW_WIDTH and rows >= _sm_count(device.index):
        return 4096, 8
    return 8192, 16


def _split_below(device):
    """Row counts below this split each wide row across programs.

    One program per row cannot fill the GPU with fewer rows than SMs. The Triton
    interpreter on CPU tensors has no SMs; 2 there makes a single row split and larger
    counts loop, so CPU tests reach both paths.
    """
    return _sm_count(device.index) if device.type == "cuda" else 2


def _row_plan(op, rows, n, device):
    """How to cover each row: ("block", width, None), ("looped", chunk, warps) or
    ("split", chunk, warps).

    Rows up to SINGLE_BLOCK_MAX fit one block (the caller picks the warps). Wider rows
    with fewer rows than SMs are split into chunks across programs: two launches, but
    every SM gets work. That won clearly for softmax and RMSNorm when the rows were far
    fewer than the SMs and was about even near that count; RMSNorm only gains above
    width 16384. Row sum gained too little to pay for the second launch. Otherwise
    one program loops over its row (_chunking). See docs/CASE_STUDIES.md.
    """
    if n <= SINGLE_BLOCK_MAX:
        return "block", triton.next_power_of_2(n), None
    split_above = {"softmax": SINGLE_BLOCK_MAX, "rmsnorm": 2 * SINGLE_BLOCK_MAX}.get(op)
    if split_above is not None and n > split_above and rows < _split_below(device):
        return ("split", 4096, 8) if n > 65536 else ("split", 2048, 4)
    return ("looped", *_chunking(rows, n, device))


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
def _softmax_split_stats(X, PM, PS, STRIDE, N: tl.constexpr, CHUNK: tl.constexpr):
    # Program (row, part) summarizes one chunk: its maximum and its sum of exponentials
    # relative to that maximum. A chunk that is all -inf keeps maximum -inf and sum 0.
    row, part = tl.program_id(0), tl.program_id(1)
    parts = (N + CHUNK - 1) // CHUNK
    col = part * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + row * STRIDE + col, col < N, other=-float("inf")).to(tl.float32)
    maximum = tl.max(x, axis=0)
    shift = tl.where(maximum == -float("inf"), 0.0, maximum)
    tl.store(PM + row * parts + part, maximum)
    tl.store(PS + row * parts + part, tl.sum(tl.exp(x - shift), axis=0))


@triton.jit
def _softmax_split_normalize(
    X, PM, PS, OUT, STRIDE, N: tl.constexpr, CHUNK: tl.constexpr, PARTS_BLOCK: tl.constexpr
):
    # Every program merges its row's partial statistics itself (workshop A1's rule), so
    # no separate merge kernel runs, then writes its own chunk.
    row, part = tl.program_id(0), tl.program_id(1)
    parts = (N + CHUNK - 1) // CHUNK
    p = tl.arange(0, PARTS_BLOCK)
    maxima = tl.load(PM + row * parts + p, p < parts, other=-float("inf"))
    sums = tl.load(PS + row * parts + p, p < parts, other=0.0)
    row_max = tl.max(maxima, axis=0)
    row_sum = tl.sum(sums * tl.exp(maxima - row_max), axis=0)
    col = part * CHUNK + tl.arange(0, CHUNK)
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


@triton.jit
def _rmsnorm_split_squares(X, R, P, SX, SR, N: tl.constexpr, CHUNK: tl.constexpr):
    # Program (row, part) writes its chunk's FP32 sum of squared residuals.
    row, part = tl.program_id(0), tl.program_id(1)
    parts = (N + CHUNK - 1) // CHUNK
    col = part * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
    r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
    tl.store(P + row * parts + part, tl.sum((x + r) * (x + r), axis=0))


@triton.jit
def _rmsnorm_split_scale(
    X,
    R,
    W,
    P,
    OUT,
    SX,
    SR,
    N: tl.constexpr,
    EPS,
    CHUNK: tl.constexpr,
    PARTS_BLOCK: tl.constexpr,
):
    # Every program adds its row's partial sums itself, then scales its own chunk.
    row, part = tl.program_id(0), tl.program_id(1)
    parts = (N + CHUNK - 1) // CHUNK
    p = tl.arange(0, PARTS_BLOCK)
    inv = tl.rsqrt(tl.sum(tl.load(P + row * parts + p, p < parts, other=0.0), axis=0) / N + EPS)
    col = part * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
    r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    tl.store(OUT + row * N + col, (x + r) * inv * w, col < N)


# Backward kernels. The upstream gradient DY can have any strides (autograd passes
# expanded or transposed gradients), so it takes a row and a column stride; Triton
# specializes a column stride of 1 into contiguous loads. Y, the saved softmax output,
# is contiguous because the forward allocated it.


@triton.jit
def _softmax_bwd(Y, DY, DX, SDY0, SDY1, N: tl.constexpr, BLOCK: tl.constexpr):
    # dx = y * (dy - sum(y * dy)): one FP32 reduction, then an elementwise update.
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    y = tl.load(Y + row * N + col, col < N, other=0).to(tl.float32)
    dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
    dot = tl.sum(y * dy, axis=0)
    tl.store(DX + row * N + col, y * (dy - dot), col < N)


@triton.jit
def _softmax_bwd_looped(Y, DY, DX, SDY0, SDY1, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, N, BLOCK):
        col = start + tl.arange(0, BLOCK)
        y = tl.load(Y + row * N + col, col < N, other=0).to(tl.float32)
        dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
        acc += y * dy
    dot = tl.sum(acc, axis=0)
    for start in range(0, N, BLOCK):
        col = start + tl.arange(0, BLOCK)
        y = tl.load(Y + row * N + col, col < N, other=0).to(tl.float32)
        dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
        tl.store(DX + row * N + col, y * (dy - dot), col < N)


@triton.jit
def _softmax_bwd_split_dots(Y, DY, P, SDY0, SDY1, N: tl.constexpr, CHUNK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    parts = (N + CHUNK - 1) // CHUNK
    col = part * CHUNK + tl.arange(0, CHUNK)
    y = tl.load(Y + row * N + col, col < N, other=0).to(tl.float32)
    dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
    tl.store(P + row * parts + part, tl.sum(y * dy, axis=0))


@triton.jit
def _softmax_bwd_split_apply(
    Y, DY, DX, P, SDY0, SDY1, N: tl.constexpr, CHUNK: tl.constexpr, PARTS_BLOCK: tl.constexpr
):
    row, part = tl.program_id(0), tl.program_id(1)
    parts = (N + CHUNK - 1) // CHUNK
    p = tl.arange(0, PARTS_BLOCK)
    dot = tl.sum(tl.load(P + row * parts + p, p < parts, other=0.0), axis=0)
    col = part * CHUNK + tl.arange(0, CHUNK)
    y = tl.load(Y + row * N + col, col < N, other=0).to(tl.float32)
    dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
    tl.store(DX + row * N + col, y * (dy - dot), col < N)


# RMSNorm backward. With z = x + r (FP32), inv = rsqrt(mean(z^2) + eps) and
# out = z * inv * w, the input gradient for both x and r is
#   dz = inv * (g - z * inv^2 * mean(g * z)),  where g = dy * w,
# and the weight gradient is dw = sum over rows of dy * z * inv. The row kernels
# recompute inv rather than saving it from the forward pass, and store it for the
# weight kernel.


@triton.jit
def _rmsnorm_bwd(
    X,
    R,
    W,
    DY,
    DX,
    INV,
    SX,
    SR,
    SDY0,
    SDY1,
    N: tl.constexpr,
    EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
    r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
    z = x + r
    g = dy * w
    inv = tl.rsqrt(tl.sum(z * z, axis=0) / N + EPS)
    scale = tl.sum(g * z, axis=0) / N * inv * inv
    tl.store(DX + row * N + col, inv * (g - z * scale), col < N)
    tl.store(INV + row, inv)


@triton.jit
def _rmsnorm_bwd_looped(
    X,
    R,
    W,
    DY,
    DX,
    INV,
    SX,
    SR,
    SDY0,
    SDY1,
    N: tl.constexpr,
    EPS,
    BLOCK: tl.constexpr,
):
    # Pass 1 accumulates both row sums (z^2 and g*z); pass 2 recomputes z and g.
    row = tl.program_id(0)
    squares = tl.zeros((BLOCK,), tl.float32)
    products = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, N, BLOCK):
        col = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
        r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
        w = tl.load(W + col, col < N, other=0).to(tl.float32)
        dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
        squares += (x + r) * (x + r)
        products += dy * w * (x + r)
    inv = tl.rsqrt(tl.sum(squares, axis=0) / N + EPS)
    scale = tl.sum(products, axis=0) / N * inv * inv
    for start in range(0, N, BLOCK):
        col = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
        r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
        w = tl.load(W + col, col < N, other=0).to(tl.float32)
        dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
        tl.store(DX + row * N + col, inv * (dy * w - (x + r) * scale), col < N)
    tl.store(INV + row, inv)


@triton.jit
def _rmsnorm_bwd_split_sums(
    X, R, W, DY, P, SX, SR, SDY0, SDY1, N: tl.constexpr, CHUNK: tl.constexpr
):
    # Program (row, part) writes its chunk's two partial sums, z^2 and g*z.
    row, part = tl.program_id(0), tl.program_id(1)
    parts = (N + CHUNK - 1) // CHUNK
    col = part * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
    r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
    z = x + r
    tl.store(P + (row * parts + part) * 2, tl.sum(z * z, axis=0))
    tl.store(P + (row * parts + part) * 2 + 1, tl.sum(dy * w * z, axis=0))


@triton.jit
def _rmsnorm_bwd_split_apply(
    X,
    R,
    W,
    DY,
    DX,
    INV,
    P,
    SX,
    SR,
    SDY0,
    SDY1,
    N: tl.constexpr,
    EPS,
    CHUNK: tl.constexpr,
    PARTS_BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    parts = (N + CHUNK - 1) // CHUNK
    p = tl.arange(0, PARTS_BLOCK)
    squares = tl.sum(tl.load(P + (row * parts + p) * 2, p < parts, other=0.0), axis=0)
    products = tl.sum(tl.load(P + (row * parts + p) * 2 + 1, p < parts, other=0.0), axis=0)
    inv = tl.rsqrt(squares / N + EPS)
    scale = products / N * inv * inv
    col = part * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
    r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
    tl.store(DX + row * N + col, inv * (dy * w - (x + r) * scale), col < N)
    if part == 0:
        tl.store(INV + row, inv)


@triton.jit
def _rmsnorm_bwd_weight(
    X,
    R,
    DY,
    INV,
    PW,
    SX,
    SR,
    SDY0,
    SDY1,
    ROWS,
    GROUPS,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # dw needs a sum over rows, which row programs cannot share without atomics.
    # Program (group, block) sums dy * z * inv over rows group, group + GROUPS, ...
    # for one block of columns into its own FP32 partial row; the launcher adds the
    # GROUPS partial rows. Fixed order, so the result is deterministic.
    group, block = tl.program_id(0), tl.program_id(1)
    col = block * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for row in range(group, ROWS, GROUPS):
        x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
        r = tl.load(R + row * SR + col, col < N, other=0).to(tl.float32)
        dy = tl.load(DY + row * SDY0 + col * SDY1, col < N, other=0).to(tl.float32)
        acc += dy * (x + r) * tl.load(INV + row)
    tl.store(PW + group * N + col, acc, col < N)


@triton.jit
def _rmsnorm_bwd_weight_finish(
    PW, DW, GROUPS, N: tl.constexpr, BLOCK: tl.constexpr, GROUP_TILE: tl.constexpr
):
    # Adds the GROUPS partial rows in a fixed order (tiles of GROUP_TILE rows, then one
    # reduction over the tile axis) and rounds once to the weight's dtype. It replaces
    # PyTorch's sum and cast, two more launches, and stays deterministic.
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, GROUP_TILE)
    acc = tl.zeros((GROUP_TILE, BLOCK), tl.float32)
    for start in range(0, GROUPS, GROUP_TILE):
        g = start + lane
        offsets = g[:, None] * N + col[None, :]
        acc += tl.load(PW + offsets, (g[:, None] < GROUPS) & (col[None, :] < N), other=0)
    tl.store(DW + col, tl.sum(acc, axis=0).to(DW.dtype.element_ty), col < N)


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


@triton.jit
def _tile_coords(pid, M, N, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
    # The same grouped program-to-tile mapping as _matmul (workshop I3 derives it).
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    group_size = GROUP_M * tiles_n
    first_m = (pid // group_size) * GROUP_M
    rows_in_group = tl.minimum(tiles_m - first_m, GROUP_M)
    return first_m + (pid % group_size) % rows_in_group, (pid % group_size) // rows_in_group


# The backward pass of C = A @ B needs two products with transposed operands:
#   dA = dC @ B^T   (rows: tokens M; reduction: N; output columns: K)
#   dB = A^T @ dC   (rows: K; reduction: tokens M; output columns: N)
# Both are OUT = X @ Y with X and Y read through explicit strides, so a transposed
# operand is a view, never a copy. Tokens vary per request, so the row count and the
# reduction length are runtime arguments; only the output width, a model dimension,
# is compile-time. Their buckets feed the autotuning key.
@triton.jit(do_not_specialize=["M_BUCKET", "K_BUCKET"])
def _matmul_strided(
    X,
    Y,
    OUT,
    M,
    M_BUCKET,
    K,
    K_BUCKET,
    SXM,
    SXK,
    SYK,
    SYN,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile_m, tile_n = _tile_coords(tl.program_id(0), M, N, BM, BN, GROUP_M)
    mi = tile_m * BM + tl.arange(0, BM)
    nj = tile_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        k = tile * BK + kk
        x = tl.load(
            X + mi[:, None] * SXM + k[None, :] * SXK, (mi[:, None] < M) & (k[None, :] < K), other=0
        )
        y = tl.load(
            Y + k[:, None] * SYK + nj[None, :] * SYN, (k[:, None] < K) & (nj[None, :] < N), other=0
        )
        acc = tl.dot(x, y, acc)
    tl.store(OUT + mi[:, None] * N + nj[None, :], acc, (mi[:, None] < M) & (nj[None, :] < N))


def _config(bm, bn, bk, group, warps, stages):
    return triton.Config(
        {"BM": bm, "BN": bn, "BK": bk, "GROUP_M": group}, num_warps=warps, num_stages=stages
    )


def _matmul_configs():
    """Fresh Config objects per autotuner. Configs that exceed the GPU's shared memory
    are skipped by the autotuner."""
    return [
        _config(32, 64, 32, 1, 4, 2),
        _config(64, 64, 32, 8, 4, 3),
        _config(64, 128, 32, 8, 4, 4),
        _config(128, 64, 32, 8, 4, 4),
        _config(64, 64, 64, 8, 4, 3),
        _config(128, 128, 32, 8, 8, 3),
        _config(128, 128, 64, 8, 8, 2),
        _config(128, 256, 32, 8, 8, 3),
    ]


# Tune once per power-of-two bucket of M, not once per exact token count. The two
# backward products tune separately: their shapes differ from the forward's.
_matmul_tuned = triton.autotune(configs=_matmul_configs(), key=["M_BUCKET", "N", "K"])(_matmul)
_matmul_grad_a_tuned = triton.autotune(
    configs=_matmul_configs(), key=["M_BUCKET", "N", "K_BUCKET"]
)(_matmul_strided)
_matmul_grad_b_tuned = triton.autotune(
    configs=_matmul_configs(), key=["M_BUCKET", "N", "K_BUCKET"]
)(_matmul_strided)

FIXED_MATMUL_CONFIG = {"BM": 32, "BN": 64, "BK": 32, "GROUP_M": 1}


def m_bucket(m):
    return triton.next_power_of_2(max(m, 16))


def _kernel(fn, wrap):
    return fn if wrap is None else wrap(fn)


def _partials(rows, n, chunk, per_part=1, device=None):
    """FP32 scratch for split kernels: per_part values for each (row, chunk)."""
    parts = triton.cdiv(n, chunk)
    return torch.empty((rows, parts * per_part), device=device, dtype=torch.float32), parts


def launch_add(x, y, out, block_size, wrap=None):
    return _kernel(_add, wrap)[(triton.cdiv(x.numel(), block_size),)](
        x, y, out, x.numel(), block_size, num_warps=4
    )


def launch_row_sum(x, out, wrap=None):
    rows, n = x.shape
    path, block, warps = _row_plan("row_sum", rows, n, x.device)
    if path == "block":
        return _kernel(_row_sum, wrap)[(rows,)](x, out, x.stride(0), n, block, num_warps=4)
    return _kernel(_row_sum_looped, wrap)[(rows,)](x, out, x.stride(0), n, block, num_warps=warps)


def launch_softmax(x, out, num_warps, wrap=None):
    rows, n = x.shape
    path, block, warps = _row_plan("softmax", rows, n, x.device)
    # num_warps is the single-block knob; looped and split rows use measured choices.
    if path == "block":
        return _kernel(_softmax, wrap)[(rows,)](x, out, x.stride(0), n, block, num_warps=num_warps)
    if path == "looped":
        return _kernel(_softmax_looped, wrap)[(rows,)](
            x, out, x.stride(0), n, block, num_warps=warps
        )
    maxima, parts = _partials(rows, n, block, device=x.device)
    sums = torch.empty_like(maxima)
    grid = (rows, parts)
    _kernel(_softmax_split_stats, wrap)[grid](
        x, maxima, sums, x.stride(0), n, block, num_warps=warps
    )
    return _kernel(_softmax_split_normalize, wrap)[grid](
        x, maxima, sums, out, x.stride(0), n, block, triton.next_power_of_2(parts), num_warps=warps
    )


def launch_rmsnorm(x, residual, weight, out, eps, wrap=None):
    rows, n = x.shape
    path, block, warps = _row_plan("rmsnorm", rows, n, x.device)
    strides = (x.stride(0), residual.stride(0))
    if path == "split":
        squares, parts = _partials(rows, n, block, device=x.device)
        grid = (rows, parts)
        _kernel(_rmsnorm_split_squares, wrap)[grid](
            x, residual, squares, *strides, n, block, num_warps=warps
        )
        return _kernel(_rmsnorm_split_scale, wrap)[grid](
            x,
            residual,
            weight,
            squares,
            out,
            *strides,
            n,
            eps,
            block,
            triton.next_power_of_2(parts),
            num_warps=warps,
        )
    kernel = _kernel(_rmsnorm_looped if path == "looped" else _rmsnorm, wrap)
    return kernel[(rows,)](x, residual, weight, out, *strides, n, eps, block, num_warps=warps or 4)


def launch_softmax_backward(y, grad, dx, wrap=None):
    """dx = y * (grad - sum(y * grad)) for the contiguous saved output y."""
    rows, n = y.shape
    path, block, warps = _row_plan("softmax", rows, n, y.device)
    strides = (grad.stride(0), grad.stride(1))
    if path == "block":
        return _kernel(_softmax_bwd, wrap)[(rows,)](y, grad, dx, *strides, n, block, num_warps=4)
    if path == "looped":
        return _kernel(_softmax_bwd_looped, wrap)[(rows,)](
            y, grad, dx, *strides, n, block, num_warps=warps
        )
    dots, parts = _partials(rows, n, block, device=y.device)
    grid = (rows, parts)
    _kernel(_softmax_bwd_split_dots, wrap)[grid](y, grad, dots, *strides, n, block, num_warps=warps)
    return _kernel(_softmax_bwd_split_apply, wrap)[grid](
        y, grad, dx, dots, *strides, n, block, triton.next_power_of_2(parts), num_warps=warps
    )


def _weight_groups(rows, blocks, device):
    """Row groups for the dw kernel: about four programs per SM in total, and never
    more groups than rows. The partial buffer is groups x width FP32."""
    target = 4 * (_sm_count(device.index) if device.type == "cuda" else 1)
    return max(1, min(rows, triton.cdiv(target, blocks)))


def launch_rmsnorm_backward(x, residual, weight, grad, dx, dweight, eps, wrap=None):
    """Writes dx (the gradient of both x and residual) and dweight."""
    rows, n = x.shape
    path, block, warps = _row_plan("rmsnorm", rows, n, x.device)
    inv = torch.empty((rows,), device=x.device, dtype=torch.float32)
    strides = (x.stride(0), residual.stride(0), grad.stride(0), grad.stride(1))
    if path == "split":
        sums, parts = _partials(rows, n, block, per_part=2, device=x.device)
        grid = (rows, parts)
        _kernel(_rmsnorm_bwd_split_sums, wrap)[grid](
            x, residual, weight, grad, sums, *strides, n, block, num_warps=warps
        )
        _kernel(_rmsnorm_bwd_split_apply, wrap)[grid](
            x,
            residual,
            weight,
            grad,
            dx,
            inv,
            sums,
            *strides,
            n,
            eps,
            block,
            triton.next_power_of_2(parts),
            num_warps=warps,
        )
    else:
        kernel = _kernel(_rmsnorm_bwd_looped if path == "looped" else _rmsnorm_bwd, wrap)
        kernel[(rows,)](
            x, residual, weight, grad, dx, inv, *strides, n, eps, block, num_warps=warps or 4
        )
    width_block = min(1024, triton.next_power_of_2(n))
    blocks = triton.cdiv(n, width_block)
    groups = _weight_groups(rows, blocks, x.device)
    partial = torch.empty((groups, n), device=x.device, dtype=torch.float32)
    _kernel(_rmsnorm_bwd_weight, wrap)[(groups, blocks)](
        x, residual, grad, inv, partial, *strides, rows, groups, n, width_block, num_warps=4
    )
    finish_block = min(128, triton.next_power_of_2(n))
    _kernel(_rmsnorm_bwd_weight_finish, wrap)[(triton.cdiv(n, finish_block),)](
        partial, dweight, groups, n, finish_block, 32, num_warps=4
    )


def _launch_matmul_strided(tuned, x, y, out, m, n, kdim, x_strides, y_strides, autotune, wrap):
    """out (m x n, contiguous) = x (m x kdim) @ y (kdim x n), each read through strides."""

    def grid(meta):
        return (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)

    args = (x, y, out, m, m_bucket(m), kdim, m_bucket(kdim), *x_strides, *y_strides, n)
    if autotune:
        return _kernel(tuned, wrap)[grid](*args)
    return _kernel(_matmul_strided, wrap)[grid](
        *args, **FIXED_MATMUL_CONFIG, num_warps=4, num_stages=2
    )


def launch_matmul_grad_a(grad, b, da, autotune, wrap=None):
    """da (M x K) = grad (M x N, any strides) @ b.T, for contiguous b (K x N)."""
    m, n = grad.shape
    k = b.shape[0]
    return _launch_matmul_strided(
        _matmul_grad_a_tuned, grad, b, da, m, k, n, grad.stride(), b.t().stride(), autotune, wrap
    )


def launch_matmul_grad_b(grad, a, db, autotune, wrap=None):
    """db (K x N) = a.T @ grad, for contiguous a (M x K) and grad (M x N, any strides)."""
    m, k = a.shape
    n = grad.shape[1]
    return _launch_matmul_strided(
        _matmul_grad_b_tuned, a, grad, db, k, n, m, a.t().stride(), grad.stride(), autotune, wrap
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
