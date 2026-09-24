"""A2: partial statistics -> merged statistics -> normalization, on one stream."""

import torch
import triton
import triton.language as tl

from learning.workshops.contracts import matrix


@triton.jit
def _partials(X, PM, PL, N, T, CHUNK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + row * N + col, col < N, other=-float("inf")).to(tl.float32)
    maximum = tl.max(x, 0)
    total = tl.sum(tl.exp(x - maximum), 0)
    tl.store(PM + row * T + part, maximum)
    tl.store(PL + row * T + part, total)


@triton.jit
def _merge(PM, PL, GM, GL, T, B: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, B)
    maxima = tl.load(PM + row * T + part, part < T, other=-float("inf"))
    totals = tl.load(PL + row * T + part, part < T, other=0)
    maximum = tl.max(maxima, 0)
    total = tl.sum(totals * tl.exp(maxima - maximum), 0)
    tl.store(GM + row, maximum)
    tl.store(GL + row, total)


@triton.jit
def _normalize(X, GM, GL, OUT, N, CHUNK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    maximum, total = tl.load(GM + row), tl.load(GL + row)
    tl.store(OUT + row * N + col, tl.exp(x - maximum) / total, col < N)


def softmax(x, *, chunk_size=1024):
    matrix(x, max_width=131072, contiguous=True)
    if chunk_size not in (256, 1024, 4096):
        raise ValueError("chunk_size must be 256, 1024, or 4096")
    rows, n = x.shape
    out = torch.empty_like(x)
    if not rows:
        return out
    parts = triton.cdiv(n, chunk_size)
    # Scratch is local to this call; independent calls/streams never share buffers.
    pm = torch.empty((rows, parts), device=x.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    gm = torch.empty((rows,), device=x.device, dtype=torch.float32)
    gl = torch.empty_like(gm)
    with torch.cuda.device(x.device):
        _partials[(rows, parts)](x, pm, pl, n, parts, chunk_size, num_warps=4)
        _merge[(rows,)](pm, pl, gm, gl, parts, triton.next_power_of_2(parts), num_warps=4)
        _normalize[(rows, parts)](x, gm, gl, out, n, chunk_size, num_warps=4)
    return out
