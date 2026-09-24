"""A3: single-head self-attention with a tiled online softmax recurrence."""

import math

import torch
import triton
import triton.language as tl

from kernel_portfolio import contracts as c


@triton.jit
def _kernel(
    Q,
    K,
    V,
    OUT,
    N,
    D: tl.constexpr,
    SCALE,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dim = tl.arange(0, D)
    keys = tl.arange(0, BN)
    q = tl.load(Q + rows[:, None] * D + dim[None, :], rows[:, None] < N, other=0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    for start in range(tl.cdiv(N, BN)):
        cols = start * BN + keys
        k = tl.load(K + cols[None, :] * D + dim[:, None], cols[None, :] < N, other=0)
        v = tl.load(V + cols[:, None] * D + dim[None, :], cols[:, None] < N, other=0)
        scores = tl.dot(q, k) * SCALE
        valid = cols[None, :] < N
        if CAUSAL:
            valid = valid & (cols[None, :] <= rows[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp(maximum - new_maximum)
        probabilities = tl.exp(scores - new_maximum[:, None])
        total = alpha * total + tl.sum(probabilities, 1)
        accumulator = accumulator * alpha[:, None]
        # Tensor-core PV uses input precision; the recurrence/accumulation stay FP32.
        accumulator = tl.dot(probabilities.to(q.dtype), v, accumulator)
        maximum = new_maximum
    tl.store(
        OUT + rows[:, None] * D + dim[None, :], accumulator / total[:, None], rows[:, None] < N
    )


def attention(q, k, v, *, causal=False, block_n=32):
    for value, name in ((q, "q"), (k, "k"), (v, "v")):
        c.tensor(value, name, ndim=2)
        c.contiguous(value)
    c.same(q, k)
    c.same(q, v)
    if q.dtype == torch.float32:
        raise TypeError("attention requires float16 or bfloat16")
    n, d = q.shape
    if n > 2048 or d not in (16, 32, 64):
        raise ValueError("Expected N in [0,2048], D in {16,32,64}")
    if not isinstance(causal, bool) or block_n not in (32, 64):
        raise ValueError("causal must be bool; block_n must be 32 or 64")
    out = torch.empty_like(q)
    if n:
        with torch.cuda.device(q.device):
            _kernel[(triton.cdiv(n, 16),)](
                q,
                k,
                v,
                out,
                n,
                d,
                1 / math.sqrt(d),
                causal,
                16,
                block_n,
                num_warps=4,
                num_stages=2,
            )
    return out
