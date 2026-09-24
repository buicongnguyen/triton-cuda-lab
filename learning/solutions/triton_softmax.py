"""Annotated reference: last-dimension softmax, finite input, forward only."""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(X, OUT, STRIDE, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    values = tl.load(X + row * STRIDE + column, column < N, other=-float("inf"))
    values = values.to(tl.float32)
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    tl.store(OUT + row * N + column, numerator / denominator, column < N)


def softmax(x):
    rows, width = x.shape
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if rows:
        with torch.cuda.device(x.device):
            softmax_kernel[(rows,)](x, out, x.stride(0), width, triton.next_power_of_2(width))
    return out
