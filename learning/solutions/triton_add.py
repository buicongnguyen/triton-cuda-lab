"""Annotated reference. The production-style input checks are in kernel_portfolio.ops."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(X, Y, OUT, N, BLOCK: tl.constexpr):
    # One program describes BLOCK values; it is not one scalar CUDA thread.
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    x = tl.load(X + index, mask=valid, other=0).to(tl.float32)
    y = tl.load(Y + index, mask=valid, other=0).to(tl.float32)
    tl.store(OUT + index, x + y, mask=valid)


def add(x, y):
    out = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), 256)
    return out
