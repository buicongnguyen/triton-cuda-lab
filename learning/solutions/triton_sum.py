"""Annotated reference for inputs satisfying the exercise's narrow contract."""

import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(X, OUT, STRIDE, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    # STRIDE locates the next physical row; N is the number of logical columns.
    values = tl.load(X + row * STRIDE + column, column < N, other=0).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(OUT + row, total)


def row_sum(x):
    rows, width = x.shape
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows:
        with torch.cuda.device(x.device):
            row_sum_kernel[(rows,)](x, out, x.stride(0), width, triton.next_power_of_2(width))
    return out
