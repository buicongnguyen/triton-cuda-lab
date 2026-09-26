"""Lesson 8, part B. One program per row. Complete row_sum_kernel, then remove the guard."""

import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(X, OUT, STRIDE, N, BLOCK: tl.constexpr):
    # 1. row = program_id(0); columns = arange(0, BLOCK).
    # 2. Address each element with X + row * STRIDE + columns.
    # 3. Load with columns < N, zero padding, then convert to float32.
    # 4. Reduce with tl.sum(..., axis=0), and store one value at OUT + row.
    pass


def row_sum(x):
    """Checker supplies 2D CUDA rows with contiguous columns, 1 <= width <= 8192."""
    raise NotImplementedError("Fill row_sum_kernel, then remove this line")
    rows, width = x.shape
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows:
        with torch.cuda.device(x.device):
            row_sum_kernel[(rows,)](x, out, x.stride(0), width, triton.next_power_of_2(width))
    return out
