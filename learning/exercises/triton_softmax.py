"""Lesson 8, part C. Build from your row_sum kernel; two reductions share one loaded row."""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(X, OUT, STRIDE, N, BLOCK: tl.constexpr):
    # 1. Load a row in float32 using its actual STRIDE.
    # 2. Padded columns must use -float("inf"), not zero.
    # 3. Shift by tl.max, exponentiate, divide by tl.sum.
    # 4. Output is contiguous: OUT + row * N + column, with a store mask.
    pass


def softmax(x):
    """Checker supplies finite, detached CUDA input: contiguous columns, 1 <= width <= 8192."""
    raise NotImplementedError("Fill softmax_kernel, then remove this line")
    rows, width = x.shape
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if rows:
        with torch.cuda.device(x.device):
            softmax_kernel[(rows,)](x, out, x.stride(0), width, triton.next_power_of_2(width))
    return out
