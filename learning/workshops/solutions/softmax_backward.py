"""I2: an explicit softmax VJP, not an autograd registration."""

import torch
import triton
import triton.language as tl

from kernel_portfolio import contracts as c
from learning.workshops.contracts import matrix, warps


@triton.jit
def _kernel(
    Y,
    G,
    DX,
    SY0,
    SY1,
    SG0,
    SG1,
    N,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    y = tl.load(Y + row * SY0 + col * SY1, col < N, other=0).to(tl.float32)
    g = tl.load(G + row * SG0 + col * SG1, col < N, other=0).to(tl.float32)
    dot = tl.sum(y * g, 0)
    tl.store(DX + row * N + col, y * (g - dot), col < N)


def backward(y, grad_output, *, num_warps=4):
    matrix(y, "y")
    matrix(grad_output, "grad_output")
    c.same(y, grad_output)
    warps(num_warps)
    dx = torch.empty(y.shape, device=y.device, dtype=y.dtype)
    if y.shape[0]:
        with torch.cuda.device(y.device):
            _kernel[(y.shape[0],)](
                y,
                grad_output,
                dx,
                *y.stride(),
                *grad_output.stride(),
                y.shape[1],
                triton.next_power_of_2(y.shape[1]),
                num_warps=num_warps,
            )
    return dx
