"""I1: stride-aware loads, contiguous stores. See lessons/01_strided_softmax.md."""

import torch
import triton
import triton.language as tl

from learning.workshops.contracts import matrix, warps


@triton.jit
def _kernel(X, Y, SR, SC, N, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * SR + col * SC, col < N, other=-float("inf")).to(tl.float32)
    e = tl.exp(x - tl.max(x, 0))
    tl.store(Y + row * N + col, e / tl.sum(e, 0), col < N)


def softmax(x, *, num_warps=4):
    matrix(x)
    warps(num_warps)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.shape[0]:
        with torch.cuda.device(x.device):
            _kernel[(x.shape[0],)](
                x,
                out,
                *x.stride(),
                x.shape[1],
                triton.next_power_of_2(x.shape[1]),
                num_warps=num_warps,
            )
    return out
