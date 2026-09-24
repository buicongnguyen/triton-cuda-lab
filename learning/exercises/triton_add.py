"""Lesson 8. Complete the kernel, then remove the NotImplementedError in add()."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(X, Y, OUT, N, BLOCK: tl.constexpr):
    # N changes per call, so it is a runtime argument; BLOCK is a compile-time tile size.
    # 1. offsets = program_id(0) * BLOCK + arange(0, BLOCK)
    # 2. mask = offsets < N
    # 3. Load X and Y with mask=mask, other=0; convert to tl.float32.
    # 4. Store their sum to OUT, with the same mask.
    pass


def add(x, y):
    """Checker supplies same-shaped contiguous CUDA vectors, detached from autograd."""
    raise NotImplementedError("Fill add_kernel, then remove this line")
    out = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), 256)
    return out
