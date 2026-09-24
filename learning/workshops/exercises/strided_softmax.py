"""Exercise: read ../lessons/01_strided_softmax.md. Solutions are separate.

Use imports of torch, triton and triton.language when implementing your kernel.
You may reuse learning.workshops.contracts for validation.
"""

# Suggested outline. Delete these comments as you implement; ignore them for more challenge.
#
# import torch
# import triton
# import triton.language as tl
# from learning.workshops.contracts import matrix, warps
#
# @triton.jit
# def _kernel(X, Y, SR, SC, N, B: tl.constexpr):
#     # SR/SC (input row/column strides) and N (width) are runtime ints, so a new
#     # shape reuses the compiled kernel. B = next_power_of_2(N) must be constexpr.
#     # 1. row = this program's id; col = arange(0, B)
#     # 2. load X + row*SR + col*SC with mask col < N and padding -inf; convert to FP32
#     # 3. stable softmax: subtract the row max, exponentiate, divide by the sum
#     # 4. store to Y + row*N + col (the output is contiguous) with the same mask
#
# def softmax(x, *, num_warps=4):
#     # 1. validate: matrix(x) and warps(num_warps)
#     # 2. out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
#     # 3. if x has rows: under torch.cuda.device(x.device), launch one program per row
#     #    with *x.stride(), the width, triton.next_power_of_2(width), num_warps=num_warps
#     # 4. return out


def softmax(x, *, num_warps=4):
    """Validate a 2D CUDA input; load both strides with a tail mask.
    Reduce max/sum in FP32 and allocate contiguous output. Support empty rows."""
    raise NotImplementedError("Implement strided_softmax; see the lesson and --hint")
