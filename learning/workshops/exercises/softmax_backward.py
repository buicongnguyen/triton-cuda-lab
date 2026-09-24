"""Exercise: read ../lessons/02_softmax_backward.md. Solutions are separate.

Use imports of torch, triton and triton.language when implementing your kernel.
You may reuse learning.workshops.contracts for validation.
"""

# Suggested outline. Delete these comments as you implement; ignore them for more challenge.
#
# @triton.jit
# def _kernel(Y, G, DX, SY0, SY1, SG0, SG1, N, B: tl.constexpr):
#     # y and g each have their own row/column strides (runtime ints); B is constexpr.
#     # 1. one program per row; load y and g with their own strides, zero padding, FP32
#     # 2. dot = sum(y * g) over the row
#     # 3. store y * (g - dot) to the contiguous output DX + row*N + col
#
# def backward(y, grad_output, *, num_warps=4):
#     # 1. validate: matrix() on both, kernel_portfolio.contracts.same(y, grad_output), warps()
#     # 2. allocate a contiguous output of y's shape and dtype
#     # 3. if there are rows: launch one program per row under the input's device


def backward(y, grad_output, *, num_warps=4):
    """Validate matching tensors. Compute y*(g-sum(y*g)) in FP32.
    y and g can each have different row and column strides."""
    raise NotImplementedError("Implement softmax_backward; see the lesson and --hint")
