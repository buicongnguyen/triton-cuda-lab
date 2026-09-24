"""Exercise: read ../lessons/03_fused_gemm.md. Solutions are separate.

Use imports of torch, triton and triton.language when implementing your kernel.
You may reuse learning.workshops.contracts for validation.
"""

# Suggested outline. Delete these comments as you implement; ignore them for more challenge.
#
# @triton.jit
# def _kernel(A, B, BIAS, OUT, M, N, K,
#             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
#     # M, N, K are runtime ints; tile sizes and GROUP are compile-time.
#     # 1. map the 1D program id to (tile_m, tile_n) with grouped ordering (see the
#     #    six-program table in the lesson; clamp the last group with tl.minimum)
#     # 2. FP32 accumulator [BM, BN]; for each of tl.cdiv(K, BK) steps, masked loads of an
#     #    A tile [BM, BK] and a B tile [BK, BN], then acc = tl.dot(a, b, acc)
#     # 3. epilogue in FP32: add bias[cols], apply ReLU; masked store (the store casts)
#
# def matmul_bias_relu(a, b, bias, *, group_m=4, tile_m=32):
#     # 1. validate: CUDA, contiguous, one FP16/BF16 dtype, A[M,K] B[K,N] bias[N],
#     #    group_m in (1, 4, 8), tile_m in (16, 32, 64), M*N < 2**31
#     # 2. allocate the [M, N] output; if M and N are nonzero, launch a 1D grid of
#     #    cdiv(M, tile_m) * cdiv(N, 64) programs with BN=64, BK=32


def matmul_bias_relu(a, b, bias, *, group_m=4, tile_m=32):
    """Implement grouped tile ownership and a masked K loop.
    Add the column bias and ReLU before converting the FP32 accumulator."""
    raise NotImplementedError("Implement fused_gemm; see the lesson and --hint")
