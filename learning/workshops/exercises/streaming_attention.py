"""Exercise: read ../lessons/06_streaming_attention.md. Solutions are separate.

Use imports of torch, triton and triton.language when implementing your kernel.
You may reuse learning.workshops.contracts for validation.
"""

# Suggested outline. Delete these comments as you implement; ignore them for more challenge.
#
# @triton.jit
# def _kernel(Q, K, V, OUT, N, D: tl.constexpr, SCALE, CAUSAL: tl.constexpr,
#             BM: tl.constexpr, BN: tl.constexpr):
#     # N and SCALE are runtime; D is constexpr because tl.arange(0, D) needs it.
#     # 1. this program owns BM query rows; load its Q tile once
#     # 2. m = -inf, l = 0, acc = zeros [BM, D], all FP32
#     # 3. for each BN-wide key tile (tl.cdiv(N, BN) of them): scores = dot(q, k^T) * SCALE;
#     #    mask key tails, and future keys when CAUSAL; then the online update from the lesson
#     # 4. store acc / l for the valid query rows
#
# def attention(q, k, v, *, causal=False, block_n=32):
#     # 1. validate: contiguous matching [N, D] FP16/BF16 tensors, N <= 2048,
#     #    D in (16, 32, 64), causal is a bool, block_n in (32, 64)
#     # 2. allocate the output; if N > 0, launch cdiv(N, 16) programs with BM=16


def attention(q, k, v, *, causal=False, block_n=32):
    """Loop over K/V tiles; maintain running max, sum and PV accumulator.
    Rescale both accumulators when max changes. Mask key tails and causal edges."""
    raise NotImplementedError("Implement streaming_attention; see the lesson and --hint")
