"""Exercise: read ../lessons/05_split_softmax.md. Solutions are separate.

Use imports of torch, triton and triton.language when implementing your kernel.
You may reuse learning.workshops.contracts for validation.
"""

# Suggested outline. Delete these comments as you implement; ignore them for more challenge.
# Three kernels, launched in this order on the current stream:
#
# @triton.jit
# def _partials(X, PM, PL, N, T, CHUNK: tl.constexpr):
#     # grid (rows, T): max and shifted exp-sum of one chunk, written to PM/PL
# @triton.jit
# def _merge(PM, PL, GM, GL, T, B: tl.constexpr):
#     # grid (rows,): rescale the T partial sums to the row max (lesson A1's merge)
# @triton.jit
# def _normalize(X, GM, GL, OUT, N, CHUNK: tl.constexpr):
#     # grid (rows, T): reread one chunk and write exp(x - max) / sum
#
# def softmax(x, *, chunk_size=1024):
#     # 1. validate: matrix(x, max_width=131072, contiguous=True); chunk_size in (256, 1024, 4096)
#     # 2. T = cdiv(width, chunk_size); allocate FP32 scratch per call:
#     #    partial max/sum [rows, T] and final max/sum [rows]
#     # 3. launch the three kernels; B = triton.next_power_of_2(T) for the merge


def softmax(x, *, chunk_size=1024):
    """Use three kernels: partial stats, rescaled merge, output.
    Allocate scratch per call; avoid global barriers or shared scratch state."""
    raise NotImplementedError("Implement split_softmax; see the lesson and --hint")
