"""Lesson 6. Start with scalar multiply-accumulate before thinking about Tensor Cores."""


def matmul(a, b):
    """Multiply nested-list matrices A[M,K] and B[K,N].

    Shapes are nonempty, rectangular and compatible. Allocate an M-by-N result;
    each output element needs a sum over K. Do not change either input.
    """
    raise NotImplementedError("Three loops: output row, output column, reduction index")


def tile_ranges(length, tile_size):
    """Return half-open (start, stop) ranges covering a dimension exactly once.

    Example: tile_ranges(7, 3) == [(0, 3), (3, 6), (6, 7)].
    The final stop is clipped to length; length can be zero.
    """
    raise NotImplementedError("Generate tile starts and clip each tile's stop")
