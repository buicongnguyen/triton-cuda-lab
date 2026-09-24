"""Lesson 1. Edit only this learner file; reference answers live in solutions/."""


def ceil_div(n, block_size):
    """Number of blocks needed for n elements. n >= 0 and block_size > 0.

    Example: ceil_div(10, 4) == 3. Think about the partly filled last block.
    """
    raise NotImplementedError("Compute integer ceiling division")


def block_offsets(n, block_size, program_id):
    """Return (offsets, mask), each a list with block_size entries.

    Example: block_offsets(10, 4, 2) == ([8, 9, 10, 11], [True, True, False, False]).
    Start = program_id * block_size; each lane adds its local index.
    Do not discard invalid offsets: their mask is what makes memory access safe.
    """
    raise NotImplementedError("Construct every offset, then its validity mask")
