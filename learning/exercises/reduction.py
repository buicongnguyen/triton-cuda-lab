"""Lesson 3. Show a reduction tree rather than calling sum() once."""


def sum_stages(values):
    """Return all stages of a pairwise sum, including the starting values.

    Pad the starting list with zeros to the next power of two. Pair neighboring
    values until only one remains. For [1, 2, 3], return
    [[1, 2, 3, 0], [3, 3], [6]]. Input is nonempty; do not modify it.
    """
    raise NotImplementedError("Pad with the sum identity, then repeatedly combine pairs")
