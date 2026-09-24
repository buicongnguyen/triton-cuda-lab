"""Lesson 2. Strides are measured in elements, not bytes."""


def offset(row, col, row_stride, col_stride=1, storage_offset=0):
    """Map logical (row, col) to a flat storage index.

    Example: offset(1, 2, 5) == 7, even if the logical row has only 3 columns.
    Include the optional offset of a view's first element in the original storage.
    """
    raise NotImplementedError("Add the base offset and both stride contributions")


def read_view(storage, rows, cols, row_stride, col_stride=1, storage_offset=0):
    """Return a nested list representing the view, using offset() for each element."""
    raise NotImplementedError("Use two loops over logical rows and columns")
