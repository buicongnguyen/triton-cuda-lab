"""Reference answer: a view changes its coordinate mapping, not the stored values."""


def offset(row, col, row_stride, col_stride=1, storage_offset=0):
    return storage_offset + row * row_stride + col * col_stride


def read_view(storage, rows, cols, row_stride, col_stride=1, storage_offset=0):
    return [
        [storage[offset(r, c, row_stride, col_stride, storage_offset)] for c in range(cols)]
        for r in range(rows)
    ]
