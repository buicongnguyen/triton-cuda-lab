"""Shared workshop contracts. Imported only for GPU work."""

from kernel_portfolio import contracts as c


def matrix(x, name="x", *, max_width=8192, contiguous=False):
    c.tensor(x, name, ndim=2)
    if not 1 <= x.shape[1] <= max_width:
        raise ValueError(f"{name} width must be in [1, {max_width}]")
    if x.numel() >= 2**31:
        raise ValueError("Output exceeds signed 32-bit indexing")
    if contiguous:
        c.contiguous(x)


def warps(value):
    if value not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
