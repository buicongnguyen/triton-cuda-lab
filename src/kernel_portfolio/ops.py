"""Validated GPU entry points. No CPU fallback, hidden copies, or autograd support.

Each operator is split into a `_*_out` helper (validation and output allocation) and
a launch. kernel_portfolio.library reuses the helpers for its torch.compile-friendly
custom ops, so both entry points enforce the same contracts.
"""

import contextlib

import torch

from . import contracts as c


def _backend():
    try:
        from . import triton_kernels
    except ImportError as exc:
        raise RuntimeError(
            "Triton is not installed: use triton on Linux/WSL or triton-windows on Windows; "
            "see docs/SETUP.md"
        ) from exc
    return triton_kernels


def _on(device: torch.device):
    """Launch on the tensor's device; skip the context switch when it is already current."""
    if device.index == torch.cuda.current_device():
        return contextlib.nullcontext()
    return torch.cuda.device(device)


def _add_out(x: torch.Tensor, y: torch.Tensor, block_size: int) -> torch.Tensor:
    c.tensor(x, "x", ndim=1)
    c.tensor(y, "y", ndim=1)
    c.same(x, y)
    c.contiguous(x)
    c.contiguous(y)
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("block_size must be 128, 256, 512, or 1024")
    return torch.empty_like(x)


def add(x: torch.Tensor, y: torch.Tensor, *, block_size: int = 256) -> torch.Tensor:
    """Add contiguous vectors of the same dtype; block_size is an experimental knob."""
    out = _add_out(x, y, block_size)
    if x.numel():
        with _on(x.device):
            _backend().launch_add(x, y, out, block_size)
    return out


def _row_sum_out(x: torch.Tensor) -> torch.Tensor:
    c.tensor(x, "x", ndim=2)
    c.rows(x)
    return torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    """Reduce each row into FP32. Width 1..2**20 (looped above 8192); row gaps are supported."""
    out = _row_sum_out(x)
    if x.shape[0]:
        with _on(x.device):
            _backend().launch_row_sum(x, out)
    return out


def _softmax_out(x: torch.Tensor, num_warps: int) -> torch.Tensor:
    c.tensor(x, "x", ndim=2)
    c.rows(x)
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def softmax(x: torch.Tensor, *, num_warps: int = 4) -> torch.Tensor:
    """Stable row softmax; width 1..2**20; same dtype output.

    Scores are finite or -inf (masked entries get probability 0). A row with no
    finite score returns NaN, as torch.softmax does. num_warps is an experimental knob
    for rows up to 8192 wide; wider rows choose their own chunking and warps.
    """
    out = _softmax_out(x, num_warps)
    if x.shape[0]:
        with _on(x.device):
            _backend().launch_softmax(x, out, num_warps)
    return out


def _rmsnorm_out(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    c.tensor(x, "x", ndim=2)
    c.tensor(residual, "residual", ndim=2)
    c.tensor(weight, "weight", ndim=1)
    c.same(x, residual)
    c.same(x, weight, shape=False)
    c.rows(x)
    c.rows(residual)
    c.contiguous(weight)
    c.epsilon(eps)
    if weight.shape[0] != x.shape[1]:
        raise ValueError("weight length must equal row width")
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def residual_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """RMSNorm(x.float() + residual.float()) * weight, rounded once to x.dtype."""
    out = _rmsnorm_out(x, residual, weight, eps)
    if x.shape[0]:
        with _on(x.device):
            _backend().launch_rmsnorm(x, residual, weight, out, eps)
    return out


def _matmul_out(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c.tensor(a, "a", ndim=2)
    c.tensor(b, "b", ndim=2)
    c.same(a, b, shape=False)
    c.contiguous(a)
    c.contiguous(b)
    if a.dtype == torch.float32:
        raise TypeError("matmul accepts float16 and bfloat16 only")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match")
    m, k = a.shape
    n = b.shape[1]
    c.matmul_shape(m, n)
    # K == 0 is an empty sum: the result is all zeros and no kernel runs.
    make = torch.zeros if k == 0 else torch.empty
    return make((m, n), dtype=a.dtype, device=a.device)


def matmul(a: torch.Tensor, b: torch.Tensor, *, autotune: bool = True) -> torch.Tensor:
    """Contiguous FP16/BF16 GEMM with FP32 accumulation, returned in input dtype."""
    out = _matmul_out(a, b)
    if out.numel() and a.shape[1]:
        with _on(a.device):
            _backend().launch_matmul(a, b, out, autotune)
    return out
