"""Shared input contracts; kept independent of Triton for CPU validation."""

import math

import torch

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Rows wider than 8192 use looped kernels; this bound keeps tests able to reach it.
MAX_ROW_WIDTH = 1 << 20


def tensor(x: torch.Tensor, name: str, *, ndim: int, gpu: bool = True) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.layout != torch.strided:
        raise ValueError(f"{name} must be a strided tensor")
    if x.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if x.dtype not in DTYPES:
        raise TypeError(f"{name} must have float16, bfloat16, or float32 dtype")
    # Parameters keep requires_grad=True under no_grad/inference_mode; only reject
    # them when autograd would otherwise record a graph this kernel cannot extend.
    # (softmax and residual_rmsnorm route gradient-tracking inputs to their custom ops,
    # whose autograd formula runs this check with gradients disabled.)
    if x.requires_grad and torch.is_grad_enabled():
        raise ValueError(
            "This operator is forward-only (only softmax and residual_rmsnorm support "
            "autograd); use torch.no_grad()/inference_mode() or detach"
        )
    # Kernel pointer offsets use signed 32-bit arithmetic. Check the physical span,
    # not just numel: a narrow view can have a very large row stride.
    if x.numel() and sum((size - 1) * stride for size, stride in zip(x.shape, x.stride())) >= 2**31:
        raise ValueError("Tensor address span exceeds the supported 32-bit indexing range")
    if gpu and (not x.is_cuda or torch.version.hip is not None):
        raise ValueError(f"{name} must be on an NVIDIA CUDA device")


def same(a: torch.Tensor, b: torch.Tensor, *, shape: bool = True) -> None:
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("Inputs must share device and dtype")
    if shape and a.shape != b.shape:
        raise ValueError("Inputs must have identical shapes")


def contiguous(x: torch.Tensor) -> None:
    if not x.is_contiguous():
        raise ValueError("This operator requires contiguous input")


def rows(x: torch.Tensor) -> None:
    if not 1 <= x.shape[1] <= MAX_ROW_WIDTH:
        raise ValueError(f"Row width must be in [1, {MAX_ROW_WIDTH}]")
    # A width-1 row has no column stride to honor, so any stride(1) is acceptable.
    if (x.shape[1] > 1 and x.stride(1) != 1) or (x.shape[0] > 1 and x.stride(0) < x.shape[1]):
        raise ValueError("Rows must not overlap and columns must be contiguous")


def epsilon(eps: float) -> None:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def matmul_shape(m: int, n: int) -> None:
    # The 1D tile grid has no practical limit; output offsets must fit signed 32 bits.
    if m * n >= 2**31:
        raise ValueError("GEMM output exceeds the supported 32-bit indexing range")
