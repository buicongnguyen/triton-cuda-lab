"""The same operators registered as PyTorch custom ops, for use inside torch.compile.

`torch.library.triton_op` makes each operator visible to torch.compile: Dynamo records
one opaque op instead of breaking the graph at a Python function, and Inductor sees
the Triton launches through `wrap_triton`. In a compiled model with
mode="reduce-overhead", the whole graph, these kernels included, replays as a CUDA
graph, which removes the per-call Python launch cost measured in docs/CASE_STUDIES.md.

    import kernel_portfolio.library  # registers torch.ops.kernel_portfolio.*
    y = torch.ops.kernel_portfolio.softmax(x)

Forward-only like kernel_portfolio.ops: no autograd formula is registered. Importing
this module requires Triton.
"""

import torch
from torch.library import triton_op, wrap_triton

from . import ops
from . import triton_kernels as k


@triton_op("kernel_portfolio::add", mutates_args={})
def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    out = ops._add_out(x, y, block_size)
    if x.numel():
        with ops._on(x.device):
            k.launch_add(x, y, out, block_size, wrap=wrap_triton)
    return out


@triton_op("kernel_portfolio::row_sum", mutates_args={})
def row_sum(x: torch.Tensor) -> torch.Tensor:
    out = ops._row_sum_out(x)
    if x.shape[0]:
        with ops._on(x.device):
            k.launch_row_sum(x, out, wrap=wrap_triton)
    return out


@triton_op("kernel_portfolio::softmax", mutates_args={})
def softmax(x: torch.Tensor, num_warps: int = 4) -> torch.Tensor:
    out = ops._softmax_out(x, num_warps)
    if x.shape[0]:
        with ops._on(x.device):
            k.launch_softmax(x, out, num_warps, wrap=wrap_triton)
    return out


@triton_op("kernel_portfolio::residual_rmsnorm", mutates_args={})
def residual_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    out = ops._rmsnorm_out(x, residual, weight, eps)
    if x.shape[0]:
        with ops._on(x.device):
            k.launch_rmsnorm(x, residual, weight, out, eps, wrap=wrap_triton)
    return out


@triton_op("kernel_portfolio::matmul", mutates_args={})
def matmul(a: torch.Tensor, b: torch.Tensor, autotune: bool = True) -> torch.Tensor:
    out = ops._matmul_out(a, b)
    if out.numel() and a.shape[1]:
        with ops._on(a.device):
            k.launch_matmul(a, b, out, autotune, wrap=wrap_triton)
    return out
