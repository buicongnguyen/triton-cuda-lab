"""The same operators registered as PyTorch custom ops, for use inside torch.compile.

`torch.library.triton_op` makes each operator visible to torch.compile: Dynamo records
one opaque op instead of breaking the graph at a Python function, and Inductor sees
the Triton launches through `wrap_triton`. In a compiled model with
mode="reduce-overhead", the whole graph, these kernels included, replays as a CUDA
graph, which removes the per-call Python launch cost measured in docs/CASE_STUDIES.md.

    import kernel_portfolio.library  # registers torch.ops.kernel_portfolio.*
    y = torch.ops.kernel_portfolio.softmax(x)

softmax and residual_rmsnorm have autograd formulas whose backward passes are the
custom ops softmax_backward and residual_rmsnorm_backward, so a compiled training
step traces both directions. add, row_sum and matmul have no autograd formula: they
accept gradient-tracking inputs but raise if backward reaches them. Double backward
is not supported. Importing this module requires Triton.
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


@triton_op("kernel_portfolio::softmax_backward", mutates_args={})
def softmax_backward(grad: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dx = ops._softmax_backward_out(grad, y)
    if y.shape[0]:
        with ops._on(y.device):
            k.launch_softmax_backward(y, grad, dx, wrap=wrap_triton)
    return dx


def _softmax_setup(ctx, inputs, output):
    # The output is all the gradient needs: dx = y * (grad - sum(y * grad)).
    ctx.save_for_backward(output)


def _softmax_grad(ctx, grad):
    (y,) = ctx.saved_tensors
    return torch.ops.kernel_portfolio.softmax_backward(grad, y), None


softmax.register_autograd(_softmax_grad, setup_context=_softmax_setup)


@triton_op("kernel_portfolio::residual_rmsnorm", mutates_args={})
def residual_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    out = ops._rmsnorm_out(x, residual, weight, eps)
    if x.shape[0]:
        with ops._on(x.device):
            k.launch_rmsnorm(x, residual, weight, out, eps, wrap=wrap_triton)
    return out


@triton_op("kernel_portfolio::residual_rmsnorm_backward", mutates_args={})
def residual_rmsnorm_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx, dweight = ops._rmsnorm_backward_out(grad, x, residual, weight, eps)
    if x.shape[0]:
        with ops._on(x.device):
            k.launch_rmsnorm_backward(x, residual, weight, grad, dx, dweight, eps, wrap=wrap_triton)
    return dx, dweight


def _rmsnorm_setup(ctx, inputs, output):
    # Save the inputs, not the normalized output: the backward kernels recompute the
    # per-row inverse RMS, which costs one extra read instead of extra saved memory.
    x, residual, weight, eps = inputs
    ctx.save_for_backward(x, residual, weight)
    ctx.eps = eps


def _rmsnorm_grad(ctx, grad):
    x, residual, weight = ctx.saved_tensors
    dx, dweight = torch.ops.kernel_portfolio.residual_rmsnorm_backward(
        grad, x, residual, weight, ctx.eps
    )
    # x + residual enters the normalization, so both receive dx (as for a + b).
    return dx, dx, dweight, None


residual_rmsnorm.register_autograd(_rmsnorm_grad, setup_context=_rmsnorm_setup)


@triton_op("kernel_portfolio::matmul", mutates_args={})
def matmul(a: torch.Tensor, b: torch.Tensor, autotune: bool = True) -> torch.Tensor:
    out = ops._matmul_out(a, b)
    if out.numel() and a.shape[1]:
        with ops._on(a.device):
            k.launch_matmul(a, b, out, autotune, wrap=wrap_triton)
    return out
