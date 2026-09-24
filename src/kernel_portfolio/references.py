"""Readable PyTorch oracles; these are deliberately independent of GPU kernels."""

import torch


def softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
    z = x.float()
    numerator = torch.exp(z - z.amax(dim=-1, keepdim=True))
    return (numerator / numerator.sum(dim=-1, keepdim=True)).to(x.dtype)


def residual_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    # Add in FP32: intentionally different from rounding x + residual to FP16 first.
    z = x.float() + residual.float()
    inverse_rms = torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + eps)
    return (z * inverse_rms * weight.float()).to(x.dtype)
