"""How much of each gradient tolerance does the worst error use?

The GPU tests compare every backward kernel with FP64 autograd under the tolerances in
kernel_portfolio.benchmark.tolerance. A tolerance that is too tight fails on some random
input; one that is too loose hides a real bug. For each gradient this script draws many
random inputs and reports the largest ratio of error to tolerance: the tolerance fails
above 1, and a ratio far below 1 means there is room for a real bug to hide.

    python scripts/tolerance_probe.py > results/tolerance-probe.log
    python scripts/tolerance_probe.py --trials 1000
"""

import argparse

import torch

from kernel_portfolio import ops
from kernel_portfolio.benchmark import tolerance

SOFTMAX_SHAPES = ((7, 33), (3, 4097), (1, 8193), (40, 1000))
RMSNORM_SHAPES = ((7, 33), (3, 4097), (40, 1000), (160, 8195))
MATMUL_SHAPES = ((31, 65, 33), (127, 255, 65), (300, 100, 50), (600, 130, 200))


def ratio(actual, expected, tol, scale=1.0):
    """Largest error over the tolerance's allowance at that element."""
    expected = expected.double()
    allowance = tol["atol"] * scale + tol["rtol"] * expected.abs()
    return ((actual.double() - expected).abs() / allowance).max().item()


def randn(shape, dtype, **kwargs):
    return torch.randn(shape, device="cuda", dtype=dtype, **kwargs)


def softmax_case(shape, dtype):
    x, dy = randn(shape, dtype), randn(shape, dtype)
    leaf = x.clone().requires_grad_()
    ops.softmax(leaf).backward(dy)
    x64 = x.double().requires_grad_()
    torch.softmax(x64, -1).backward(dy.double())
    return {"dx": ratio(leaf.grad, x64.grad.to(dtype), tolerance("softmax_backward", dtype))}


def rmsnorm_case(shape, dtype):
    rows, width = shape
    x, r = randn(shape, dtype, requires_grad=True), randn(shape, dtype, requires_grad=True)
    w = randn(width, dtype, requires_grad=True)
    dy = randn(shape, dtype)
    ops.residual_rmsnorm(x, r, w).backward(dy)
    x64, r64, w64 = (t.detach().double().requires_grad_() for t in (x, r, w))
    z = x64 + r64
    (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5) * w64).backward(dy.double())
    tol = tolerance("rmsnorm", dtype)
    return {
        "dx": ratio(x.grad, x64.grad.to(dtype), tol),
        "dresidual": ratio(r.grad, r64.grad.to(dtype), tol),
        # One FP32 term per row is added to the weight gradient: its error grows like sqrt(rows).
        "dweight": ratio(w.grad, w64.grad.to(dtype), tol, max(1.0, rows**0.5)),
    }


def matmul_case(shape, dtype):
    m, n, k = shape
    a, b = randn((m, k), dtype, requires_grad=True), randn((k, n), dtype, requires_grad=True)
    dy = randn((m, n), dtype)
    ops.matmul(a, b, autotune=False).backward(dy)
    a64, b64 = a.detach().double().requires_grad_(), b.detach().double().requires_grad_()
    (a64 @ b64).backward(dy.double())
    tol = tolerance("matmul", dtype)
    return {
        "dA": ratio(a.grad, a64.grad.to(dtype), tol),
        "dB": ratio(b.grad, b64.grad.to(dtype), tol),
    }


SUITES = (
    (
        "softmax backward",
        softmax_case,
        SOFTMAX_SHAPES,
        (torch.float32, torch.float16, torch.bfloat16),
    ),
    (
        "residual RMSNorm backward",
        rmsnorm_case,
        RMSNORM_SHAPES,
        (torch.float32, torch.float16, torch.bfloat16),
    ),
    ("matmul backward", matmul_case, MATMUL_SHAPES, (torch.float16, torch.bfloat16)),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200, help="random inputs per gradient")
    args = parser.parse_args()
    torch.manual_seed(2026)
    print(f"Worst error / tolerance over {args.trials} random inputs per row (fails above 1)")
    for name, case, shapes, dtypes in SUITES:
        print()
        print(name)
        for dtype in dtypes:
            worst = {}
            for trial in range(args.trials):
                for key, value in case(shapes[trial % len(shapes)], dtype).items():
                    worst[key] = max(worst.get(key, 0.0), value)
            cells = "  ".join(f"{key} {value:5.2f}" for key, value in worst.items())
            print(f"  {str(dtype).removeprefix('torch.'):9} {cells}")


if __name__ == "__main__":
    main()
