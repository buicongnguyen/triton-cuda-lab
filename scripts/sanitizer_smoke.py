"""Run every Triton kernel path once, checked, for Compute Sanitizer (memcheck, racecheck).

The full GPU test suite takes over an hour under racecheck. This script covers each
compiled kernel variant on a modest shape, in every supported dtype (each dtype is a
separate compilation): add; every row plan of each row kernel (one block, split
across programs, and both looped chunkings), with softmax also given -inf masked
entries; the softmax and residual RMSNorm backward kernels on every plan, including
the weight-gradient kernels; the fixed GEMM and every autotuning configuration in FP16
and BF16, forward and both backward products; and one call through the torch.compile
custom ops. Results are compared with PyTorch.

    compute-sanitizer --tool racecheck --error-exitcode 1 python scripts/sanitizer_smoke.py
"""

import torch
import triton

import kernel_portfolio.library  # noqa: F401  (registers torch.ops.kernel_portfolio.*)
from kernel_portfolio import ops, references
from kernel_portfolio import triton_kernels as k
from kernel_portfolio.benchmark import tolerance

DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def check(label, actual, expected, op):
    torch.testing.assert_close(actual, expected, **tolerance(op, actual.dtype))
    print(f"ok  {label}")


def row_kernels(dtype):
    sms = k._sm_count(0)
    rows_per_path = {
        "single block": (7, 33),
        "split (softmax and RMSNorm; row sum loops)": (3, 20000),
        "looped, 8192-wide chunks, few rows (softmax splits)": (3, 8193),
        "looped, 4096-wide chunks": (sms, 8193),
        "looped, 8192-wide chunks, wide rows": (sms, 32769),
    }
    for label, (rows, width) in rows_per_path.items():
        label = f"{label}, {dtype}"
        x = torch.randn((rows, width + 3), device="cuda", dtype=dtype)[:, :width]
        r = torch.randn_like(x)
        w = torch.randn(width, device="cuda", dtype=dtype)
        check(f"row_sum, {label}", ops.row_sum(x), x.double().sum(-1).float(), "row_sum")
        y = ops.softmax(x)
        check(f"softmax, {label}", y, torch.softmax(x.double(), -1).to(dtype), "softmax")
        masked = x.clone()
        masked[:, : width // 3] = float("-inf")
        check(
            f"softmax with -inf mask, {label}",
            ops.softmax(masked),
            torch.softmax(masked.double(), -1).to(dtype),
            "softmax",
        )
        check(
            f"rmsnorm, {label}",
            ops.residual_rmsnorm(x, r, w),
            references.residual_rmsnorm(x, r, w),
            "rmsnorm",
        )
        dy = torch.randn((width, rows), device="cuda", dtype=dtype).T
        y64, dy64 = y.double(), dy.double()
        check(
            f"softmax backward, {label}",
            ops.softmax_backward(dy, y),
            (y64 * (dy64 - (y64 * dy64).sum(-1, keepdim=True))).to(dtype),
            "softmax_backward",
        )
        dx, dw = ops.residual_rmsnorm_backward(dy, x, r, w)
        x64, r64, w64 = (t.double().requires_grad_() for t in (x, r, w))
        z = x64 + r64
        (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5) * w64).backward(dy64)
        check(f"rmsnorm backward dx, {label}", dx, x64.grad.to(dtype), "rmsnorm")
        # The weight gradient sums one term per row; see tests/test_gpu.py.
        tol = tolerance("rmsnorm", dtype)
        torch.testing.assert_close(
            dw, w64.grad.to(dtype), rtol=tol["rtol"], atol=tol["atol"] * rows**0.5
        )
        print(f"ok  rmsnorm backward dweight, {label}")


def gemm(dtype):
    m, n, kk = 127, 255, 65
    a = torch.randn((m, kk), device="cuda", dtype=dtype)
    b = torch.randn((kk, n), device="cuda", dtype=dtype)
    expected = (a.double() @ b.double()).to(dtype)
    check(f"matmul, fixed tile, {dtype}", ops.matmul(a, b, autotune=False), expected, "matmul")
    for config in k._matmul_tuned.configs:
        meta = config.kwargs
        out = torch.empty((m, n), device="cuda", dtype=dtype)
        grid = (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
        try:
            k._matmul[grid](
                a,
                b,
                out,
                m,
                k.m_bucket(m),
                n,
                kk,
                **meta,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
        except triton.runtime.errors.OutOfResources:
            print(f"skip {config} (exceeds this GPU's shared memory)")
            continue
        check(f"matmul, {config}, {dtype}", out, expected, "matmul")
    gemm_backward(a, b, dtype)


def gemm_backward(a, b, dtype):
    """Both backward products, fixed tile and every configuration, with a transposed gradient."""
    m, kk = a.shape
    n = b.shape[1]
    g = torch.randn((n, m), device="cuda", dtype=dtype).T
    da_expected = (g.double() @ b.double().T).to(dtype)
    db_expected = (a.double().T @ g.double()).to(dtype)
    da, db = ops.matmul_backward(g, a, b, autotune=False)
    check(f"matmul dA, fixed tile, {dtype}", da, da_expected, "matmul")
    check(f"matmul dB, fixed tile, {dtype}", db, db_expected, "matmul")
    products = (
        ("dA", g, b, (m, kk), (m, kk, n), g.stride(), b.t().stride(), da_expected),
        ("dB", a, g, (kk, n), (kk, n, m), a.t().stride(), g.stride(), db_expected),
    )
    for config in k._matmul_tuned.configs:
        meta = config.kwargs
        for name, x, y, shape, (rows, cols, red), xs, ys, expected in products:
            out = torch.empty(shape, device="cuda", dtype=dtype)
            grid = (triton.cdiv(rows, meta["BM"]) * triton.cdiv(cols, meta["BN"]),)
            args = (x, y, out, rows, k.m_bucket(rows), red, k.m_bucket(red), *xs, *ys, cols)
            try:
                k._matmul_strided[grid](
                    *args, **meta, num_warps=config.num_warps, num_stages=config.num_stages
                )
            except triton.runtime.errors.OutOfResources:
                print(f"skip {config} (exceeds this GPU's shared memory)")
                continue
            check(f"matmul {name}, {config}, {dtype}", out, expected, "matmul")


def main():
    torch.manual_seed(2026)
    for dtype in DTYPES:
        x = torch.randn(257, device="cuda", dtype=dtype)
        check(f"add, {dtype}", ops.add(x, x), x + x, "add")
        row_kernels(dtype)
    for dtype in (torch.float16, torch.bfloat16):
        gemm(dtype)
    x = torch.randn((5, 33), device="cuda", dtype=torch.float16)
    check("custom op softmax", torch.ops.kernel_portfolio.softmax(x), ops.softmax(x), "softmax")
    torch.cuda.synchronize()
    print("all kernel paths passed")


if __name__ == "__main__":
    main()
