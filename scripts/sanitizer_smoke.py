"""Run every Triton kernel path once, checked, for Compute Sanitizer (memcheck, racecheck).

The full GPU test suite takes over an hour under racecheck. This script covers each
compiled kernel variant on a modest shape, in every supported dtype (each dtype is a
separate compilation): add, the single-block and both looped chunkings of each row
kernel (softmax also with -inf masked entries), the fixed GEMM and every autotuning
configuration in FP16 and BF16, and one call through the torch.compile custom ops.
Results are compared with PyTorch.

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
    rows_per_path = {
        "single block": (7, 33),
        "looped, few rows (8192-wide chunks)": (3, 8193),
        "looped, many rows (2048-wide chunks)": (2 * k._sm_count(0), 8193),
    }
    for label, (rows, width) in rows_per_path.items():
        label = f"{label}, {dtype}"
        x = torch.randn((rows, width + 3), device="cuda", dtype=dtype)[:, :width]
        r = torch.randn_like(x)
        w = torch.randn(width, device="cuda", dtype=dtype)
        check(f"row_sum, {label}", ops.row_sum(x), x.double().sum(-1).float(), "row_sum")
        expected = torch.softmax(x.double(), -1).to(dtype)
        check(f"softmax, {label}", ops.softmax(x), expected, "softmax")
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
