"""Run every Triton kernel path once, checked, for Compute Sanitizer (memcheck, racecheck).

The full GPU test suite takes over an hour under racecheck. This script covers each
compiled kernel variant on a modest shape: add, the single-block and both looped
chunkings of each row kernel, the fixed GEMM and every autotuning configuration,
and one call through the torch.compile custom ops. Results are compared with PyTorch.

    compute-sanitizer --tool racecheck --error-exitcode 1 python scripts/sanitizer_smoke.py
"""

import torch
import triton

import kernel_portfolio.library  # noqa: F401  (registers torch.ops.kernel_portfolio.*)
from kernel_portfolio import ops, references
from kernel_portfolio import triton_kernels as k
from kernel_portfolio.benchmark import tolerance


def check(label, actual, expected, op):
    torch.testing.assert_close(actual, expected, **tolerance(op, actual.dtype))
    print(f"ok  {label}")


def main():
    torch.manual_seed(2026)
    half = torch.float16
    x = torch.randn(257, device="cuda", dtype=half)
    check("add", ops.add(x, x), x + x, "add")

    rows_per_path = {
        "single block": (7, 33),
        "looped, few rows (8192-wide chunks)": (3, 8193),
        "looped, many rows (2048-wide chunks)": (2 * k._sm_count(0), 8193),
    }
    for label, (rows, width) in rows_per_path.items():
        x = torch.randn((rows, width + 3), device="cuda", dtype=half)[:, :width]
        r = torch.randn_like(x)
        w = torch.randn(width, device="cuda", dtype=half)
        check(f"row_sum, {label}", ops.row_sum(x), x.double().sum(-1).float(), "row_sum")
        check(
            f"softmax, {label}",
            ops.softmax(x),
            torch.softmax(x.double(), -1).to(half),
            "softmax",
        )
        check(
            f"rmsnorm, {label}",
            ops.residual_rmsnorm(x, r, w),
            references.residual_rmsnorm(x, r, w),
            "rmsnorm",
        )

    m, n, kk = 127, 255, 65
    a = torch.randn((m, kk), device="cuda", dtype=half)
    b = torch.randn((kk, n), device="cuda", dtype=half)
    expected = (a.double() @ b.double()).half()
    check("matmul, fixed tile", ops.matmul(a, b, autotune=False), expected, "matmul")
    for config in k._matmul_tuned.configs:
        meta = config.kwargs
        out = torch.empty((m, n), device="cuda", dtype=half)
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
        check(f"matmul, {config}", out, expected, "matmul")

    y = torch.ops.kernel_portfolio.softmax(x)
    check("custom op softmax", y, ops.softmax(x), "softmax")
    torch.cuda.synchronize()
    print("all kernel paths passed")


if __name__ == "__main__":
    main()
