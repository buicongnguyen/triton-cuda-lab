"""Compare three kernel versions to show what compile-time sizes buy and cost.

- pre-review: every size and stride tl.constexpr (one compile per shape)
- runtime:    every size a runtime argument (the intermediate version)
- current:    row width and GEMM N/K constexpr; counts, strides and GEMM M runtime

The two older versions are loaded from results/archive/. Timing matches kernel-bench:
CUDA-graph replay of 30 calls, median of 9 samples, warm L2, FP16.
"""

import argparse
import importlib.util
import statistics
from pathlib import Path

import torch
import triton

from kernel_portfolio import triton_kernels as current
from kernel_portfolio.benchmark import prepare_timer, sample_ms

ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "archive"
ROW_SHAPES = ((1024, 1024), (512, 3072), (512, 4097), (512, 4104), (512, 4112), (512, 5120))
GEMM_SHAPES = ((127, 255, 65), (512, 512, 512), (1024, 1024, 1024))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def time_us(fn):
    fn()
    run, count = prepare_timer(fn, "graph", 30)
    return statistics.median(sample_ms(run, count) for _ in range(9)) * 1e3


def row_launch(module, op, x, r, w, out, sums):
    rows, n = x.shape
    block = triton.next_power_of_2(n)
    if op == "softmax":
        return module._softmax[(rows,)](x, out, x.stride(0), n, block, num_warps=4)
    if op == "row_sum":
        return module._row_sum[(rows,)](x, sums, x.stride(0), n, block, num_warps=4)
    return module._rmsnorm[(rows,)](
        x, r, w, out, x.stride(0), r.stride(0), n, 1e-5, block, num_warps=4
    )


def gemm_launch(module, a, b, out):
    if module is current:  # 1D grouped grid; the fixed config is the same 32x64x32 tile
        return current.launch_matmul(a, b, out, False)
    (m, k), n = a.shape, b.shape[1]
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 64))  # both archived versions use a 2D grid
    return module._matmul[grid](a, b, out, m, n, k, BM=32, BN=64, BK=32, num_warps=4, num_stages=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    versions = {
        "pre-review": load(ARCHIVE / "pre-review-triton_kernels.py", "pre_review_kernels"),
        "runtime": load(ARCHIVE / "runtime-args-triton_kernels.py", "runtime_kernels"),
        "current": current,
    }
    torch.manual_seed(2026)
    print("Median microseconds; registers per thread for softmax in brackets.")
    print(f"{'shape':12} {'op':8} " + "".join(f"{name:>20}" for name in versions))
    for rows, n in ROW_SHAPES:
        x = torch.randn((rows, n), device="cuda", dtype=torch.float16)
        r, w = torch.randn_like(x), torch.randn(n, device="cuda", dtype=torch.float16)
        out, sums = torch.empty_like(x), torch.empty(rows, device="cuda", dtype=torch.float32)
        for op in ("softmax", "row_sum", "rmsnorm"):
            cells = []
            for module in versions.values():
                compiled = row_launch(module, op, x, r, w, out, sums)
                note = f" [{compiled.n_regs:3}]" if op == "softmax" else ""
                us = time_us(lambda m=module, o=op: row_launch(m, o, x, r, w, out, sums))
                cells.append(f"{us:14.2f}{note:>6}")
            print(f"{rows}x{n:<7} {op:8} " + "".join(cells))
    for m, n, k in GEMM_SHAPES:
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        out = torch.empty((m, n), device="cuda", dtype=torch.float16)
        cells = [
            f"{time_us(lambda v=v: gemm_launch(v, a, b, out)):20.2f}" for v in versions.values()
        ]
        print(f"{m}x{n}x{k:<4} {'gemm':8} " + "".join(cells))


if __name__ == "__main__":
    main()
