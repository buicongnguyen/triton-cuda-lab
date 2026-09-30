"""Isolate grouped tile ordering: the same GEMM tile config with GROUP_M = 1, 4 and 8.

FP16, CUDA-graph replay of 10 calls, median of 7 samples, warm L2. Results are
checked against a float64 reference before timing.
"""

import argparse
import statistics

import torch
import triton

from kernel_portfolio import triton_kernels as k
from kernel_portfolio.benchmark import ensure_warm, prepare_timer, sample_ms, tolerance

TILE = {"BM": 128, "BN": 128, "BK": 32}  # the autotuner's pick at 4096^3 on an RTX 4080 SUPER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    args = parser.parse_args()
    torch.manual_seed(2026)
    print(f"Tile {TILE}, 8 warps, 3 stages; median microseconds and TFLOP/s")
    for size in args.sizes:
        a = torch.randn((size, size), device="cuda", dtype=torch.float16)
        b = torch.randn((size, size), device="cuda", dtype=torch.float16)
        expected = (a.double() @ b.double()).half()
        cells = []
        for group in (1, 4, 8):
            out = torch.empty_like(a)
            grid = (triton.cdiv(size, TILE["BM"]) * triton.cdiv(size, TILE["BN"]),)

            def run(out=out, group=group, grid=grid):
                k._matmul[grid](
                    a,
                    b,
                    out,
                    size,
                    k.m_bucket(size),
                    size,
                    size,
                    **TILE,
                    GROUP_M=group,
                    num_warps=8,
                    num_stages=3,
                )

            run()
            torch.testing.assert_close(out, expected, **tolerance("matmul", torch.float16))
            timer, count = prepare_timer(run, "graph", 10)
            ensure_warm()
            us = statistics.median(sample_ms(timer, count) for _ in range(7)) * 1e3
            cells.append(f"GROUP_M={group}: {us:9.1f} us {2 * size**3 / (us * 1e6):6.1f} TF")
        print(f"{size}^3  " + " | ".join(cells))


if __name__ == "__main__":
    main()
