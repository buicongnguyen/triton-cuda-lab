"""Evidence for triton_kernels._chunking: looped-row chunk width and warps vs row count.

Forces each (chunk, warps) choice for rows wider than 8192, then runs the automatic
choice. FP16, CUDA-graph replay of 20 calls, median of 7 samples, warm L2, microseconds.
"""

import argparse
import statistics

import torch

from kernel_portfolio import ops
from kernel_portfolio import triton_kernels as k
from kernel_portfolio.benchmark import prepare_timer, sample_ms

CHOICES = ((2048, 4), (4096, 8), (8192, 8), (8192, 16))
SHAPES = ((4, 32769), (64, 131072), (1024, 16384))


def time_us(fn):
    fn()
    run, count = prepare_timer(fn, "graph", 20)
    return statistics.median(sample_ms(run, count) for _ in range(7)) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.manual_seed(2026)
    automatic = k._chunking
    sms = k._sm_count(torch.cuda.current_device())
    print(f"{sms} SMs; automatic rule: 2048/4w when rows >= {2 * sms}, else 8192/16w")
    header = "".join(f"{f'{c}/{w}w':>12}" for c, w in CHOICES)
    print(f"{'shape':12} {'op':8} {'torch':>10}{header}{'auto':>12}")
    try:
        for rows, n in SHAPES:
            x = torch.randn((rows, n), device="cuda", dtype=torch.float16)
            r, w = torch.randn_like(x), torch.randn(n, device="cuda", dtype=torch.float16)
            runs = {
                "softmax": (lambda: torch.softmax(x, -1), lambda: ops.softmax(x)),
                "row_sum": (
                    lambda: x.sum(-1, dtype=torch.float32),
                    lambda: ops.row_sum(x),
                ),
                "rmsnorm": (None, lambda: ops.residual_rmsnorm(x, r, w)),
            }
            for op, (baseline, candidate) in runs.items():
                cells = [f"{time_us(baseline):10.2f}" if baseline else f"{'-':>10}"]
                for choice in CHOICES:
                    k._chunking = lambda rows, device, choice=choice: choice
                    cells.append(f"{time_us(candidate):12.2f}")
                k._chunking = automatic
                cells.append(f"{time_us(candidate):12.2f}")
                print(f"{rows}x{n:<8} {op:8} " + "".join(cells))
    finally:
        k._chunking = automatic


if __name__ == "__main__":
    main()
