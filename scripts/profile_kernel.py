"""A warmed-up NVTX range for Nsight; profile one operator without a benchmark sweep."""

import argparse

import torch

from kernel_portfolio import matmul, softmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=["softmax", "matmul"], default="softmax")
    args = parser.parse_args()
    torch.manual_seed(2026)
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    for _ in range(5):
        matmul(x, x) if args.op == "matmul" else softmax(x)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range(args.op):
        for _ in range(10):
            matmul(x, x) if args.op == "matmul" else softmax(x)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
