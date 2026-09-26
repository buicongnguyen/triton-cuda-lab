"""Host-side cost of each piece of an eager ops.softmax call (Python dispatch, not GPU time)."""

import argparse
import time

import torch

from kernel_portfolio import contracts as c
from kernel_portfolio import ops, triton_kernels


def host_us(fn, calls):
    for _ in range(200):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return elapsed / calls * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=20000)
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("--calls must be at least 1")
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)
    parts = {
        "torch.softmax": lambda: torch.softmax(x, -1),
        "input validation": lambda: (c.tensor(x, "x", ndim=2), c.rows(x)),
        "output allocation": lambda: torch.empty(x.shape, device=x.device, dtype=x.dtype),
        "device guard": lambda: ops._on(x.device).__enter__(),
        "Triton launch only": lambda: triton_kernels.launch_softmax(x, out, 4),
        "ops.softmax total": lambda: ops.softmax(x),
    }
    print(f"FP16 1024x1024, {args.calls} calls each; host microseconds per call")
    for label, fn in parts.items():
        print(f"{label:20} {host_us(fn, args.calls):7.2f}")


if __name__ == "__main__":
    main()
