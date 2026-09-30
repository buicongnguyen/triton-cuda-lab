"""Host-side cost of each piece of an eager ops.softmax call (Python dispatch, not GPU time).

Each piece is timed in several interleaved rounds and the table shows the median, so a burst
of CPU load from another program spoils one round of each piece, not one whole piece.
"""

import argparse
import random
import statistics
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
    parser.add_argument("--calls", type=int, default=3000, help="calls per piece per round")
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    if args.calls < 1 or args.rounds < 1:
        parser.error("--calls and --rounds must be at least 1")
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
    rng = random.Random(2026)
    samples = {label: [] for label in parts}
    for _ in range(args.rounds):
        labels = list(parts)
        rng.shuffle(labels)
        for label in labels:
            samples[label].append(host_us(parts[label], args.calls))
    print(
        f"FP16 1024x1024, median of {args.rounds} interleaved rounds of {args.calls} calls; "
        "host microseconds per call"
    )
    for label, values in samples.items():
        print(f"{label:20} {statistics.median(values):7.2f}")


if __name__ == "__main__":
    main()
