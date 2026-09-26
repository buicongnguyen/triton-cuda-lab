"""Host cost per call: eager wrapper vs the torch.compile custom ops (Python dispatch).

Two workloads: one tiny softmax, and a 16-op chain (8 x residual RMSNorm + softmax)
standing in for a model layer stack. Variants run interleaved in one process over
several rounds and the median is reported, so background load on a desktop GPU
affects every variant alike. Device time is not measured here; see kernel-bench.
"""

import argparse
import statistics
import time

import torch

import kernel_portfolio.library  # noqa: F401  (registers torch.ops.kernel_portfolio.*)
from kernel_portfolio import ops

LAYERS = 8


def host_us(fn, calls):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return elapsed / calls * 1e6


def op_softmax(t):
    return torch.ops.kernel_portfolio.softmax(t)


def chain_eager(x, r, w):
    for _ in range(LAYERS):
        x = ops.softmax(ops.residual_rmsnorm(x, r, w))
    return x


def chain_ops(x, r, w):
    for _ in range(LAYERS):
        x = torch.ops.kernel_portfolio.softmax(torch.ops.kernel_portfolio.residual_rmsnorm(x, r, w))
    return x


def measure(title, variants, calls, rounds):
    for fn in variants.values():  # compile, record CUDA graphs, warm up
        for _ in range(20):
            fn()
    samples = {name: [] for name in variants}
    for _ in range(rounds):
        for name, fn in variants.items():
            samples[name].append(host_us(fn, calls))
    print(title)
    for name, values in samples.items():
        print(f"  {name:44} median {statistics.median(values):8.2f}  min {min(values):8.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if args.calls < 1 or args.rounds < 1:
        parser.error("--calls and --rounds must be at least 1")
    torch.manual_seed(2026)
    x = torch.randn((32, 127), device="cuda", dtype=torch.float16)
    r = torch.randn_like(x)
    w = torch.randn(127, device="cuda", dtype=torch.float16)

    single_compiled = torch.compile(op_softmax, fullgraph=True)
    single_graphed = torch.compile(op_softmax, fullgraph=True, mode="reduce-overhead")
    chain_compiled = torch.compile(chain_ops, fullgraph=True)
    chain_graphed = torch.compile(chain_ops, fullgraph=True, mode="reduce-overhead")
    for _ in range(3):
        torch.testing.assert_close(single_graphed(x), torch.softmax(x, -1))
        torch.testing.assert_close(chain_graphed(x, r, w), chain_eager(x, r, w))

    print("FP16 rows of width 127; host microseconds per call")
    measure(
        "One softmax:",
        {
            "torch.softmax, eager": lambda: torch.softmax(x, -1),
            "ops.softmax, eager wrapper": lambda: ops.softmax(x),
            "torch.ops custom op, eager": lambda: op_softmax(x),
            "custom op, torch.compile": lambda: single_compiled(x),
            "custom op, torch.compile reduce-overhead": lambda: single_graphed(x),
        },
        args.calls,
        args.rounds,
    )
    measure(
        f"{2 * LAYERS}-op chain ({LAYERS} x residual RMSNorm + softmax):",
        {
            "ops.* eager wrappers": lambda: chain_eager(x, r, w),
            "custom ops, torch.compile": lambda: chain_compiled(x, r, w),
            "custom ops, torch.compile reduce-overhead": lambda: chain_graphed(x, r, w),
        },
        max(args.calls // 10, 50),
        args.rounds,
    )
    counters = torch._dynamo.utils.counters
    print("dynamo graphs captured:", dict(counters["stats"]))
    print("cudagraph skips:", dict(counters["inductor"].get("cudagraph_skips", {})) or "none")


if __name__ == "__main__":
    main()
