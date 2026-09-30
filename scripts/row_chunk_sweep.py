"""Evidence for triton_kernels._row_plan on rows wider than 8192.

Forces each looped (chunk, warps) choice and each split choice (softmax and RMSNorm
only; row sum has no split path), then runs the automatic plan. FP16, CUDA-graph
replay of 20 calls, warm L2, microseconds. Every variant of a shape is captured
first, then all are replayed in a shuffled order in each of 9 rounds, so a burst of
activity from another GPU user lands on all of them alike; the table shows medians.
"""

import argparse
import random
import statistics

import torch

from kernel_portfolio import ops
from kernel_portfolio import triton_kernels as k
from kernel_portfolio.benchmark import prepare_timer, sample_ms

CHOICES = (
    ("looped", 2048, 4),
    ("looped", 4096, 8),
    ("looped", 8192, 8),
    ("looped", 8192, 16),
    ("split", 2048, 4),
    ("split", 4096, 8),
)
# Few rows (split), the rule's crossover side (128 >= 80 SMs: looped), many rows.
SHAPES = ((4, 32769), (64, 131072), (128, 32769), (1024, 16384))
ROUNDS = 9


def capture(fn):
    fn()
    return prepare_timer(fn, "graph", 20)


def interleaved_us(timers, rng):
    samples = {name: [] for name in timers}
    for _ in range(ROUNDS):
        names = list(timers)
        rng.shuffle(names)
        for name in names:
            run, count = timers[name]
            samples[name].append(sample_ms(run, count))
    return {name: statistics.median(values) * 1e3 for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.manual_seed(2026)
    rng = random.Random(2026)
    automatic = k._row_plan
    sms = k._sm_count(torch.cuda.current_device())
    print(
        f"{sms} SMs; automatic plan: split softmax (and RMSNorm wider than 16384) when "
        f"rows < {sms}; otherwise loop, 2048/4w when rows >= {2 * sms}, else 8192/16w"
    )
    header = "".join(f"{f'{p} {c}/{w}w':>16}" for p, c, w in CHOICES)
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
                # A captured graph keeps the plan that was active when it was captured.
                timers = {"torch": capture(baseline)} if baseline else {}
                for choice in CHOICES:
                    if choice[0] == "split" and op == "row_sum":
                        continue
                    k._row_plan = lambda op, rows, n, device, choice=choice: choice
                    timers[choice] = capture(candidate)
                k._row_plan = automatic
                timers["auto"] = capture(candidate)
                us = interleaved_us(timers, rng)
                cells = [f"{us['torch']:10.2f}" if baseline else f"{'-':>10}"]
                cells += [f"{us[c]:16.2f}" if c in us else f"{'-':>16}" for c in CHOICES]
                plan = automatic(op, rows, n, x.device)
                cells.append(f"{us['auto']:12.2f}  ({plan[0]} {plan[1]}/{plan[2]}w)")
                print(f"{rows}x{n:<8} {op:8} " + "".join(cells))
                timers.clear()
    finally:
        k._row_plan = automatic


if __name__ == "__main__":
    main()
