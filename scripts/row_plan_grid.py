"""The measurements behind triton_kernels._row_plan: every plan on a grid of shapes.

For each operator (softmax, residual RMSNorm), row count and row width, times the looped
plans (chunk widths 8192, 4096, 2048 and 1024 elements) and the split plans and prints
which family wins. Widths are the ones models use (hidden and vocabulary sizes), all
above the single-block limit of 8192. FP16, CUDA-graph replay, GPU warmed first, all plans
of a cell sampled in shuffled rounds. The table ends with a summary: how far each rule
(the current one and the one it replaced) is from the best plan per shape, and how often
splitting wins at each row count.

    python scripts/row_plan_grid.py --full > results/row-plan-grid.log   # about two minutes
    python scripts/row_plan_grid.py --full > results/row-plan-grid-repeat.log  # noise check
    python scripts/row_plan_grid.py                  # fewer row counts (about a minute)
    python scripts/row_plan_grid.py --quick          # a few rows and widths
"""

import argparse
import random
import statistics

import torch

from kernel_portfolio import ops
from kernel_portfolio import triton_kernels as k
from kernel_portfolio.benchmark import ensure_warm, prepare_timer, sample_ms

ROWS = (1, 4, 16, 64, 96, 128, 256, 512, 1024)
WIDTHS = (12288, 16384, 24576, 28672, 32000, 50257, 65536, 128256, 262144, 1048576)
FULL_ROWS = (1, 2, 4, 8, 16, 32, 64, 80, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
QUICK_ROWS, QUICK_WIDTHS = (1, 16, 64, 256, 1024), (16384, 50257, 262144)
MAX_ELEMENTS = 1 << 27  # 256 MB of FP16 per tensor
LOOPED = (("looped", 8192, 16), ("looped", 4096, 8), ("looped", 2048, 4), ("looped", 1024, 4))
SPLIT = (("split", 2048, 4), ("split", 4096, 8))
ROUNDS = 15
SPLIT_ABOVE = {"softmax": 8192, "rmsnorm": 16384}  # widths where the rule may split


def label(plan):
    return f"{plan[0][0].upper()}{plan[1]}/{plan[2]}"


def measure(call, plans, rng):
    original = k._row_plan
    timers = {}
    try:
        for plan in plans:
            k._row_plan = lambda op, rows, n, device, plan=plan: plan
            call()
            timers[label(plan)] = prepare_timer(call, "graph", 10)
    finally:
        k._row_plan = original
    ensure_warm()
    samples = {name: [] for name in timers}
    for _ in range(ROUNDS):
        names = list(timers)
        rng.shuffle(names)
        for name in names:
            run, count = timers[name]
            samples[name].append(sample_ms(run, count))
    return {name: statistics.median(values) * 1e3 for name, values in samples.items()}


def previous_label(op, rows, width, sms):
    """The rule this one replaced: same split test, then 2048-wide chunks from two rows/SM."""
    if width > SPLIT_ABOVE[op] and rows < sms:
        return label(("split", 4096, 8) if width > 65536 else ("split", 2048, 4))
    return label(("looped", 2048, 4) if rows >= 2 * sms else ("looped", 8192, 16))


def run_shape(rows, width, rng, names):
    """One table row per operator, returned as cells; the tensors are freed on return."""
    x = torch.randn((rows, width), device="cuda", dtype=torch.float16)
    r = torch.randn_like(x)
    w = torch.randn(width, device="cuda", dtype=torch.float16)
    calls = {"softmax": lambda: ops.softmax(x), "rmsnorm": lambda: ops.residual_rmsnorm(x, r, w)}
    cells = []
    for op, call in calls.items():
        us = measure(call, LOOPED + SPLIT, rng)
        loop = min(us[label(p)] for p in LOOPED)
        split = min(us[label(p)] for p in SPLIT)
        chosen = label(k._row_plan(op, rows, width, x.device))
        columns = "".join(f"{us[name]:10.2f}" for name in names)
        print(f"{op:8} {rows:5} {width:8} {columns}{split / loop:12.2f}{chosen:>10}", flush=True)
        cells.append(
            {
                "op": op,
                "rows": rows,
                "width": width,
                "us": us,
                "chosen": chosen,
                "ratio": split / loop,
            }
        )
    return cells


def percent(value):
    return f"{100 * value:5.1f}%"


def summarize(cells, sms):
    """Regret of a rule = its plan's time over the best plan's time for the same shape, minus 1."""
    print()
    print(f"Summary over {len(cells)} operator-shape cells. The regret of a rule is its plan's")
    print("time over the best plan's time in the same cell, minus 1. Repeat runs differ by a few")
    print("percent in a typical cell and by more in about one cell in ten.")
    rules = {
        "current (4096-wide chunks for narrow rows with many rows)": lambda c: c["chosen"],
        "previous (2048-wide chunks from two rows per SM)": lambda c: previous_label(
            c["op"], c["rows"], c["width"], sms
        ),
        "no narrow rule (8192-wide chunks for every looped row)": lambda c: (
            c["chosen"] if c["chosen"].startswith("S") else "L8192/16"
        ),
    }
    for scope in ("both operators", "softmax", "rmsnorm"):
        subset = cells if scope == "both operators" else [c for c in cells if c["op"] == scope]
        if not subset:
            continue
        print()
        print(f"{scope}: {len(subset)} cells")
        print(f"  {'rule':56} {'mean':>7} {'median':>7} {'p90':>7} {'max':>7}  cells over 10%")
        for name, pick in rules.items():
            regrets = sorted(c["us"][pick(c)] / min(c["us"].values()) - 1 for c in subset)
            over = sum(regret > 0.10 for regret in regrets)
            p90 = regrets[int(0.9 * len(regrets))]
            print(
                f"  {name:56} {percent(statistics.fmean(regrets))} "
                f"{percent(statistics.median(regrets))} {percent(p90)} {percent(regrets[-1])}"
                f"  {over} of {len(regrets)}"
            )
    differing = [
        c for c in cells if c["chosen"] != previous_label(c["op"], c["rows"], c["width"], sms)
    ]
    faster = sum(
        c["us"][previous_label(c["op"], c["rows"], c["width"], sms)] >= 1.1 * c["us"][c["chosen"]]
        for c in differing
    )
    slower = sum(
        c["us"][c["chosen"]] >= 1.1 * c["us"][previous_label(c["op"], c["rows"], c["width"], sms)]
        for c in differing
    )
    print()
    print(f"The current and previous rules choose different plans in {len(differing)} cells: the")
    print(f"current plan is at least 10% faster in {faster} of them and at least 10% slower")
    print(f"in {slower}.")
    print()
    print("Best split plan over best looped plan by row count (below 1 the split wins), for the")
    print("widths where the rule may split: softmax above 8192, RMSNorm above 16384.")
    heading = ("op", "rows", "cells", "median", "split wins", "even", "loop wins")
    print(
        f"{heading[0]:8} {heading[1]:>5} {heading[2]:>6} {heading[3]:>7} {heading[4]:>11}", end=""
    )
    print(f" {heading[5]:>5} {heading[6]:>10}")
    for op, above in SPLIT_ABOVE.items():
        for rows in sorted({c["rows"] for c in cells}):
            ratios = [
                c["ratio"]
                for c in cells
                if c["op"] == op and c["rows"] == rows and c["width"] > above
            ]
            if ratios:
                wins = sum(r < 0.95 for r in ratios)
                loses = sum(r > 1.05 for r in ratios)
                print(
                    f"{op:8} {rows:5} {len(ratios):6} {statistics.median(ratios):7.2f} "
                    f"{wins:11} {len(ratios) - wins - loses:5} {loses:10}"
                )
    print("(split wins and loop wins: by more than 5%; even: within 5%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    rows_axis, widths = (QUICK_ROWS, QUICK_WIDTHS) if args.quick else (ROWS, WIDTHS)
    if args.full:
        rows_axis = FULL_ROWS
    torch.manual_seed(2026)
    rng = random.Random(2026)
    sms = k._sm_count(torch.cuda.current_device())
    print(f"{sms} SMs. Median microseconds per call; split/loop is the best split plan's time")
    print("over the best looped plan's, so a value below 1 means splitting wins.")
    names = [label(p) for p in LOOPED + SPLIT]
    header = "".join(f"{name:>10}" for name in names)
    print(f"{'op':8} {'rows':>5} {'width':>8} {header}{'split/loop':>12}{'auto':>10}")
    cells = []
    for width in widths:
        for rows in rows_axis:
            if rows * width <= MAX_ELEMENTS:
                cells += run_shape(rows, width, rng, names)
                torch.cuda.empty_cache()
    summarize(cells, sms)


if __name__ == "__main__":
    main()
