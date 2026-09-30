"""Repeat the FP16 sweep and compare the Triton/PyTorch ratios run to run.

Each run is a full `kernel-bench --op all` sweep (or `--op training`) in this process.
The table lists the speedup of every Triton variant over PyTorch in each run and the
spread (highest over lowest). A comparison whose spread exceeds the size of the effect
being claimed is unresolved on this machine; docs/CASE_STUDIES.md quotes these spreads.

    python scripts/benchmark_stability.py --runs 3 > results/benchmark-stability.log
    python scripts/benchmark_stability.py --runs 3 --cache cold
    python scripts/benchmark_stability.py --runs 3 --op training \
        > results/benchmark-stability-training.log
"""

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

from kernel_portfolio import benchmark
from kernel_portfolio.environment import describe


def one_run(directory, index, cache, op):
    path = Path(directory) / f"run{index}.json"
    argv = ["--op", op, "--dtype", "float16", "--cache", cache, "--output", str(path)]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        benchmark.main(argv)
    report = json.loads(path.read_text(encoding="utf-8"))
    ratios = {}
    for case in report["cases"]:
        for variant, stats in case["variants"].items():
            if not variant.startswith("torch"):
                key = f"{case['op']} {' x '.join(map(str, case['shape']))} {variant}"
                ratios[key] = stats["speedup_vs_torch"]
    busy = (report.get("gpu_utilization_percent") or {}).get("before") or [0]
    return ratios, sorted(busy)[len(busy) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cache", choices=["warm", "cold"], default="warm")
    parser.add_argument("--op", choices=["all", "training"], default="all")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    runs, loads = [], []
    with tempfile.TemporaryDirectory() as directory:
        for index in range(args.runs):
            ratios, load = one_run(directory, index, args.cache, args.op)
            runs.append(ratios)
            loads.append(load)
    gpu = describe().get("gpu", "unknown GPU")
    suite = "training steps" if args.op == "training" else "full sweeps"
    print(f"Triton / PyTorch speedup, FP16, {args.cache} L2, {args.runs} {suite} on {gpu}.")
    print("Other GPU load before each run (percent): " + ", ".join(map(str, loads)))
    print(
        f"\n{'case':38}"
        + "".join(f"{f'run {i + 1}':>9}" for i in range(args.runs))
        + f"{'spread':>9}"
    )
    widest = []
    for key in runs[0]:
        values = [run[key] for run in runs]
        spread = max(values) / min(values) - 1
        widest.append(spread)
        cells = "".join(f"{value:8.2f}x" for value in values)
        print(f"{key:38}{cells}{spread:9.0%}")
    within = sum(spread <= 0.10 for spread in widest)
    summary = f"{within} of {len(widest)} comparisons agree within 10%"
    print(f"\n{summary}; the widest spread is {max(widest):.0%}.")


if __name__ == "__main__":
    main()
