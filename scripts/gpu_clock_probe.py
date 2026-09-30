"""Evidence for benchmark.warm_gpu: how an idle GPU's clocks distort short measurements.

A 2 MiB FP16 softmax takes a few microseconds at boost clocks. After the GPU has been idle
for several seconds it starts at reduced clocks, and a measurement that begins then can run
several times slower until the driver raises them again. Whether and for how long depends
on what else uses the GPU, so this script reports what it sees, including the SM clock and
utilization that nvidia-smi reports when each idle gap ends, and it says so when nothing
slowed the kernel.

Part 1 sleeps for growing gaps and times the softmax right afterwards: the median of the
first 100 ms of calls, and the steady state a second later. Part 2 takes the shortest gap
that slowed the first calls and repeats it with a warm-up burst of several lengths before
the timing.

    python scripts/gpu_clock_probe.py > results/gpu-clock-probe.log
"""

import statistics
import subprocess
import time

import torch

from kernel_portfolio import ops
from kernel_portfolio.benchmark import prepare_timer, sample_ms, warm_gpu

GAPS_S = (2, 5, 10, 20, 40)
BURSTS_MS = (0, 20, 50, 150)
AFTER_MS = (0, 500)
TRIALS = 3
SLOW = 2.0  # a first window this many times slower than the steady state marks the slow state


def smi():
    """(SM clock in MHz, utilization in percent) as nvidia-smi reports them, or None."""
    query = "clocks.sm,utilization.gpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split(",")
        return int(out[0]), int(out[1])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def per_call_us(run, count, repeats=5):
    return statistics.median(sample_ms(run, count) * 1e3 for _ in range(repeats))


def windows(run, count, edges=(0.0, 0.1, 0.4, 0.8, 1.2)):
    """Median microseconds per call in windows of time after the first call."""
    start = time.perf_counter()
    samples = []
    while time.perf_counter() - start < edges[-1]:
        samples.append((time.perf_counter() - start, sample_ms(run, count) * 1e3))
    return [
        statistics.median(v for t, v in samples if low <= t < high)
        for low, high in zip(edges, edges[1:])
    ]


def part_one(run, count):
    print("Part 1: the softmax right after an idle gap; the GPU was warmed before each gap.")
    print(f"{'gap':>6} {'SM clock':>10} {'util':>6}   median us per call, by time after")
    print(f"{'':>6} {'at wake':>10} {'':>6}   the first call (ms):")
    windows_ms = ("0-100", "100-400", "400-800", "800-1200")
    print(f"{'':>24}" + "".join(f"{w:>10}" for w in windows_ms))
    slow_gap = None
    for gap in GAPS_S:
        warm_gpu()
        time.sleep(gap)
        state = smi()
        cells = windows(run, count)
        clock = f"{state[0]} MHz" if state else "?"
        util = f"{state[1]}%" if state else "?"
        print(
            f"{gap:>4} s {clock:>10} {util:>6} " + "".join(f"{v:10.2f}" for v in cells), flush=True
        )
        if slow_gap is None and cells[0] > SLOW * cells[-1]:
            slow_gap = gap
    return slow_gap


def part_two(run, count, gap):
    print()
    print(
        f"Part 2: after a {gap} s idle gap (which slowed the first calls), a warm-up burst of large"
    )
    print(
        f"GEMMs, then a pause, then the softmax; median (min-max) of {TRIALS} trials, us per call."
    )
    print(f"{'burst':>9}" + "".join(f"{f'pause {p} ms':>24}" for p in AFTER_MS))
    for burst in BURSTS_MS:
        cells = []
        for pause in AFTER_MS:
            values = []
            for _ in range(TRIALS):
                time.sleep(gap)
                if burst:
                    warm_gpu(burst / 1000)
                time.sleep(pause / 1000)
                values.append(per_call_us(run, count))
            cells.append(
                f"{statistics.median(values):8.2f} ({min(values):6.2f}-{max(values):6.2f})"
            )
        label = f"{burst} ms" if burst else "none"
        print(f"{label:>9}" + "".join(f"{cell:>24}" for cell in cells), flush=True)


def main():
    torch.manual_seed(2026)
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    run, count = prepare_timer(lambda: ops.softmax(x), "graph", 20)
    warm_gpu(0.01)  # allocates the burst operands and loads the GEMM kernels
    torch.cuda.synchronize()
    print("Softmax 1024 x 1024 FP16, CUDA-graph replay of 20 calls per sample")
    print(f"GPU state before the probe (SM clock MHz, utilization %): {smi()}")
    print()
    slow_gap = part_one(run, count)
    if slow_gap is None:
        print()
        print("No idle gap slowed the softmax: other programs were using the GPU, which keeps its")
        print("clocks up. The warm-up burst is not needed on a busy machine but does no harm.")
    else:
        part_two(run, count, slow_gap)
    print()
    print("benchmark.py warms the GPU for 200 ms before a measurement whenever the previous")
    print("burst ended more than 250 ms earlier.")


if __name__ == "__main__":
    main()
