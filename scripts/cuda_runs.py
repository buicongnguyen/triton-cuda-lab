"""Time the standalone CUDA program three times and print one row per run.

The program (cuda/kernels.cu) warms the GPU with a 250 ms compute burst, then samples
all five kernels in rotating order. Each run is a separate process, so the rows show how
much the medians move from launch to launch. Before each run the script waits (at most a
minute) for the GPU to be quiet and prints the load it found. The JSON of the run whose
parallel-softmax median is in the middle is saved as the result.

    python scripts/cuda_runs.py > results/cuda-softmax-runs.log
    python scripts/cuda_runs.py --runs 5 --exe build/cuda_portfolio --json-out /tmp/cuda.json
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "vector_add",
    "softmax_serial",
    "softmax_parallel",
    "softmax_online",
    "softmax_online_scalar",
)
RESULT = ROOT / "results" / "rtx4080super-cuda.json"
QUIET_PERCENT = 10  # utilization at or below this, three seconds running, counts as quiet


def default_exe():
    name = "cuda_portfolio.exe" if sys.platform == "win32" else "cuda_portfolio"
    return ROOT / "build" / name


def utilization():
    """nvidia-smi's GPU utilization in percent, or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return int(out.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def wait_until_quiet(limit_seconds=60):
    """Poll once a second until three readings in a row are quiet; return the last reading."""
    quiet, reading = 0, utilization()
    for _ in range(limit_seconds):
        if reading is None or reading <= QUIET_PERCENT:
            quiet += 1
            if quiet >= 3:
                break
        else:
            quiet = 0
        time.sleep(1)
        reading = utilization()
    return reading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--exe", type=Path, default=default_exe())
    parser.add_argument("--json-out", type=Path, default=RESULT)
    args = parser.parse_args()
    if not args.exe.exists():
        sys.exit(f"{args.exe} not found: build it first (docs/SETUP.md)")
    print(
        f"{args.runs} runs of {args.exe.name} --json, kernels interleaved, 15 samples of 50 "
        "launches after a 250 ms compute warm-up; median us"
    )
    print(
        f"{'run':<4}{'load':>6}{'vector_add':>11}{'serial':>9}{'parallel':>10}{'online(float4)':>16}"
        f"{'online_scalar':>15}{'parallel/online':>17}{'parallel/scalar':>17}"
    )
    kept = []  # (parallel-softmax median, JSON path) per run
    with tempfile.TemporaryDirectory() as directory:
        for run in range(1, args.runs + 1):
            path = Path(directory) / f"run{run}.json"
            load = wait_until_quiet()
            subprocess.run([str(args.exe), "--json", str(path)], check=True, capture_output=True)
            medians = json.loads(path.read_text(encoding="utf-8"))["variants"]
            us = [medians[name]["median_ms"] * 1e3 for name in VARIANTS]
            kept.append((us[2], path))
            shown = "?" if load is None else f"{load}%"
            print(
                f"{run:<4}{shown:>6}{us[0]:11.3f}{us[1]:9.3f}{us[2]:10.3f}{us[3]:16.3f}"
                f"{us[4]:15.3f}{us[2] / us[3]:16.2f}x{us[2] / us[4]:16.2f}x"
            )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        middle = sorted(kept)[len(kept) // 2]
        shutil.copyfile(middle[1], args.json_out)
        print(f"Saved the run with the middle parallel-softmax median ({middle[0]:.3f} us).")


if __name__ == "__main__":
    main()
