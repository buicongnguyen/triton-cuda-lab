"""Correctness-gated, reproducible microbenchmarks. See docs/BENCHMARKING.md."""

import argparse
import datetime
import hashlib
import json
import random
import statistics
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from . import ops, references
from .environment import describe

# 16384 x 4096 exceeds a 64 MiB L2 even in FP16, so at least one case per memory-bound
# operator measures DRAM rather than cache-resident traffic. The two widest shapes use
# the looped row kernels: few very wide rows, then many wide rows.
ROW_SHAPES = [(32, 127), (1024, 1024), (512, 4097), (16384, 4096), (4, 32769), (64, 131072)]
SUITES = {
    "add": [(257,), (1 << 20,), (1 << 24,)],
    "row_sum": ROW_SHAPES,
    "softmax": ROW_SHAPES,
    "rmsnorm": ROW_SHAPES,
    "matmul": [(127, 255, 65), (512, 512, 512), (1024, 1024, 1024), (4096, 4096, 4096)],
}


def make_case(op, shape, dtype):
    """Return named functions, an independent oracle, and minimal logical I/O bytes."""

    def rand(dims):
        return torch.randn(dims, device="cuda", dtype=dtype)

    itemsize = torch.empty((), dtype=dtype).element_size()
    if op == "add":
        x, y = rand(shape), rand(shape)
        return (
            {
                "torch": lambda: x + y,
                "triton_256": lambda: ops.add(x, y),
                "triton_1024": lambda: ops.add(x, y, block_size=1024),
            },
            (x.double() + y.double()).to(dtype),
            3 * x.numel() * itemsize,
        )
    if op == "matmul":
        m, n, k = shape
        a, b = rand((m, k)), rand((k, n))
        return (
            {
                "torch": lambda: torch.mm(a, b),
                "triton_fixed": lambda: ops.matmul(a, b, autotune=False),
                "triton_tuned": lambda: ops.matmul(a, b),
            },
            (a.double() @ b.double()).to(dtype),
            (m * k + k * n + m * n) * itemsize,
        )
    x = rand(shape)
    if op == "row_sum":
        return (
            {"torch": lambda: x.sum(dim=-1, dtype=torch.float32), "triton": lambda: ops.row_sum(x)},
            x.double().sum(-1).float(),
            (x.numel() * itemsize + shape[0] * 4),
        )
    if op == "softmax":
        return (
            {
                "torch": lambda: torch.softmax(x, dim=-1),
                "torch_decomposed": lambda: references.softmax_decomposed(x),
                "triton_4w": lambda: ops.softmax(x),
                "triton_8w": lambda: ops.softmax(x, num_warps=8),
            },
            (torch.softmax(x.double(), dim=-1).to(dtype)),
            2 * x.numel() * itemsize,
        )
    if op != "rmsnorm":
        raise ValueError(f"Unknown operation {op}")
    residual, weight = rand(shape), rand((shape[1],))
    z = x.double() + residual.double()
    oracle = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5) * weight.double()
    # "torch" is the readable composition (~10 eager kernels); "torch_rms_norm" uses the
    # built-in normalization op with the same FP32 residual semantics, so the gap between
    # them separates "fewer launches" from "a better normalization kernel".
    return (
        {
            "torch": lambda: references.residual_rmsnorm(x, residual, weight),
            "torch_rms_norm": lambda: F.rms_norm(
                x.float() + residual.float(), (shape[1],), weight.float(), 1e-5
            ).to(dtype),
            "triton": lambda: ops.residual_rmsnorm(x, residual, weight),
        },
        oracle.to(dtype),
        ((3 * x.numel() + weight.numel()) * itemsize),
    )


def tolerance(op, dtype):
    if op == "row_sum":
        return {"atol": 2e-4, "rtol": 2e-4}
    if op == "matmul":
        return {
            "atol": 0.06 if dtype == torch.bfloat16 else 0.015,
            "rtol": 0.02 if dtype == torch.bfloat16 else 0.003,
        }
    if op == "softmax":
        return {
            "atol": 1e-5 if dtype == torch.bfloat16 else (1e-6 if dtype == torch.float16 else 2e-7),
            "rtol": 0.01
            if dtype == torch.bfloat16
            else (0.002 if dtype == torch.float16 else 2e-5),
        }
    return {
        "atol": 0.02 if dtype == torch.bfloat16 else (0.002 if dtype == torch.float16 else 2e-6),
        "rtol": 0.02 if dtype == torch.bfloat16 else (0.002 if dtype == torch.float16 else 2e-5),
    }


def prepare_timer(fn, timing, iterations):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    if timing == "graph":
        graph = torch.cuda.CUDAGraph()
        # Compilation/autotuning already happened during correctness and warmup.
        with torch.cuda.graph(graph):
            for _ in range(iterations):
                fn()
        torch.cuda.synchronize()
        return graph.replay, iterations

    def call_batch():
        for _ in range(iterations):
            fn()

    return call_batch, iterations


def sample_ms(run, count):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / count


def l2_flush_buffer():
    """Scratch larger than L2; zeroing it evicts earlier inputs (as triton.testing.do_bench)."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "L2_cache_size", 0)
    return torch.empty(max(2 * l2_bytes, 256 << 20), dtype=torch.int8, device="cuda")


def sample_cold_ms(run, repeats, flush):
    """Mean of single-call timings; an untimed L2 flush precedes each call."""
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    for start, end in pairs:
        flush.zero_()
        start.record()
        run()
        end.record()
    torch.cuda.synchronize()
    return statistics.fmean(start.elapsed_time(end) for start, end in pairs)


def summarize(samples):
    if not samples or any(v <= 0 for v in samples):
        raise ValueError("Timing samples must be positive")
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def revision():
    """Git state of the checkout containing this package, independent of the caller's cwd."""
    here = Path(__file__).resolve().parent

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=here, capture_output=True, text=True, check=False, timeout=5
        )

    try:
        result = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain")
        return {
            "commit": result.stdout.strip() if result.returncode == 0 else None,
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def source_hashes():
    """Content identity remains useful before a first Git commit exists."""
    root = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=["all", *SUITES], default="all")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--shape", type=int, nargs="+", help="Custom shape; GEMM order is M N K")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing", choices=["graph", "events"], default="graph")
    parser.add_argument(
        "--cache",
        choices=["warm", "cold"],
        default="warm",
        help="cold flushes L2 before every timed call, so inputs come from DRAM",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile_torch",
        help="Also compare torch.compile on the primary PyTorch callable",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/local/benchmark.json"))
    args = parser.parse_args(argv)
    if args.samples < 3 or args.iterations < 1:
        parser.error("Use at least 3 samples and 1 iteration")
    if args.shape and (args.op == "all" or any(d <= 0 for d in args.shape)):
        parser.error("A custom positive shape requires one --op")
    if args.shape and len(args.shape) != len(SUITES[args.op][0]):
        parser.error("Shape rank does not match operation")
    if args.dtype == "float32" and args.op in ("all", "matmul"):
        parser.error("GEMM accepts FP16/BF16; use --op for FP32 row/vector kernels")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("An NVIDIA GPU is required; see docs/SETUP.md")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    rng = random.Random(args.seed)
    dtype = getattr(torch, args.dtype)
    flush = l2_flush_buffer() if args.cache == "cold" else None
    calls_per_timer = 1 if flush is not None else args.iterations
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": describe(),
        "git": revision(),
        "settings": {**vars(args), "output": str(args.output)},
        "source_sha256": source_hashes(),
        "cases": [],
    }
    selected = SUITES if args.op == "all" else {args.op: SUITES[args.op]}
    with torch.inference_mode():
        for op, shapes in selected.items():
            for shape in [tuple(args.shape)] if args.shape else shapes:
                functions, expected, logical_bytes = make_case(op, shape, dtype)
                if args.compile_torch:
                    functions["torch_compiled"] = torch.compile(functions["torch"], fullgraph=True)
                times, timers, errors = {}, {}, {}
                for name, fn in functions.items():
                    result = fn()
                    torch.testing.assert_close(result, expected, **tolerance(op, dtype))
                    errors[name] = (result.float() - expected.float()).abs().max().item()
                    timers[name] = prepare_timer(fn, args.timing, calls_per_timer)
                    times[name] = []
                for _ in range(args.samples):
                    names = list(functions)
                    rng.shuffle(names)
                    for name in names:
                        run, count = timers[name]
                        times[name].append(
                            sample_ms(run, count)
                            if flush is None
                            else sample_cold_ms(run, args.iterations, flush)
                        )
                baseline = statistics.median(times["torch"])
                case = {
                    "op": op,
                    "shape": shape,
                    "dtype": args.dtype,
                    "logical_bytes": logical_bytes,
                    "variants": {},
                }
                if op == "matmul":
                    from .triton_kernels import _matmul_tuned

                    case["autotune_config"] = str(_matmul_tuned.best_config)
                for name, samples in times.items():
                    stats = summarize(samples)
                    ms = stats["median_ms"]
                    stats.update(
                        speedup_vs_torch=baseline / ms,
                        max_abs_error=errors[name],
                        logical_gbps=logical_bytes / (ms * 1e6),
                    )
                    if op == "matmul":
                        m, n, k = shape
                        stats["tflops"] = 2 * m * n * k / (ms * 1e9)
                    case["variants"][name] = stats
                    print(
                        f"{op:8} {str(shape):23} {name:18} {ms * 1e3:9.3f} us {baseline / ms:6.2f}x"
                    )
                report["cases"].append(case)
                # Release graph pools before the next shape.
                timers.clear()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
