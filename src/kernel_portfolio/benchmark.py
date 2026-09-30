"""Correctness-gated, reproducible microbenchmarks. See docs/BENCHMARKING.md."""

import argparse
import datetime
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import ops, references
from .environment import describe, gpu_utilization

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
    # Forward plus backward through autograd, compared on the input gradient.
    "softmax_train": ROW_SHAPES,
    "rmsnorm_train": ROW_SHAPES,
}
TRAINING = ("softmax_train", "rmsnorm_train")
# --op all runs the forward suites; --op training runs both training suites. They are
# separate runs because every prepared case stays in GPU memory until sampling ends.
GROUPS = {
    "all": [op for op in SUITES if op not in TRAINING],
    "training": list(TRAINING),
}


def _with_autograd(fn):
    """Run fn with autograd available inside the benchmark's inference mode."""

    def run():
        with torch.inference_mode(False), torch.enable_grad():
            return fn()

    return run


def _training_case(op, shape, dtype, compile_torch):
    """One forward and backward pass per call, returning the input gradient.

    The whole step is timed because a CUDA graph can only replay a backward pass whose
    forward was captured with it. The oracle is the closed-form FP64 gradient.
    """
    rows, width = shape
    itemsize = torch.empty((), dtype=dtype).element_size()
    with torch.inference_mode(False):
        x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        dy = torch.randn(shape, device="cuda", dtype=dtype)
        if op == "softmax_train":
            leaves = (x,)
            forwards = {"torch": lambda x: torch.softmax(x, dim=-1), "triton": ops.softmax}
            y = torch.softmax(x.detach().double(), dim=-1)
            oracle = y * (dy.double() - (y * dy.double()).sum(-1, keepdim=True))
            # Forward: read x, write y. Backward: read y and dy, write dx.
            logical = 5 * x.numel() * itemsize
        else:
            residual = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
            weight = torch.randn(width, device="cuda", dtype=dtype, requires_grad=True)
            leaves = (x, residual, weight)
            forwards = {
                "torch": references.residual_rmsnorm,
                "torch_rms_norm": lambda x, r, w: F.rms_norm(
                    x.float() + r.float(), (width,), w.float(), 1e-5
                ).to(dtype),
                "triton": ops.residual_rmsnorm,
            }
            z = x.detach().double() + residual.detach().double()
            inverse = torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5)
            g = dy.double() * weight.detach().double()
            oracle = inverse * (g - z * inverse.square() * (g * z).mean(-1, keepdim=True))
            del z, g
            # Forward: read x, r, w, write out. Backward: read x, r, w, dy; write dx, dw.
            logical = (7 * x.numel() + 3 * width) * itemsize
    if compile_torch:
        forwards["torch_compiled"] = torch.compile(forwards["torch"], fullgraph=True)

    def step(forward):
        return _with_autograd(lambda: torch.autograd.grad(forward(*leaves), leaves, dy)[0])

    return {name: step(fn) for name, fn in forwards.items()}, oracle.to(dtype), logical


def make_case(op, shape, dtype, compile_torch=False):
    """Return named functions, an independent oracle, and minimal logical I/O bytes."""
    if op in TRAINING:
        return _training_case(op, shape, dtype, compile_torch)
    functions, expected, logical = _forward_case(op, shape, dtype)
    if compile_torch:
        functions["torch_compiled"] = torch.compile(functions["torch"], fullgraph=True)
    return functions, expected, logical


def _forward_case(op, shape, dtype):
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
        from .triton_kernels import SINGLE_BLOCK_MAX

        # num_warps only applies to single-block rows; looped rows pick their own warps.
        if shape[1] <= SINGLE_BLOCK_MAX:
            triton = {
                "triton_4w": lambda: ops.softmax(x),
                "triton_8w": lambda: ops.softmax(x, num_warps=8),
            }
        else:
            triton = {"triton": lambda: ops.softmax(x)}
        return (
            {
                "torch": lambda: torch.softmax(x, dim=-1),
                "torch_decomposed": lambda: references.softmax_decomposed(x),
                **triton,
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
    if op in ("softmax_backward", "softmax_train"):
        # Gradients mix saved rounded probabilities with the upstream gradient.
        return {
            "atol": 1e-4 if dtype == torch.bfloat16 else (1e-5 if dtype == torch.float16 else 1e-6),
            "rtol": 0.016
            if dtype == torch.bfloat16
            else (0.004 if dtype == torch.float16 else 2e-5),
        }
    if op == "rmsnorm_train":
        op = "rmsnorm"
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
        # --no-optional-locks: status only reads. Without it, git takes .git/index.lock
        # to refresh the index, and a status killed by the timeout on a busy machine
        # leaves that lock behind, blocking the user's next commit.
        return subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=here,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
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


def file_sha256(path):
    """SHA-256 with CRLF read as LF, so a Windows working copy and a Git checkout agree."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_hashes():
    """Content identity remains useful before a first Git commit exists."""
    root = Path(__file__).resolve().parent
    return {p.name: file_sha256(p) for p in sorted(root.glob("*.py"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op",
        choices=[*GROUPS, *SUITES],
        default="all",
        help="all: forward suites; training: forward+backward suites; or one operation",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--shape", type=int, nargs="+", help="Custom shape; GEMM order is M N K")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument(
        "--visits",
        type=int,
        default=3,
        help="Spread each case's samples over this many visits across the run",
    )
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
    if args.visits < 1 or args.samples % args.visits:
        parser.error("--samples must be a multiple of --visits")
    if args.shape and (args.op in GROUPS or any(d <= 0 for d in args.shape)):
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
    selected = {op: SUITES[op] for op in GROUPS.get(args.op, [args.op])}
    # Other programs' GPU use, sampled before this run starts and after it ends.
    busy = {"before": gpu_utilization()}
    if busy["before"] and statistics.median(busy["before"]) >= 10:
        print(
            f"note: the GPU was {statistics.median(busy['before']):.0f}% busy before this run; "
            "other programs can shift results between runs (docs/BENCHMARKING.md)",
            file=sys.stderr,
        )
    with torch.inference_mode():
        # Check and prepare every case first, then sample them all together (below).
        prepared = []
        for op, shapes in selected.items():
            for shape in [tuple(args.shape)] if args.shape else shapes:
                functions, expected, logical_bytes = make_case(
                    op, shape, dtype, compile_torch=args.compile_torch
                )
                case = {
                    "op": op,
                    "shape": shape,
                    "dtype": args.dtype,
                    "logical_bytes": logical_bytes,
                    "variants": {},
                }
                timers, errors = {}, {}
                for name, fn in functions.items():
                    result = fn()
                    torch.testing.assert_close(result, expected, **tolerance(op, dtype))
                    errors[name] = (result.float() - expected.float()).abs().max().item()
                    timers[name] = prepare_timer(fn, args.timing, calls_per_timer)
                if op == "matmul":
                    from .triton_kernels import _matmul_tuned

                    # best_config belongs to the last tuned call: record it for this shape now.
                    case["autotune_config"] = str(_matmul_tuned.best_config)
                # A captured graph replays at its inputs' addresses, and only the functions
                # own those inputs: keep them alive for as long as the graph can replay.
                prepared.append(
                    {"case": case, "functions": functions, "timers": timers, "errors": errors}
                )

        # Every case is visited --visits times, in a fresh random order each time. A visit
        # times the case's variants back to back in shuffled rounds, so a ratio compares
        # measurements taken moments apart. A burst of activity from another GPU user then
        # spoils one visit of one case, which the median over all visits discards.
        times = [{name: [] for name in p["timers"]} for p in prepared]
        for _ in range(args.visits):
            for i in rng.sample(range(len(prepared)), len(prepared)):
                timers = prepared[i]["timers"]
                # Untimed first: after other cases ran, a graph's first replay pays to reload
                # it and warm the caches, which timing a case back to back never sees.
                for run, _ in timers.values():
                    run()
                for _ in range(args.samples // args.visits):
                    names = list(timers)
                    rng.shuffle(names)
                    for name in names:
                        run, count = timers[name]
                        times[i][name].append(
                            sample_ms(run, count)
                            if flush is None
                            else sample_cold_ms(run, args.iterations, flush)
                        )

        for p, samples_by_name in zip(prepared, times):
            case, errors = p["case"], p["errors"]
            op, shape = case["op"], case["shape"]
            baseline = statistics.median(samples_by_name["torch"])
            for name, samples in samples_by_name.items():
                stats = summarize(samples)
                ms = stats["median_ms"]
                stats.update(
                    speedup_vs_torch=baseline / ms,
                    max_abs_error=errors[name],
                    logical_gbps=case["logical_bytes"] / (ms * 1e6),
                )
                if op == "matmul":
                    m, n, k = shape
                    stats["tflops"] = 2 * m * n * k / (ms * 1e9)
                case["variants"][name] = stats
                print(f"{op:8} {str(shape):23} {name:18} {ms * 1e3:9.3f} us {baseline / ms:6.2f}x")
            report["cases"].append(case)
        prepared.clear()
    torch.cuda.synchronize()
    time.sleep(0.5)  # let this process's own work drop out of the utilization window
    busy["after"] = gpu_utilization()
    report["gpu_utilization_percent"] = busy
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
