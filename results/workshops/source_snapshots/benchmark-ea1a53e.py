"""Correctness-gated workshop experiments; defaults to learner implementations."""

import argparse
import datetime
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from kernel_portfolio.benchmark import (
    file_sha256,
    l2_flush_buffer,
    prepare_timer,
    revision,
    sample_cold_ms,
    sample_ms,
    summarize,
)
from kernel_portfolio.environment import describe
from learning.check import load_exercise
from learning.workshops.check import ROOT
from learning.workshops.checks import WORKSHOPS
from learning.workshops.gpu_checks import attention_oracle, tolerance

SUITES = {
    "strided_softmax": [(256, 257), (1024, 1024)],
    "softmax_backward": [(256, 1025), (1024, 4097)],
    "fused_gemm": [(127, 65, 33), (512, 512, 512)],
    "split_softmax": [(4, 32769), (64, 131072)],
    "streaming_attention": [(129, 32), (512, 64)],
}


def make_case(name, module, shape, dtype):
    def rand(dims):
        return torch.randn(dims, device="cuda", dtype=dtype)

    metadata = {}
    if name == "strided_softmax":
        rows, width = shape
        x = rand((width, rows)).T
        metadata["input_strides"] = list(x.stride())
        functions = {
            "torch": lambda: x.softmax(-1),
            "direct_4w": lambda: module.softmax(x, num_warps=4),
            "direct_8w": lambda: module.softmax(x, num_warps=8),
            "copy_then_kernel": lambda: module.softmax(x.contiguous()),
        }
        expected = x.double().softmax(-1).to(dtype)
    elif name == "softmax_backward":
        y = rand(shape).float().softmax(-1).to(dtype)
        g = rand(shape)

        def composition():
            yf, gf = y.float(), g.float()
            return (yf * (gf - (yf * gf).sum(-1, keepdim=True))).to(dtype)

        functions = {"torch": composition, "fused_vjp": lambda: module.backward(y, g)}
        expected = (y.double() * (g.double() - (y.double() * g.double()).sum(-1, keepdim=True))).to(
            dtype
        )
        metadata["baseline"] = "explicit FP32 VJP composition; no autograd-engine overhead"
    elif name == "fused_gemm":
        m, n, k = shape
        a, b, bias = rand((m, k)), rand((k, n)), rand((n,))

        def composition():
            return (a.float() @ b.float() + bias.float()).relu().to(dtype)

        functions = {"torch": composition}
        for group in (1, 4, 8):
            functions[f"group_{group}"] = lambda group=group: module.matmul_bias_relu(
                a, b, bias, group_m=group
            )
        functions["group_4_bm64"] = lambda: module.matmul_bias_relu(
            a, b, bias, group_m=4, tile_m=64
        )
        expected = (a.double() @ b.double() + bias.double()).relu().to(dtype)
        metadata["baseline"] = "FP32 GEMM+bias+ReLU then cast, TF32 disabled; includes input casts"
        metadata["warning"] = "Not a speedup claim against tuned half-precision cuBLASLt epilogues"
    elif name == "split_softmax":
        x = rand(shape)
        functions = {"torch": lambda: x.softmax(-1)}
        for chunk in (256, 1024, 4096):
            functions[f"chunk_{chunk}"] = lambda chunk=chunk: module.softmax(x, chunk_size=chunk)
        expected = x.double().softmax(-1).to(dtype)
        metadata["launches_per_candidate_call"] = 3
        metadata["scratch_bytes_by_chunk"] = {
            str(chunk): 8 * shape[0] * (math.ceil(shape[1] / chunk) + 1)
            for chunk in (256, 1024, 4096)
        }
    else:
        q, k, v = (rand(shape) for _ in range(3))
        causal = True

        def dense():
            scores = q.float() @ k.float().T / math.sqrt(shape[1])
            scores = scores.masked_fill(
                torch.ones_like(scores, dtype=torch.bool).triu(1), -math.inf
            )
            return (scores.softmax(-1) @ v.float()).to(dtype)

        functions = {
            "torch": lambda: F.scaled_dot_product_attention(
                q[None, None], k[None, None], v[None, None], is_causal=causal
            )[0, 0],
            "torch_dense_fp32": dense,
            "stream_bn32": lambda: module.attention(q, k, v, causal=causal, block_n=32),
            "stream_bn64": lambda: module.attention(q, k, v, causal=causal, block_n=64),
        }
        expected = attention_oracle(q, k, v, causal)
        metadata.update(
            causal=True,
            baseline="PyTorch SDPA with automatic backend selection",
            dense_score_tensor_bytes=4 * shape[0] ** 2,
            warning="Score tensor size is analytic, not measured peak allocated memory",
        )
    return functions, expected, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=["all", *SUITES], default="all")
    parser.add_argument("--solution", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--timing", choices=["graph", "events"], default="graph")
    parser.add_argument(
        "--cache",
        choices=["warm", "cold"],
        default="warm",
        help="cold flushes L2 before every timed call, so inputs come from DRAM",
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/local/workshops.json"))
    args = parser.parse_args(argv)
    if args.samples < 3 or args.iterations < 1:
        parser.error("Use at least 3 samples and 1 iteration")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("NVIDIA CUDA GPU required; no timings were collected")
    print("REFERENCE SOLUTIONS (not learner completion)" if args.solution else "YOUR EXERCISES")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    rng = random.Random(args.seed)
    flush = l2_flush_buffer() if args.cache == "cold" else None
    calls_per_timer = 1 if flush is not None else args.iterations
    repo = ROOT.parents[1]
    sources = list(ROOT.rglob("*.py")) + list((repo / "src/kernel_portfolio").glob("*.py"))
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": describe(),
        "git": revision(),
        "settings": {**vars(args), "output": str(args.output), "tf32": False},
        "implementation": "reference solutions" if args.solution else "learner exercises",
        "source_sha256": {p.relative_to(repo).as_posix(): file_sha256(p) for p in sorted(sources)},
        "cases": [],
        "skipped": [],
    }
    selected = SUITES if args.op == "all" else {args.op: SUITES[args.op]}
    with torch.inference_mode():
        for name, shapes in selected.items():
            assert WORKSHOPS[name][1] == "GPU"
            module = load_exercise(
                ROOT / ("solutions" if args.solution else "exercises") / f"{name}.py"
            )
            # An unfinished learner exercise raises NotImplementedError on its first
            # call; record it and keep measuring the others instead of losing the run.
            try:
                for shape in shapes:
                    functions, expected, metadata = make_case(
                        name, module, shape, getattr(torch, args.dtype)
                    )
                    timers, times, errors = {}, {}, {}
                    for label, fn in functions.items():
                        actual = fn()
                        torch.testing.assert_close(
                            actual, expected, **tolerance(name, expected.dtype)
                        )
                        errors[label] = (actual.float() - expected.float()).abs().max().item()
                        timers[label] = prepare_timer(fn, args.timing, calls_per_timer)
                        times[label] = []
                    for _ in range(args.samples):
                        labels = list(functions)
                        rng.shuffle(labels)
                        for label in labels:
                            run, count = timers[label]
                            times[label].append(
                                sample_ms(run, count)
                                if flush is None
                                else sample_cold_ms(run, args.iterations, flush)
                            )
                    base = summarize(times["torch"])["median_ms"]
                    case = {
                        "op": name,
                        "shape": shape,
                        "dtype": args.dtype,
                        "metadata": metadata,
                        "variants": {},
                    }
                    for label, samples in times.items():
                        stats = summarize(samples)
                        stats.update(
                            speedup_vs_torch=base / stats["median_ms"], max_abs_error=errors[label]
                        )
                        case["variants"][label] = stats
                        print(
                            f"{name:20} {str(shape):18} {label:20} "
                            f"{stats['median_ms'] * 1000:9.3f} us {stats['speedup_vs_torch']:6.2f}x"
                        )
                    report["cases"].append(case)
                    timers.clear()
            except NotImplementedError as exc:
                print(f"{name:20} skipped: unfinished exercise ({exc})")
                report["skipped"].append({"op": name, "reason": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
