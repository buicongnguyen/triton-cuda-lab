"""Compile a representative kernel and save its IR/PTX plus resource metadata."""

import argparse
import json
from pathlib import Path

import torch

from kernel_portfolio.environment import describe
from kernel_portfolio.triton_kernels import launch_matmul, launch_softmax

DEFAULT_SHAPES = {"softmax": [1024, 1024], "matmul": [1024, 1024, 1024]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=["softmax", "matmul"], default="softmax")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        help="softmax: ROWS WIDTH (default 1024 1024); matmul: M N K (default 1024 1024 1024)",
    )
    parser.add_argument("--output", type=Path, default=Path("results/local/ir"))
    args = parser.parse_args()
    shape = args.shape or DEFAULT_SHAPES[args.op]
    if len(shape) != len(DEFAULT_SHAPES[args.op]) or min(shape) < 1:
        parser.error(f"--shape for {args.op} needs {len(DEFAULT_SHAPES[args.op])} positive sizes")
    torch.manual_seed(2026)
    if args.op == "softmax":
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        out = torch.empty_like(x)
        compiled = launch_softmax(x, out, 4)
        expected = torch.softmax(x, -1)
    else:
        m, n, k = shape
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        out = torch.empty((m, n), device="cuda", dtype=torch.float16)
        compiled = launch_matmul(a, b, out, False)
        expected = (a.double() @ b.double()).half()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out, expected, atol=0.015 if args.op == "matmul" else 1e-6, rtol=0.003
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stages = []
    for stage in ("ttir", "ttgir", "llir", "ptx"):
        assembly = compiled.asm.get(stage)
        if isinstance(assembly, str):
            (args.output / f"{args.op}.{stage}").write_text(assembly, encoding="utf-8")
            stages.append(stage)
    summary = {
        "op": args.op,
        "shape": shape,
        "dtype": "float16",
        "stages": stages,
        "registers_per_thread": getattr(compiled, "n_regs", None),
        "spills": getattr(compiled, "n_spills", None),
        "shared_memory_bytes": getattr(compiled.metadata, "shared", None),
        "environment": describe(),
    }
    (args.output / f"{args.op}-metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
