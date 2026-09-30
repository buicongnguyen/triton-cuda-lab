"""Which kernels does torch.compile generate for residual RMSNorm and softmax rows?

Compiles the PyTorch reference compositions, captures Inductor's generated code and
lists each Triton kernel with its launch sizes: xnumel is the number of outputs a
kernel covers (rows, or row pieces for a split reduction) and r0_numel the length
each one reduces. Explains the wide-row results in docs/CASE_STUDIES.md.

    python scripts/inspect_inductor.py > results/inductor-wide-rows.log
"""

import argparse
import re

import torch
from torch._inductor.utils import run_and_get_code

from kernel_portfolio import references

CASES = (
    ("residual RMSNorm FP16", references.residual_rmsnorm, torch.float16, 3),
    ("softmax FP32", lambda x: torch.softmax(x, -1), torch.float32, 1),
)
SHAPES = ((1024, 1024), (64, 131072))


def inputs(count, shape, dtype):
    tensors = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(min(count, 2))]
    if count == 3:
        tensors.append(torch.randn(shape[1], device="cuda", dtype=dtype))
    return tensors


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.manual_seed(2026)
    print(f"torch {torch.__version__}; kernels in launch order")
    for label, fn, dtype, count in CASES:
        for shape in SHAPES:
            torch._dynamo.reset()
            _, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True), *inputs(count, shape, dtype)
            )
            code = "\n".join(codes)
            launches = re.findall(r"(triton_\w+)\.run\(", code)
            plural = "" if len(launches) == 1 else "es"
            print(f"\n{label}, {shape[0]} x {shape[1]}: {len(launches)} kernel launch{plural}")
            for kernel in launches:
                body = code[code.index(f"def {kernel}(") :]
                body = body[: body.find("\n\n\n")]
                sizes = dict(re.findall(r"\b(xnumel|r0_numel)\s*=\s*(\d+)", body))
                kind = {"red": "looped reduction", "per": "one-block reduction", "poi": "pointwise"}
                role = kind.get(kernel.split("_")[1], "other")
                print(f"  {kernel}: {role}; " + ", ".join(f"{k}={v}" for k, v in sizes.items()))


if __name__ == "__main__":
    main()
