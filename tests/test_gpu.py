"""Actual CUDA execution. Set KERNEL_REQUIRE_GPU=1 to forbid a skipped GPU suite."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from kernel_portfolio import ops, references
from kernel_portfolio.benchmark import tolerance

# CUDA-graph capture needs PyTorch's caching allocator. Sanitizer runs switch it off
# (PYTORCH_NO_CUDA_MEMORY_CACHING=1), so benchmark tests fall back to event timing.
GRAPHS_OK = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1"
TIMING = "graph" if GRAPHS_OK else "events"


class _GpuCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        available = (
            torch.cuda.is_available()
            and torch.version.hip is None
            and importlib.util.find_spec("triton") is not None
        )
        if not available:
            if os.environ.get("KERNEL_REQUIRE_GPU") == "1":
                raise RuntimeError("GPU validation required but CUDA/Triton unavailable")
            raise unittest.SkipTest("NVIDIA CUDA and Triton required")
        torch.manual_seed(2026)
        cls.dtypes = (torch.float32, torch.float16, torch.bfloat16)
        # Looped row kernels switch to narrower chunks at two rows per SM; test both sides.
        cls.many_rows = 2 * torch.cuda.get_device_properties(0).multi_processor_count


class GpuTests(_GpuCase):
    """Kernel correctness; the target for Compute Sanitizer runs."""

    def test_add(self):
        for dtype in self.dtypes:
            for n in (0, 1, 127, 257, 4097, 65537):
                for block in (128, 256, 512, 1024):
                    with self.subTest(dtype=dtype, n=n, block=block):
                        x = torch.randn(n, dtype=dtype, device="cuda")
                        y = torch.randn_like(x)
                        actual = ops.add(x, y, block_size=block)
                        torch.testing.assert_close(actual, x + y, rtol=0, atol=0)

    def test_row_ops(self):
        for dtype in self.dtypes:
            for rows, width in (
                (0, 3),
                (1, 1),
                (7, 33),
                (32, 127),
                (5, 1024),
                (3, 4097),
                (2, 8192),
                (3, 8193),
                (2, 32769),
                (1, 131072),
                (self.many_rows, 8195),
            ):
                for padded in (False, True):
                    with self.subTest(dtype=dtype, shape=(rows, width), padded=padded):
                        backing = torch.randn(
                            (rows, width + (7 if padded else 0)), device="cuda", dtype=dtype
                        )
                        x = backing[:, :width]
                        before = backing.clone()
                        expected_sum = x.double().sum(-1).float()
                        torch.testing.assert_close(
                            ops.row_sum(x), expected_sum, **tolerance("row_sum", dtype)
                        )
                        for warps in (4, 8):
                            actual = ops.softmax(x, num_warps=warps)
                            expected = torch.softmax(x.double(), -1).to(dtype)
                            torch.testing.assert_close(
                                actual, expected, **tolerance("softmax", dtype)
                            )
                            torch.testing.assert_close(
                                actual.float().sum(-1),
                                torch.ones(rows, device="cuda"),
                                atol=0.004,
                                rtol=0.004,
                            )
                        torch.testing.assert_close(backing, before, atol=0, rtol=0)

    def test_extreme_softmax(self):
        for dtype in self.dtypes:
            x = torch.tensor(
                [[10000.0, 9999.0, -10000.0], [-9000.0, -9000.0, -9000.0], [0.0, 0.0, 0.0]],
                device="cuda",
                dtype=dtype,
            )
            actual = ops.softmax(x)
            self.assertTrue(torch.isfinite(actual).all().item())
            torch.testing.assert_close(
                actual, torch.softmax(x.double(), -1).to(dtype), **tolerance("softmax", dtype)
            )

    def test_extreme_softmax_wide_rows(self):
        for dtype in self.dtypes:
            x = torch.zeros((3, 16385), device="cuda", dtype=dtype)
            x[0] = 1000
            x[1] = -1000
            x[2] = -1000
            x[2, -1] = 1000
            actual = ops.softmax(x)
            self.assertTrue(torch.isfinite(actual).all().item())
            torch.testing.assert_close(
                actual, torch.softmax(x.double(), -1).to(dtype), **tolerance("softmax", dtype)
            )

    def test_masked_softmax(self):
        """-inf scores (attention masks, padding) give probability 0 at every width."""
        for dtype in self.dtypes:
            for rows, width in ((3, 33), (3, 4097), (3, 8193), (self.many_rows, 8195), (2, 32769)):
                with self.subTest(dtype=dtype, shape=(rows, width)):
                    x = torch.randn((rows, width), device="cuda", dtype=dtype)
                    # A masked prefix longer than one chunk, plus scattered masked entries.
                    x[:, : width // 3] = float("-inf")
                    x[:, 1::5] = float("-inf")
                    actual = ops.softmax(x)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    torch.testing.assert_close(
                        actual,
                        torch.softmax(x.double(), -1).to(dtype),
                        **tolerance("softmax", dtype),
                    )
            for width in (33, 8193):
                with self.subTest(dtype=dtype, fully_masked=width):
                    # No finite score: NaN, exactly as torch.softmax.
                    x = torch.full((2, width), float("-inf"), device="cuda", dtype=dtype)
                    self.assertTrue(torch.isnan(ops.softmax(x)).all().item())

    def test_rmsnorm(self):
        for dtype in self.dtypes:
            for rows, width in (
                (0, 3),
                (1, 1),
                (7, 33),
                (5, 1024),
                (3, 4097),
                (2, 8192),
                (3, 8193),
                (2, 32769),
                (self.many_rows, 8195),
            ):
                with self.subTest(dtype=dtype, width=width):
                    x = torch.randn((rows, width + 3), device="cuda", dtype=dtype)[:, :width]
                    r = torch.randn((rows, width + 5), device="cuda", dtype=dtype)[:, :width]
                    w = torch.randn(width, device="cuda", dtype=dtype)
                    saved = [t.clone() for t in (x, r, w)]
                    z = x.double() + r.double()
                    expected = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5)
                    expected = (expected * w.double()).to(dtype)
                    torch.testing.assert_close(
                        ops.residual_rmsnorm(x, r, w), expected, **tolerance("rmsnorm", dtype)
                    )
                    for a, b in zip((x, r, w), saved):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
            x = torch.zeros((3, 33), device="cuda", dtype=dtype)
            torch.testing.assert_close(ops.residual_rmsnorm(x, x, torch.ones_like(x[0])), x)

    def test_gemm(self):
        for dtype in (torch.float16, torch.bfloat16):
            for m, n, k in (
                (1, 1, 1),
                (31, 65, 33),
                (127, 255, 65),
                (128, 128, 128),
                (0, 3, 4),
                (3, 0, 4),
                (3, 4, 0),
            ):
                with self.subTest(dtype=dtype, shape=(m, n, k)):
                    a = torch.randn((m, k), device="cuda", dtype=dtype)
                    b = torch.randn((k, n), device="cuda", dtype=dtype)
                    expected = (a.double() @ b.double()).to(dtype)
                    for autotune in (False, True):
                        torch.testing.assert_close(
                            ops.matmul(a, b, autotune=autotune),
                            expected,
                            **tolerance("matmul", dtype),
                        )

    def test_every_gemm_config(self):
        import triton

        from kernel_portfolio import triton_kernels as k

        tried = 0
        for config in k._matmul_tuned.configs:
            # 257 rows leave an incomplete final GROUP_M group for every tile height.
            for m, n, kk in ((127, 255, 65), (257, 129, 96)):
                with self.subTest(config=str(config), shape=(m, n, kk)):
                    a = torch.randn((m, kk), device="cuda", dtype=torch.float16)
                    b = torch.randn((kk, n), device="cuda", dtype=torch.float16)
                    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
                    meta = config.kwargs
                    grid = (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
                    try:
                        k._matmul[grid](
                            a,
                            b,
                            out,
                            m,
                            k.m_bucket(m),
                            n,
                            kk,
                            **meta,
                            num_warps=config.num_warps,
                            num_stages=config.num_stages,
                        )
                    except triton.runtime.errors.OutOfResources:
                        continue  # the autotuner skips these configs too
                    torch.testing.assert_close(
                        out, (a.double() @ b.double()).half(), **tolerance("matmul", torch.float16)
                    )
                    tried += 1
        self.assertGreater(tried, 0)

    def test_invalid_gpu_inputs(self):
        x = torch.ones((2, 3), device="cuda")
        for call in (
            lambda: ops.softmax(x[:, ::2]),
            lambda: ops.softmax(torch.ones((1, (1 << 20) + 1), device="cuda")),
            lambda: ops.softmax(x, num_warps=3),
            lambda: ops.softmax(x.clone().requires_grad_()),
            lambda: ops.residual_rmsnorm(x, x, x[0], eps=0),
            lambda: ops.residual_rmsnorm(x, x, x[0, :2]),
            lambda: ops.add(x[0], x[0, :2]),
            lambda: ops.add(x[0], x[0], block_size=17),
        ):
            with self.assertRaises(ValueError):
                call()
        with self.assertRaises(TypeError):
            ops.matmul(x, x.T.contiguous())

    def test_inference_with_parameters(self):
        x = torch.randn((3, 33), device="cuda")
        weight = torch.nn.Parameter(torch.randn(33, device="cuda"))
        expected = references.residual_rmsnorm(x, x, weight.detach())
        with torch.no_grad():
            actual = ops.residual_rmsnorm(x, x, weight)
        torch.testing.assert_close(actual, expected, **tolerance("rmsnorm", torch.float32))
        with self.assertRaisesRegex(ValueError, "forward-only"):
            ops.residual_rmsnorm(x, x, weight)

    def test_varying_sizes_reuse_compiled_kernels(self):
        """Per-request sizes (length, rows, stride, GEMM M) must not recompile or retune."""
        from kernel_portfolio import triton_kernels as k

        def compiled(jit_fn):
            try:
                return sum(len(entry[0]) for entry in jit_fn.device_caches.values())
            except (AttributeError, TypeError, IndexError):
                self.skipTest("This Triton version does not expose JIT caches this way")

        # Odd lengths share BLOCK and Triton's divisibility specialization class.
        ops.add(torch.randn(257, device="cuda"), torch.randn(257, device="cuda"))
        before = compiled(k._add)
        for n in (259, 1001, 4097):
            ops.add(torch.randn(n, device="cuda"), torch.randn(n, device="cuda"))
        self.assertEqual(compiled(k._add), before, "a new vector length must not recompile")

        # Row width is compile-time by design; row count and (odd) row stride are not.
        ops.softmax(torch.randn((3, 33), device="cuda"))
        before = compiled(k._softmax)
        for rows, stride in ((5, 33), (64, 35), (1, 37)):
            ops.softmax(torch.randn((rows, stride), device="cuda")[:, :33])
        self.assertEqual(compiled(k._softmax), before, "new rows/strides must not recompile")

        tuner_cache = getattr(k._matmul_tuned, "cache", None)
        if tuner_cache is None:
            self.skipTest("This Triton version does not expose the autotuner cache")
        b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
        ops.matmul(torch.randn((33, 64), device="cuda", dtype=torch.float16), b)
        tuned = len(tuner_cache)
        ops.matmul(torch.randn((40, 64), device="cuda", dtype=torch.float16), b)
        self.assertEqual(len(tuner_cache), tuned, "M=33 and M=40 share one tuning bucket")
        ops.matmul(torch.randn((100, 64), device="cuda", dtype=torch.float16), b)
        self.assertEqual(len(tuner_cache), tuned + 1, "M=100 starts a new bucket")

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randn((5, 33), device="cuda")
            out = ops.softmax(x)
            expected = torch.softmax(x, -1)
        stream.synchronize()
        torch.testing.assert_close(out, expected)

    @unittest.skipUnless(torch.cuda.device_count() > 1, "Needs two NVIDIA GPUs")
    def test_device_guard(self):
        with torch.cuda.device(0):
            x = torch.randn((3, 33), device="cuda:1")
            out = ops.softmax(x)
            self.assertEqual(torch.cuda.current_device(), 0)
            torch.testing.assert_close(out, torch.softmax(x, -1))


class BenchmarkSmokeTests(_GpuCase):
    """Benchmark CLIs end to end. Kept apart from GpuTests: under racecheck the
    repeated calls and 256 MB L2 flushes take hours without covering new kernels."""

    def test_benchmark_cli_modes(self):
        from kernel_portfolio import benchmark

        with tempfile.TemporaryDirectory() as directory:
            modes = [("events", "warm"), ("events", "cold")]
            if GRAPHS_OK:
                modes += [("graph", "warm"), ("graph", "cold")]
            for timing, cache in modes:
                with self.subTest(timing=timing, cache=cache):
                    path = Path(directory) / f"{timing}-{cache}.json"
                    argv = ["--op", "rmsnorm", "--shape", "8", "33", "--samples", "3"]
                    argv += ["--iterations", "2", "--timing", timing, "--cache", cache]
                    with contextlib.redirect_stdout(io.StringIO()):
                        benchmark.main([*argv, "--output", str(path)])
                    report = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(report["settings"]["cache"], cache)
                    variants = report["cases"][0]["variants"]
                    self.assertEqual(set(variants), {"torch", "torch_rms_norm", "triton"})
                    for stats in variants.values():
                        self.assertEqual(len(stats["samples_ms"]), 3)
                        self.assertGreater(stats["median_ms"], 0)
            path = Path(directory) / "compiled.json"
            argv = ["--op", "add", "--shape", "257", "--samples", "3", "--iterations", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.main([*argv, "--compile", "--timing", TIMING, "--output", str(path)])
            variants = json.loads(path.read_text(encoding="utf-8"))["cases"][0]["variants"]
            self.assertIn("torch_compiled", variants)

    def test_benchmark_interleaves_a_whole_suite(self):
        """All cases are prepared, then sampled together: every case's inputs must stay
        alive until its captured graphs have finished replaying."""
        from kernel_portfolio import benchmark

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "softmax.json"
            argv = ["--op", "softmax", "--samples", "3", "--iterations", "2", "--timing", TIMING]
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.main([*argv, "--output", str(path)])
            cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
            self.assertEqual([tuple(c["shape"]) for c in cases], benchmark.ROW_SHAPES)
            for case in cases:
                # Looped widths have one Triton variant; num_warps only applies up to 8192.
                wide = case["shape"][1] > 8192
                self.assertEqual("triton" in case["variants"], wide)
                self.assertEqual("triton_8w" in case["variants"], not wide)

    def test_workshop_benchmark_cold_cache(self):
        from learning.provenance import measured_workshop_hashes
        from learning.workshops import benchmark

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workshop.json"
            argv = ["--op", "strided_softmax", "--solution", "--samples", "3", "--iterations", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.main(
                    [*argv, "--cache", "cold", "--timing", TIMING, "--output", str(path)]
                )
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["settings"]["cache"], "cold")
            self.assertEqual(len(report["cases"]), 2)
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(measured_workshop_hashes(report), report["source_sha256"])
            self.assertFalse(any("/exercises/" in path for path in report["source_sha256"]))


class CompileTests(_GpuCase):
    """kernel_portfolio.library: the same kernels as torch.compile-friendly custom ops."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import kernel_portfolio.library  # noqa: F401  (registers torch.ops.kernel_portfolio.*)

        torch.manual_seed(2026)
        cls.x = torch.randn((8, 33), device="cuda", dtype=torch.float16)
        cls.r = torch.randn_like(cls.x)
        cls.w = torch.randn(33, device="cuda", dtype=torch.float16)
        cls.a = torch.randn((40, 33), device="cuda", dtype=torch.float16)
        cls.b = torch.randn((33, 24), device="cuda", dtype=torch.float16)

    def test_custom_op_registration(self):
        ns = torch.ops.kernel_portfolio
        for op, args in (
            (ns.softmax.default, (self.x,)),
            (ns.row_sum.default, (self.x,)),
            (ns.residual_rmsnorm.default, (self.x, self.r, self.w, 1e-5)),
            (ns.matmul.default, (self.a, self.b)),
            (ns.add.default, (self.x[0], self.r[0])),
        ):
            with self.subTest(op=str(op)):
                torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))

    def test_custom_ops_are_forward_only(self):
        ns = torch.ops.kernel_portfolio
        for op, args in (
            (ns.softmax, (self.x,)),
            (ns.row_sum, (self.x,)),
            (ns.residual_rmsnorm, (self.x, self.r, self.w)),
            (ns.matmul, (self.a, self.b)),
            (ns.add, (self.x[0], self.r[0])),
        ):
            with self.subTest(op=str(op)):
                inputs = tuple(x.detach().requires_grad_() for x in args)
                with torch.enable_grad():
                    out = op(*inputs)
                    with self.assertRaisesRegex(RuntimeError, "no autograd formula"):
                        out.sum().backward()
                with torch.no_grad():
                    inference = op(*inputs)
                self.assertFalse(inference.requires_grad)
                torch.testing.assert_close(inference, out.detach())

    @staticmethod
    def model(x, r, w, a, b):
        ns = torch.ops.kernel_portfolio
        y = ns.softmax(ns.residual_rmsnorm(x, r, w, 1e-5) * 2)
        return y, ns.matmul(a, b)

    def test_compiled_graph_matches_eager(self):
        expected = (
            ops.softmax(ops.residual_rmsnorm(self.x, self.r, self.w) * 2),
            ops.matmul(self.a, self.b),
        )
        modes = [{"fullgraph": True}]
        if GRAPHS_OK:  # reduce-overhead records CUDA graphs
            modes.append({"fullgraph": True, "mode": "reduce-overhead"})
        for kwargs in modes:
            with self.subTest(**kwargs):
                compiled = torch.compile(self.model, **kwargs)
                for _ in range(3):  # reduce-overhead records on a later call
                    actual = compiled(self.x, self.r, self.w, self.a, self.b)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want)

    def test_dynamic_row_count(self):
        compiled = torch.compile(self.model, fullgraph=True, dynamic=True)
        for rows in (5, 17, 64):
            x = torch.randn((rows, 33), device="cuda", dtype=torch.float16)
            a = torch.randn((rows, 33), device="cuda", dtype=torch.float16)
            y, c = compiled(x, torch.zeros_like(x), self.w, a, self.b)
            torch.testing.assert_close(y, ops.softmax(ops.residual_rmsnorm(x, x * 0, self.w) * 2))
            torch.testing.assert_close(c, ops.matmul(a, self.b))


class CheckerMutationTests(_GpuCase):
    """The course checkers must reject known-broken kernels, not only accept correct ones.

    Each mutation edits one line of a reference solution. Only the check aimed at that bug
    runs, so the broken kernels never touch memory outside their own allocations.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def failed_checks(self, relative, label, cases, mutation=None):
        from learning.check import load_exercise

        source = (self.ROOT / relative).read_text(encoding="utf-8")
        if mutation:
            old, new = mutation
            self.assertEqual(source.count(old), 1, f"{relative} changed; update this mutation")
            source = source.replace(old, new)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / Path(relative).name
            path.write_text(source, encoding="utf-8")
            module = load_exercise(path)
            selected = [check for name, check in cases if label in name]
            self.assertTrue(selected, f"no check labelled {label!r}")
            failed = 0
            for check in selected:
                try:
                    check(module)
                except AssertionError:
                    failed += 1
            return failed, len(selected)

    def test_add_checker_rejects_unmasked_store(self):
        from learning.gpu_checks import gpu_cases

        args = (
            "learning/solutions/triton_add.py",
            "no store past the end",
            gpu_cases("triton_add"),
        )
        self.assertEqual(self.failed_checks(*args)[0], 0)
        unmasked = ("tl.store(OUT + index, x + y, mask=valid)", "tl.store(OUT + index, x + y)")
        failed, total = self.failed_checks(*args, mutation=unmasked)
        self.assertEqual(failed, total)

    def test_gemm_checker_rejects_early_rounding(self):
        from learning.workshops.gpu_checks import gpu_cases

        args = ("learning/workshops/solutions/fused_gemm.py", "exact FP16", gpu_cases("fused_gemm"))
        self.assertEqual(self.failed_checks(*args)[0], 0)
        for mutation in (
            (
                "acc = tl.dot(a, b, acc)",
                "acc = tl.dot(a, b, acc).to(A.dtype.element_ty).to(tl.float32)",
            ),
            ("out = tl.maximum(acc + bias", "out = tl.maximum(acc.to(OUT.dtype.element_ty) + bias"),
        ):
            with self.subTest(mutation=mutation[1]):
                self.assertEqual(self.failed_checks(*args, mutation=mutation)[0], 1)


if __name__ == "__main__":
    unittest.main()
