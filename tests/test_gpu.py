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
        # Rows past the single-block limit take one of three plans (see _row_plan): split
        # across programs (softmax; RMSNorm above width 16384) below one row per SM, looped
        # in 4096-wide chunks from one row per SM if the row is at most 28672 wide, and
        # looped in 8192-wide chunks otherwise. Widths and row counts below reach each plan.
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        cls.sm_rows, cls.many_rows = sms, 2 * sms


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
                (self.sm_rows, 8193),
                (self.many_rows, 8195),
                (self.sm_rows, 32769),
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
            for rows, width in (
                (3, 33),
                (3, 4097),
                (3, 8193),
                (self.sm_rows, 8193),
                (self.many_rows, 8195),
                (self.sm_rows, 32769),
                (2, 32769),
            ):
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
            for rows, width in ((2, 33), (2, 8193), (self.sm_rows, 8193)):
                with self.subTest(dtype=dtype, fully_masked=(rows, width)):
                    # No finite score: NaN, exactly as torch.softmax.
                    x = torch.full((rows, width), float("-inf"), device="cuda", dtype=dtype)
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
                (4, 20000),
                (self.sm_rows, 8193),
                (self.many_rows, 8195),
                (self.sm_rows, 32769),
            ):
                with self.subTest(dtype=dtype, shape=(rows, width)):
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

    def grad_shapes(self):
        # One shape or more per row plan: single block, split, both looped chunkings.
        return (
            (0, 33),
            (1, 1),
            (7, 33),
            (3, 4097),
            (3, 8193),
            (2, 32769),
            (1, 131072),
            (4, 20000),
            (self.sm_rows, 8193),
            (self.many_rows, 8195),
            (self.sm_rows, 32769),
        )

    @staticmethod
    def upstream(rows, width, dtype, layout):
        """Upstream gradients as autograd delivers them: dense, transposed or broadcast.

        `row_broadcast` (strides 1, 0) is what row_sum hands back, `scalar` (0, 0) what a
        summed loss hands back, and `shifted` has a large mean: the row dot product
        y * dy of a softmax is then O(1) instead of tiny, so a lost term shows up.
        """
        if layout == "transposed":
            return torch.randn((width, rows), device="cuda", dtype=dtype).T
        if layout == "expanded":
            return torch.randn((1, width), device="cuda", dtype=dtype).expand(rows, width)
        if layout == "row_broadcast":
            return torch.randn((rows, 1), device="cuda", dtype=dtype).expand(rows, width)
        if layout == "scalar":
            return torch.randn((), device="cuda", dtype=dtype).expand(rows, width)
        if layout == "shifted":
            return torch.randn((rows, width), device="cuda", dtype=dtype) + 2
        return torch.randn((rows, width), device="cuda", dtype=dtype)

    @staticmethod
    def layouts(dtype, extra):
        """Three layouts in every dtype; the extra ones in FP16 alone (they do not depend on it)."""
        return ("contiguous", "transposed", "expanded") + (extra if dtype == torch.float16 else ())

    def test_softmax_backward(self):
        """Autograd through ops.softmax matches FP64 autograd for every row plan."""
        for dtype in self.dtypes:
            for rows, width in self.grad_shapes():
                for layout in self.layouts(dtype, ("row_broadcast", "scalar", "shifted")):
                    with self.subTest(dtype=dtype, shape=(rows, width), grad=layout):
                        x = torch.randn((rows, width), device="cuda", dtype=dtype)
                        dy = self.upstream(rows, width, dtype, layout)
                        leaf = x.clone().requires_grad_()
                        ops.softmax(leaf).backward(dy)
                        x64 = x.double().requires_grad_()
                        torch.softmax(x64, -1).backward(dy.double())
                        torch.testing.assert_close(
                            leaf.grad, x64.grad.to(dtype), **tolerance("softmax_backward", dtype)
                        )

    def test_rmsnorm_backward(self):
        """dx, dresidual and dweight match FP64 autograd; dweight is deterministic."""
        for dtype in self.dtypes:
            tol = tolerance("rmsnorm", dtype)
            for rows, width in self.grad_shapes():
                for layout in self.layouts(dtype, ("row_broadcast", "scalar")):
                    with self.subTest(dtype=dtype, shape=(rows, width), grad=layout):
                        # Views with row gaps: the padding must receive zero gradient.
                        bx = torch.randn((rows, width + 3), device="cuda", dtype=dtype)
                        br = torch.randn((rows, width + 5), device="cuda", dtype=dtype)
                        w = torch.randn(width, device="cuda", dtype=dtype)
                        bx, br, w = (t.requires_grad_() for t in (bx, br, w))
                        x, r = bx[:, :width], br[:, :width]
                        dy = self.upstream(rows, width, dtype, layout)
                        ops.residual_rmsnorm(x, r, w).backward(dy)
                        x64, r64, w64 = (t.detach().double().requires_grad_() for t in (x, r, w))
                        z = x64 + r64
                        rms = torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5)
                        (z * rms * w64).backward(dy.double())
                        torch.testing.assert_close(bx.grad[:, :width], x64.grad.to(dtype), **tol)
                        torch.testing.assert_close(br.grad[:, :width], r64.grad.to(dtype), **tol)
                        # dw adds one FP32 term per row; where the terms cancel, its absolute
                        # rounding error grows like sqrt(rows) times one term's.
                        dw_tol = {**tol, "atol": tol["atol"] * max(1.0, rows**0.5)}
                        torch.testing.assert_close(w.grad, w64.grad.to(dtype), **dw_tol)
                        self.assertEqual(bx.grad[:, width:].abs().sum().item(), 0)
            x = torch.randn((self.many_rows, 4097), device="cuda", dtype=dtype)
            args = (torch.randn_like(x), x, torch.randn_like(x), torch.randn_like(x[0]))
            first, second = (ops.residual_rmsnorm_backward(*args) for _ in range(2))
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_rmsnorm_backward_with_correlated_gradient(self):
        """The row term z * sum(dy * w * z) is O(1) only if dy follows z and w has a mean."""
        for dtype in (torch.float16, torch.bfloat16):
            tol = tolerance("rmsnorm", dtype)
            for rows, width in self.grad_shapes():
                if not rows:
                    continue
                with self.subTest(dtype=dtype, shape=(rows, width)):
                    x = torch.randn((rows, width), device="cuda", dtype=dtype, requires_grad=True)
                    r = torch.randn((rows, width), device="cuda", dtype=dtype, requires_grad=True)
                    w = (1 + 0.5 * torch.randn(width, device="cuda")).to(dtype).requires_grad_()
                    dy = (x + r).detach()
                    ops.residual_rmsnorm(x, r, w).backward(dy)
                    x64, r64, w64 = (t.detach().double().requires_grad_() for t in (x, r, w))
                    z = x64 + r64
                    (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5) * w64).backward(
                        dy.double()
                    )
                    torch.testing.assert_close(x.grad, x64.grad.to(dtype), **tol)
                    torch.testing.assert_close(r.grad, r64.grad.to(dtype), **tol)
                    dw_tol = {**tol, "atol": tol["atol"] * max(1.0, rows**0.5)}
                    torch.testing.assert_close(w.grad, w64.grad.to(dtype), **dw_tol)

    def test_add_and_row_sum_gradients(self):
        """add passes the gradient through; row_sum broadcasts it (both exact)."""
        for dtype in self.dtypes:
            for n in (0, 1, 257):
                with self.subTest(dtype=dtype, op="add", n=n):
                    x = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True)
                    y = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True)
                    w = torch.randn(n, device="cuda", dtype=dtype)
                    (ops.add(x, y) * w).sum().backward()
                    torch.testing.assert_close(x.grad, w, atol=0, rtol=0)
                    torch.testing.assert_close(y.grad, w, atol=0, rtol=0)
            for rows, width in ((0, 3), (1, 1), (7, 33), (3, 8193), (self.sm_rows, 8193)):
                with self.subTest(dtype=dtype, op="row_sum", shape=(rows, width)):
                    shape = (rows, width + 3)  # a view with row gaps: the gaps get no gradient
                    backing = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
                    weights = torch.randn(rows, device="cuda")
                    (ops.row_sum(backing[:, :width]) * weights).sum().backward()
                    expected = weights.to(dtype)[:, None].expand(rows, width)
                    torch.testing.assert_close(backing.grad[:, :width], expected, atol=0, rtol=0)
                    self.assertEqual(backing.grad[:, width:].abs().sum().item(), 0)

    def test_shared_gradients_do_not_alias(self):
        """add and RMSNorm hand one tensor to two inputs; each leaf must own its gradient."""
        x, y = (torch.randn(257, device="cuda", requires_grad=True) for _ in range(2))
        ops.add(x, y).sum().backward()
        self.assertNotEqual(x.grad.data_ptr(), y.grad.data_ptr())
        h, r = (torch.randn((3, 33), device="cuda", requires_grad=True) for _ in range(2))
        w = torch.randn(33, device="cuda", requires_grad=True)
        ops.residual_rmsnorm(h, r, w).square().sum().backward()
        self.assertNotEqual(h.grad.data_ptr(), r.grad.data_ptr())
        h.grad.zero_()  # would zero r.grad as well if the two shared storage
        self.assertGreater(r.grad.abs().sum().item(), 0)

    def test_matmul_backward(self):
        """dA and dB match FP64 autograd: ragged, empty, and non-dense upstream gradients."""
        shapes = (
            (1, 1, 1),
            (31, 65, 33),
            (127, 255, 65),
            (257, 129, 96),
            (300, 100, 50),
            (600, 130, 200),
            (0, 3, 4),
            (3, 0, 4),
            (3, 4, 0),
        )
        for dtype in (torch.float16, torch.bfloat16):
            tol = tolerance("matmul", dtype)
            for m, n, k in shapes:
                for layout in self.layouts(dtype, ("row_broadcast", "scalar")):
                    # The autotuner times 8 configs for a new shape: cover it on one shape.
                    tuned = (True, False) if (m, n, k) == (127, 255, 65) else (False,)
                    for autotune in tuned:
                        with self.subTest(
                            dtype=dtype, shape=(m, n, k), grad=layout, tuned=autotune
                        ):
                            a = torch.randn((m, k), device="cuda", dtype=dtype)
                            b = torch.randn((k, n), device="cuda", dtype=dtype)
                            dy = self.upstream(m, n, dtype, layout)
                            a_leaf, b_leaf = a.clone().requires_grad_(), b.clone().requires_grad_()
                            ops.matmul(a_leaf, b_leaf, autotune=autotune).backward(dy)
                            a64, b64 = a.double().requires_grad_(), b.double().requires_grad_()
                            (a64 @ b64).backward(dy.double())
                            torch.testing.assert_close(a_leaf.grad, a64.grad.to(dtype), **tol)
                            torch.testing.assert_close(b_leaf.grad, b64.grad.to(dtype), **tol)

    def test_matmul_backward_frozen_operands_and_direct_call(self):
        half = {"device": "cuda", "dtype": torch.float16}
        a, b, dy = (
            torch.randn((40, 33), **half),
            torch.randn((33, 24), **half),
            torch.randn((40, 24), **half),
        )
        da, db = ops.matmul_backward(dy, a, b)
        only_a = a.clone().requires_grad_()  # b is frozen: dB is never computed
        ops.matmul(only_a, b).backward(dy)
        only_b = b.clone().requires_grad_()  # a is frozen: dA is never computed
        ops.matmul(a, only_b).backward(dy)
        torch.testing.assert_close(only_a.grad, da, atol=0, rtol=0)
        torch.testing.assert_close(only_b.grad, db, atol=0, rtol=0)
        for got, want in zip(ops.matmul_backward(dy, a, b), (da, db)):  # deterministic
            torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_every_backward_gemm_config(self):
        import triton

        from kernel_portfolio import triton_kernels as k

        tried = 0
        for config in k._matmul_configs():
            for m, n, kk in ((127, 255, 65), (257, 129, 96)):
                with self.subTest(config=str(config), shape=(m, n, kk)):
                    half = {"device": "cuda", "dtype": torch.float16}
                    a, b, g = (
                        torch.randn((m, kk), **half),
                        torch.randn((kk, n), **half),
                        torch.randn((m, n), **half),
                    )
                    da = torch.full((m, kk), float("nan"), **half)
                    db = torch.full((kk, n), float("nan"), **half)
                    meta = config.kwargs

                    def run(x, y, out, rows, cols, red, xs, ys, meta=meta, config=config):
                        grid = (triton.cdiv(rows, meta["BM"]) * triton.cdiv(cols, meta["BN"]),)
                        args = (
                            x,
                            y,
                            out,
                            rows,
                            k.m_bucket(rows),
                            red,
                            k.m_bucket(red),
                            *xs,
                            *ys,
                            cols,
                        )
                        k._matmul_strided[grid](
                            *args, **meta, num_warps=config.num_warps, num_stages=config.num_stages
                        )

                    try:
                        run(g, b, da, m, kk, n, g.stride(), b.t().stride())
                        run(a, g, db, kk, n, m, a.t().stride(), g.stride())
                    except triton.runtime.errors.OutOfResources:
                        continue  # the autotuner skips these configs too
                    tol = tolerance("matmul", torch.float16)
                    torch.testing.assert_close(da, (g.double() @ b.double().T).half(), **tol)
                    torch.testing.assert_close(db, (a.double().T @ g.double()).half(), **tol)
                    tried += 1
        # 8 configs on 2 shapes; a GPU with less shared memory may skip a few.
        self.assertGreaterEqual(tried, 8)

    def test_backward_reuses_tuning_across_batch_sizes(self):
        """Token counts in one power-of-two bucket share tuning; a new bucket adds one."""
        from kernel_portfolio import triton_kernels as k

        half = {"device": "cuda", "dtype": torch.float16}
        b = torch.randn((64, 48), **half)

        def run(tokens):
            ops.matmul_backward(
                torch.randn((tokens, 48), **half), torch.randn((tokens, 64), **half), b
            )

        def tuned():
            return len(k._matmul_grad_a_tuned.cache), len(k._matmul_grad_b_tuned.cache)

        run(33)
        before = tuned()
        run(40)
        self.assertEqual(tuned(), before, "33 and 40 tokens share one tuning bucket")
        run(100)
        self.assertEqual(tuned(), (before[0] + 1, before[1] + 1), "100 tokens start a new bucket")

    def test_matmul_saves_only_the_operands_its_gradients_read(self):
        half = {"device": "cuda", "dtype": torch.float16}
        a, b = torch.randn((8, 4), **half), torch.randn((4, 6), **half)
        for a_grad, b_grad, shapes in (
            (True, False, [(4, 6)]),  # dA reads b
            (False, True, [(8, 4)]),  # dB reads a
            (True, True, [(4, 6), (8, 4)]),
        ):
            saved = []

            def pack(tensor, saved=saved):
                saved.append(tuple(tensor.shape))
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                ops.matmul(a.clone().requires_grad_(a_grad), b.clone().requires_grad_(b_grad))
            with self.subTest(a_grad=a_grad, b_grad=b_grad):
                self.assertEqual(saved, shapes)

    def test_double_backward_is_rejected(self):
        x = torch.randn((3, 33), device="cuda", requires_grad=True)
        (g,) = torch.autograd.grad(ops.softmax(x).square().sum(), x, create_graph=True)
        with self.assertRaisesRegex(RuntimeError, "no autograd formula"):
            g.square().sum().backward()

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
            lambda: ops.softmax_backward(x.clone().requires_grad_(), x),
            lambda: ops.softmax_backward(x, x.T.contiguous().T),
            lambda: ops.softmax_backward(x[:, :2], x),
            lambda: ops.residual_rmsnorm_backward(x[:1], x, x, x[0]),
            lambda: ops.residual_rmsnorm(x, x, x[0], eps=0),
            lambda: ops.residual_rmsnorm(x, x, x[0, :2]),
            lambda: ops.add(x[0], x[0, :2]),
            lambda: ops.add(x[0], x[0], block_size=17),
        ):
            with self.assertRaises(ValueError):
                call()
        with self.assertRaises(TypeError):
            ops.matmul(x, x.T.contiguous())
        half = {"device": "cuda", "dtype": torch.float16}
        a, b, g = torch.ones((2, 3), **half), torch.ones((3, 4), **half), torch.ones((2, 4), **half)
        for call in (
            lambda: ops.matmul_backward(torch.ones((2, 5), **half), a, b),
            lambda: ops.matmul_backward(g[:1], a, b),
            lambda: ops.matmul_backward(g, a, b.T),
            lambda: ops.matmul_backward(g, a, torch.ones((5, 4), **half)),  # a is 2x3, b is 5x4
            lambda: ops.matmul_backward(g.clone().requires_grad_(), a, b),
        ):
            with self.assertRaises(ValueError):
                call()
        with self.assertRaises(TypeError):
            ops.matmul_backward(g.float(), a.float(), b.float())

    def test_parameters_train_and_infer(self):
        """A weight Parameter trains under autograd and is accepted under no_grad."""
        x = torch.randn((3, 33), device="cuda")
        weight = torch.nn.Parameter(torch.randn(33, device="cuda"))
        expected = references.residual_rmsnorm(x, x, weight.detach())
        with torch.no_grad():
            actual = ops.residual_rmsnorm(x, x, weight)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, expected, **tolerance("rmsnorm", torch.float32))
        ops.residual_rmsnorm(x, x, weight).square().sum().backward()
        w64 = weight.detach().double().requires_grad_()
        z = 2 * x.double()
        (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5) * w64).square().sum().backward()
        torch.testing.assert_close(
            weight.grad, w64.grad.float(), **tolerance("rmsnorm", torch.float32)
        )
        with self.assertRaisesRegex(ValueError, "not differentiable"):
            ops.softmax_backward(x.clone().requires_grad_(), x)

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

    def test_training_benchmark(self):
        """Forward+backward suites run through autograd in both timing modes."""
        from kernel_portfolio import benchmark

        expected = {
            "softmax_train": {"torch", "triton"},
            "rmsnorm_train": {"torch", "torch_rms_norm", "triton"},
            "matmul_train": {"torch", "triton"},
        }
        shapes = {"softmax_train": "8 33", "rmsnorm_train": "8 33", "matmul_train": "8 33 16"}
        timings = ["events"] + (["graph"] if GRAPHS_OK else [])
        with tempfile.TemporaryDirectory() as directory:
            for op, variants in expected.items():
                for timing in timings:
                    with self.subTest(op=op, timing=timing):
                        path = Path(directory) / f"{op}-{timing}.json"
                        argv = ["--op", op, "--shape", *shapes[op].split(), "--samples", "3"]
                        argv += ["--iterations", "2", "--timing", timing, "--output", str(path)]
                        quiet = io.StringIO()
                        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                            benchmark.main(argv)
                        report = json.loads(path.read_text(encoding="utf-8"))
                        self.assertEqual(set(report["cases"][0]["variants"]), variants)
                        self.assertIn("before", report["gpu_utilization_percent"])

    def test_warm_up_runs_inside_and_outside_inference_mode(self):
        """The burst's operands are cached; allocating them in inference mode must not break
        a later burst outside it (benchmark.main runs in inference mode, other callers do not)."""
        from unittest import mock

        from kernel_portfolio import benchmark

        with mock.patch.dict(benchmark._warm, {"operands": None, "end": 0.0, "bursts": 0}):
            with torch.inference_mode():
                benchmark.warm_gpu(0.01)
            benchmark.warm_gpu(0.01)
            self.assertEqual(benchmark._warm["bursts"], 2)

    def test_report_counts_only_its_own_warm_up_bursts(self):
        from unittest import mock

        from kernel_portfolio import benchmark

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "add.json"
            argv = ["--op", "add", "--shape", "257", "--samples", "3", "--iterations", "2"]
            argv += ["--timing", "events", "--output", str(path)]
            with mock.patch.dict(benchmark._warm, {"bursts": 100}):
                with contextlib.redirect_stdout(io.StringIO()):
                    benchmark.main(argv)
            warmup = json.loads(path.read_text(encoding="utf-8"))["warmup"]
            self.assertGreaterEqual(warmup["bursts"], 1)
            self.assertLess(warmup["bursts"], 100, "the count includes bursts of earlier runs")

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
        plain = ("test_schema", "test_faketensor")
        trained = (*plain, "test_autograd_registration", "test_aot_dispatch_dynamic")
        wide = torch.randn((3, 20000), device="cuda", dtype=torch.float16)

        def leaf(t):
            return t.detach().clone().requires_grad_()

        y = torch.softmax(self.x.float(), -1).half()
        grad_ab = torch.randn((40, 24), device="cuda", dtype=torch.float16)
        for op, args, utils in (
            (ns.softmax.default, (leaf(self.x),), trained),
            (ns.softmax.default, (leaf(wide),), trained),
            (
                ns.residual_rmsnorm.default,
                (leaf(self.x), leaf(self.r), leaf(self.w), 1e-5),
                trained,
            ),
            (ns.row_sum.default, (leaf(self.x),), trained),
            (ns.add.default, (leaf(self.x[0]), leaf(self.r[0])), trained),
            (ns.matmul.default, (leaf(self.a), leaf(self.b)), trained),
            (ns.softmax_backward.default, (torch.randn_like(self.x), y), plain),
            (
                ns.residual_rmsnorm_backward.default,
                (torch.randn_like(self.x), self.x, self.r, self.w, 1e-5),
                plain,
            ),
            (ns.matmul_grad_a.default, (grad_ab, self.b), plain),
            (ns.matmul_grad_b.default, (grad_ab, self.a), plain),
        ):
            with self.subTest(op=str(op), width=args[0].shape[-1]):
                torch.library.opcheck(op, args, test_utils=utils)

    def test_compiled_training_step_matches_eager(self):
        ns = torch.ops.kernel_portfolio

        def step(x, r, w):
            y = ns.softmax(ns.residual_rmsnorm(x, r, w, 1e-5) * 2)
            return (y.float() * torch.arange(y.shape[1], device=y.device)).sum()

        results = []
        for mode in ("eager", "compiled", "dynamic"):
            leaves = [t.detach().clone().requires_grad_() for t in (self.x, self.r, self.w)]
            fn = step if mode == "eager" else torch.compile(step, dynamic=mode == "dynamic")
            loss = fn(*leaves)
            loss.backward()
            results.append((loss.detach(), *(t.grad for t in leaves)))
        for mode, got in zip(("compiled", "dynamic"), results[1:]):
            with self.subTest(mode=mode):
                for a, b in zip(got, results[0]):
                    torch.testing.assert_close(a, b)

    def test_compiled_full_training_step_matches_eager(self):
        """One step through every operator: norm, softmax, GEMM, reduction and add."""
        ns = torch.ops.kernel_portfolio
        offsets = torch.randn(8, device="cuda")

        def step(x, r, w, b, t):
            p = ns.softmax(ns.residual_rmsnorm(x, r, w, 1e-5) * 2)
            s = ns.row_sum(ns.matmul(p, b))
            return (ns.add(s, t) * torch.arange(8, device=s.device)).sum()

        results = []
        for mode in ("eager", "compiled", "dynamic"):
            leaves = [
                t.detach().clone().requires_grad_()
                for t in (self.x, self.r, self.w, self.b, offsets)
            ]
            fn = step if mode == "eager" else torch.compile(step, dynamic=mode == "dynamic")
            loss = fn(*leaves)
            loss.backward()
            results.append((loss.detach(), *(t.grad for t in leaves)))
        for mode, got in zip(("compiled", "dynamic"), results[1:]):
            with self.subTest(mode=mode):
                for a, b in zip(got, results[0]):
                    # Compiled and eager run the same kernels: over 60 seeds the worst
                    # difference was 0.2% of this tolerance.
                    torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)

    def test_custom_ops_train_and_infer(self):
        ns = torch.ops.kernel_portfolio
        for op, args in (
            (ns.row_sum, (self.x,)),
            (ns.matmul, (self.a, self.b)),
            (ns.add, (self.x[0], self.r[0])),
        ):
            with self.subTest(op=str(op)):
                inputs = tuple(x.detach().requires_grad_() for x in args)
                out = op(*inputs)
                self.assertTrue(out.requires_grad)
                out.float().sum().backward()
                for leaf in inputs:
                    self.assertEqual(leaf.grad.shape, leaf.shape)
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


@unittest.skipUnless(GRAPHS_OK, "Sixty training steps are too slow under Compute Sanitizer")
class TrainingExampleTests(_GpuCase):
    """examples/train_tiny.py: a whole model through the operators against plain PyTorch."""

    STEPS = 60

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        path = Path(__file__).resolve().parents[1] / "examples" / "train_tiny.py"
        spec = importlib.util.spec_from_file_location("train_tiny", path)
        cls.tiny = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tiny)

    def check_training(self, **kwargs):
        ours = self.tiny.train(self.tiny.forward_triton, self.STEPS, **kwargs)
        theirs = self.tiny.train(self.tiny.forward_torch, self.STEPS)
        for step, ((a, _), (b, _)) in enumerate(zip(ours, theirs)):
            # BF16 rounding differs per operator; six seeds tracked within 8e-4 relative.
            self.assertLess(abs(a - b) / b, 0.005, f"loss at step {step}")
        for name, history in (("Triton", ours), ("PyTorch", theirs)):
            first, last = history[0][0], history[-1][0]
            self.assertGreater(first / last, 2.0, f"{name} loss barely fell: {first} -> {last}")
            accuracy = sum(item[1] for item in history[-5:]) / 5
            self.assertGreater(accuracy, 0.7, f"{name} accuracy {accuracy}")

    def test_tiny_model_trains_like_pytorch(self):
        self.check_training()

    def test_tiny_model_trains_compiled(self):
        self.check_training(compile_step=True)


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
