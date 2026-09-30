"""Triton kernels on the CPU interpreter, so CI executes kernel code without a GPU.

Run with TRITON_INTERPRET=1 set before Python starts. The public ops reject CPU
tensors by design, so these tests call the launchers directly. The interpreter
checks indexing, masking and math; it says nothing about GPU performance or races.
Triton 3.6's interpreter needs NumPy < 2.4: newer NumPy breaks its GEMM loop bounds.
"""

import importlib.util
import os
import unittest

import torch

from kernel_portfolio.benchmark import tolerance

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1" and importlib.util.find_spec("triton")
# BF16 tl.dot returns garbage in the Triton 3.6 interpreter; BF16 GEMM stays GPU-only.
ROW_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class InterpreterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not INTERPRETED:
            # CI sets KERNEL_REQUIRE_INTERPRETER=1 so a lost flag cannot pass as a skip.
            if os.environ.get("KERNEL_REQUIRE_INTERPRETER") == "1":
                raise RuntimeError("Interpreter tests required: set TRITON_INTERPRET=1")
            raise unittest.SkipTest("Set TRITON_INTERPRET=1 with Triton installed")
        from kernel_portfolio import triton_kernels

        cls.kernels = triton_kernels
        torch.manual_seed(2026)

    def test_add_tails(self):
        for dtype in (torch.float32, torch.float16):
            for n in (1, 257, 4097):
                with self.subTest(dtype=dtype, n=n):
                    x, y = torch.randn(n, dtype=dtype), torch.randn(n, dtype=dtype)
                    out = torch.empty_like(x)
                    self.kernels.launch_add(x, y, out, 256)
                    torch.testing.assert_close(out, x + y, atol=0, rtol=0)

    def test_row_kernels_with_row_gaps(self):
        for dtype in ROW_DTYPES:
            # On CPU tensors one wide row splits across programs and more rows loop in
            # 2048-wide chunks (see _row_plan): 8193 loops; a single 12001 row splits for
            # softmax; a single 20000 row also splits for RMSNorm.
            for rows, width in ((1, 1), (7, 33), (3, 130), (2, 8193), (1, 12001), (1, 20000)):
                with self.subTest(dtype=dtype, shape=(rows, width)):
                    x = torch.randn((rows, width + 5), dtype=dtype)[:, :width]
                    r = torch.randn((rows, width + 3), dtype=dtype)[:, :width]
                    w = torch.randn(width, dtype=dtype)
                    sums = torch.empty(rows, dtype=torch.float32)
                    self.kernels.launch_row_sum(x, sums)
                    torch.testing.assert_close(
                        sums, x.double().sum(-1).float(), **tolerance("row_sum", dtype)
                    )
                    out = torch.empty((rows, width), dtype=dtype)
                    self.kernels.launch_softmax(x, out, 4)
                    torch.testing.assert_close(
                        out, x.double().softmax(-1).to(dtype), **tolerance("softmax", dtype)
                    )
                    self.kernels.launch_rmsnorm(x, r, w, out, 1e-5)
                    z = x.double() + r.double()
                    expected = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5)
                    torch.testing.assert_close(
                        out, (expected * w.double()).to(dtype), **tolerance("rmsnorm", dtype)
                    )

    def test_masked_softmax(self):
        # -inf entries, including a masked prefix longer than one chunk: one block,
        # a looped row pair and a single split row.
        for rows, width in ((2, 33), (2, 8193), (1, 8193)):
            with self.subTest(shape=(rows, width)):
                x = torch.randn((rows, width))
                x[:, : width // 3] = float("-inf")
                x[:, 1::5] = float("-inf")
                out = torch.empty_like(x)
                self.kernels.launch_softmax(x, out, 4)
                self.assertTrue(torch.isfinite(out).all().item())
                torch.testing.assert_close(
                    out, x.double().softmax(-1).float(), **tolerance("softmax", torch.float32)
                )

    def test_backward_kernels(self):
        # Every row plan: one block, looped and split, with a transposed upstream gradient.
        for rows, width in ((1, 1), (3, 33), (2, 8193), (1, 12001), (1, 20000)):
            with self.subTest(shape=(rows, width)):
                x = torch.randn((rows, width + 5))[:, :width]
                r = torch.randn((rows, width + 3))[:, :width]
                w = torch.randn(width)
                dy = torch.randn((width, rows)).T
                y = torch.softmax(x, -1).contiguous()
                dx = torch.empty_like(y)
                self.kernels.launch_softmax_backward(y, dy, dx)
                y64, dy64 = y.double(), dy.double()
                expected = y64 * (dy64 - (y64 * dy64).sum(-1, keepdim=True))
                torch.testing.assert_close(
                    dx, expected.float(), **tolerance("softmax_backward", torch.float32)
                )
                dz, dw = torch.empty((rows, width)), torch.empty(width)
                self.kernels.launch_rmsnorm_backward(x, r, w, dy, dz, dw, 1e-5)
                x64, r64, w64 = (t.double().requires_grad_() for t in (x, r, w))
                z = x64 + r64
                (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5) * w64).backward(dy64)
                tol = tolerance("rmsnorm", torch.float32)
                torch.testing.assert_close(dz, x64.grad.float(), **tol)
                torch.testing.assert_close(dw, w64.grad.float(), **tol)

    def test_gemm_ragged_tiles(self):
        import triton

        for m, n, k in ((1, 1, 1), (31, 65, 33), (127, 255, 65)):
            with self.subTest(shape=(m, n, k)):
                a = torch.randn((m, k), dtype=torch.float16)
                b = torch.randn((k, n), dtype=torch.float16)
                out = torch.empty((m, n), dtype=torch.float16)
                self.kernels.launch_matmul(a, b, out, False)
                expected = (a.double() @ b.double()).half()
                torch.testing.assert_close(out, expected, **tolerance("matmul", torch.float16))
                # GROUP_M=3 leaves an incomplete final group of tile rows.
                grouped = torch.empty_like(out)
                grid = (triton.cdiv(m, 32) * triton.cdiv(n, 64),)
                self.kernels._matmul[grid](
                    a, b, grouped, m, self.kernels.m_bucket(m), n, k, BM=32, BN=64, BK=32, GROUP_M=3
                )
                torch.testing.assert_close(grouped, expected, **tolerance("matmul", torch.float16))


if __name__ == "__main__":
    unittest.main()
