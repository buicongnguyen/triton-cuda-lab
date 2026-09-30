"""Triton kernels on the CPU interpreter, so CI executes kernel code without a GPU.

Run with TRITON_INTERPRET=1 set before Python starts. The public ops reject CPU
tensors by design, so these tests call the launchers directly. The interpreter
checks indexing, masking and math; it says nothing about GPU performance or races.
Triton 3.6's interpreter needs NumPy < 2.4: newer NumPy breaks its GEMM loop bounds.
"""

import importlib.util
import os
import unittest
from unittest import mock

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

    def test_matmul_backward_kernels(self):
        # Ragged shapes with a transposed upstream gradient; the fixed tile, no autotuning.
        tol = tolerance("matmul", torch.float16)
        for m, n, k in ((1, 1, 1), (31, 65, 33), (127, 100, 70)):
            with self.subTest(shape=(m, n, k)):
                a = torch.randn((m, k), dtype=torch.float16)
                b = torch.randn((k, n), dtype=torch.float16)
                g = torch.randn((n, m), dtype=torch.float16).T
                da = torch.empty((m, k), dtype=torch.float16)
                db = torch.empty((k, n), dtype=torch.float16)
                self.kernels.launch_matmul_grad_a(g, b, da, False)
                self.kernels.launch_matmul_grad_b(g, a, db, False)
                torch.testing.assert_close(da, (g.double() @ b.double().T).half(), **tol)
                torch.testing.assert_close(db, (a.double().T @ g.double()).half(), **tol)

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


@unittest.skipUnless(importlib.util.find_spec("triton"), "Triton is not installed")
class RowPlanTests(unittest.TestCase):
    """The plan rules as pure functions, for an 80-SM GPU (no GPU or interpreter needed)."""

    SMS = 80

    def plan(self, op, rows, width, device="cuda"):
        from kernel_portfolio import triton_kernels as k

        with mock.patch.object(k, "_sm_count", return_value=self.SMS):
            return k._row_plan(
                op,
                rows,
                width,
                torch.device(device, 0) if device == "cuda" else torch.device(device),
            )

    def test_rows_up_to_8192_fit_one_block(self):
        self.assertEqual(self.plan("softmax", 5, 1), ("block", 1, None))
        self.assertEqual(self.plan("softmax", 5, 1000), ("block", 1024, None))
        self.assertEqual(self.plan("rmsnorm", 1000, 8192), ("block", 8192, None))

    def test_few_wide_rows_split_where_it_paid(self):
        # Fewer rows than SMs: softmax splits above 8192, RMSNorm above 16384, row sum never.
        self.assertEqual(self.plan("softmax", 79, 12288), ("split", 2048, 4))
        self.assertEqual(self.plan("softmax", 1, 65536), ("split", 2048, 4))
        self.assertEqual(self.plan("softmax", 1, 65537), ("split", 4096, 8))
        self.assertEqual(self.plan("rmsnorm", 1, 16385), ("split", 2048, 4))
        self.assertEqual(self.plan("rmsnorm", 1, 16384), ("looped", 8192, 16))
        self.assertEqual(self.plan("row_sum", 1, 100000), ("looped", 8192, 16))

    def test_looped_chunks_follow_row_count_and_width(self):
        # At least one row per SM: narrow rows take 4096-wide chunks, wide rows 8192-wide.
        self.assertEqual(self.plan("softmax", 80, 12288), ("looped", 4096, 8))
        self.assertEqual(self.plan("softmax", 1024, 28672), ("looped", 4096, 8))
        self.assertEqual(self.plan("softmax", 80, 28673), ("looped", 8192, 16))
        self.assertEqual(self.plan("rmsnorm", 1024, 50257), ("looped", 8192, 16))
        # Below one row per SM a looped row still takes the widest chunk.
        self.assertEqual(self.plan("row_sum", 79, 12288), ("looped", 8192, 16))

    def test_cpu_interpreter_plans_reach_both_paths(self):
        self.assertEqual(self.plan("softmax", 1, 12001, "cpu"), ("split", 2048, 4))
        self.assertEqual(self.plan("softmax", 2, 8193, "cpu"), ("looped", 2048, 4))

    def test_every_plan_is_launchable(self):
        from kernel_portfolio import triton_kernels as k

        for op in ("row_sum", "softmax", "rmsnorm"):
            for rows in (1, 7, 79, 80, 160, 2048):
                for width in (1, 33, 8192, 8193, 16385, 28672, 32769, 50257, 131072, 1 << 20):
                    kind, block, warps = self.plan(op, rows, width)
                    with self.subTest(op=op, rows=rows, width=width):
                        self.assertIn(kind, ("block", "looped", "split"))
                        self.assertEqual(block & (block - 1), 0, "chunk is a power of two")
                        self.assertLessEqual(block, k.SINGLE_BLOCK_MAX)
                        if kind != "block":
                            self.assertIn(warps, (4, 8, 16))
                            self.assertGreater(width, k.SINGLE_BLOCK_MAX)

    def test_grid_script_scores_the_current_and_previous_rules(self):
        import contextlib
        import io

        from scripts import row_plan_grid as grid

        # The rule this repository replaced: 2048-wide chunks from two rows per SM.
        self.assertEqual(grid.previous_label("softmax", 4, 32769, 80), "S2048/4")
        self.assertEqual(grid.previous_label("softmax", 100, 32769, 80), "L8192/16")
        self.assertEqual(grid.previous_label("softmax", 160, 32769, 80), "L2048/4")
        self.assertEqual(grid.previous_label("rmsnorm", 4, 12288, 80), "L8192/16")
        times = {
            "L8192/16": 10.0,
            "L4096/8": 8.0,
            "L2048/4": 12.0,
            "L1024/4": 14.0,
            "S2048/4": 9.0,
            "S4096/8": 9.5,
        }
        cell = {
            "op": "softmax",
            "rows": 160,
            "width": 32769,
            "us": times,
            "chosen": "L4096/8",
            "ratio": 9.0 / 8.0,
        }
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            grid.summarize([cell], 80)
        lines = [line.strip() for line in out.getvalue().splitlines()]
        current = next(line for line in lines if line.startswith("current"))
        previous = next(line for line in lines if line.startswith("previous"))
        no_narrow = next(line for line in lines if line.startswith("no narrow rule"))
        # The chosen plan is the best one: every regret column is 0.0%, none over 10%.
        self.assertRegex(current, r"\s0\.0%\s+0\.0%\s+0\.0%\s+0\.0%\s+0 of 1")
        # 2048-wide chunks take 12 us against the best 8 us: 50% regret, over the 10% mark.
        self.assertRegex(previous, r"\s50\.0%\s+50\.0%\s+50\.0%\s+50\.0%\s+1 of 1")
        # 8192-wide chunks take 10 us against the best 8 us.
        self.assertRegex(no_narrow, r"\s25\.0%\s+25\.0%\s+25\.0%\s+25\.0%\s+1 of 1")
        text = " ".join(lines)
        self.assertIn("choose different plans in 1 cells", text)
        self.assertIn("at least 10% faster in 1 of them and at least 10% slower in 0", text)
        # Best split 9 us over best looped 8 us: the loop wins by more than 5% at 160 rows.
        self.assertTrue(any(line.split()[:3] == ["softmax", "160", "1"] for line in lines))


if __name__ == "__main__":
    unittest.main()
