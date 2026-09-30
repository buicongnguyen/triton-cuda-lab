"""CPU tests are explicit; a green CPU run is not evidence of GPU correctness."""

import contextlib
import io
import os
import subprocess
import tempfile
import unittest
import unittest.mock

import torch

from kernel_portfolio import contracts as c
from kernel_portfolio import ops, references
from kernel_portfolio.benchmark import main as benchmark_main
from kernel_portfolio.benchmark import revision, summarize


class CpuTests(unittest.TestCase):
    def test_stable_softmax(self):
        x = torch.tensor([[10000.0, 9999.0, -10000.0], [-3.0, -3.0, -3.0]])
        actual = references.softmax_decomposed(x)
        torch.testing.assert_close(actual, torch.softmax(x, -1))
        torch.testing.assert_close(actual.sum(-1), torch.ones(2))

    def test_rmsnorm_semantics(self):
        x = torch.tensor([[1.0, 2.0, 4.0]])
        r, w = torch.ones_like(x), torch.tensor([1.0, 2.0, 3.0])
        z = x.double() + r.double()
        expected = z / (z.square().mean(-1, keepdim=True) + 1e-5).sqrt() * w.double()
        torch.testing.assert_close(references.residual_rmsnorm(x, r, w), expected.float())

    def test_input_contracts(self):
        c.tensor(torch.ones(3), "x", ndim=1, gpu=False)
        for bad in (torch.ones(3, dtype=torch.int32), torch.ones(3, dtype=torch.float64)):
            with self.assertRaises(TypeError):
                c.tensor(bad, "x", ndim=1, gpu=False)
        with self.assertRaises(ValueError):
            c.tensor(torch.ones(3, requires_grad=True), "x", ndim=1, gpu=False)
        with self.assertRaises(ValueError):
            c.same(torch.ones(2), torch.ones(3))
        with self.assertRaises(ValueError):
            c.same(torch.ones(2), torch.ones(2, dtype=torch.float16))
        with self.assertRaises(ValueError):
            c.tensor(torch.ones(2, 2), "x", ndim=1, gpu=False)

    def test_parameters_allowed_without_autograd(self):
        weight = torch.nn.Parameter(torch.ones(3))
        with self.assertRaisesRegex(ValueError, "forward-only"):
            c.tensor(weight, "weight", ndim=1, gpu=False)
        with torch.no_grad():
            c.tensor(weight, "weight", ndim=1, gpu=False)
        with torch.inference_mode():
            c.tensor(weight, "weight", ndim=1, gpu=False)

    def test_rows_contract(self):
        c.rows(torch.ones(4, 12)[:, :7])
        c.rows(torch.ones(0, 12))
        # A single column needs no unit column stride.
        c.rows(torch.ones(4, 12)[:, ::2][:, :1])
        for x in (
            torch.ones(2, 0),
            torch.ones(1, (1 << 20) + 1),
            torch.ones(4, 12)[:, ::2],
            torch.ones(1, 12).expand(4, 12),
        ):
            with self.assertRaises(ValueError):
                c.rows(x)

    def test_epsilon(self):
        c.epsilon(1e-5)
        for eps in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                c.epsilon(eps)

    def test_index_bounds_without_large_allocations(self):
        view = torch.empty_strided((2, 2), (2**31, 1), device="meta")
        with self.assertRaisesRegex(ValueError, "indexing"):
            c.tensor(view, "view", ndim=2, gpu=False)
        c.matmul_shape(1024, 1024)
        with self.assertRaisesRegex(ValueError, "indexing"):
            c.matmul_shape(65536, 65536)
        c.matmul_shape(1, 64 * 65535 + 1)  # the 1D tile grid has no N limit

    def test_no_silent_cpu_fallback(self):
        with self.assertRaisesRegex(ValueError, "NVIDIA"):
            ops.softmax(torch.ones(2, 3))

    def test_revision_ignores_working_directory(self):
        expected = revision()
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                self.assertEqual(revision(), expected)
            finally:
                os.chdir(previous)

    def test_revision_takes_no_git_locks(self):
        # A killed `git status` that had taken .git/index.lock would leave the repo locked.
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, "abc\n", "")

        with unittest.mock.patch("kernel_portfolio.benchmark.subprocess.run", fake_run):
            self.assertEqual(revision(), {"commit": "abc", "dirty": True})
        self.assertTrue(calls)
        for args in calls:
            self.assertEqual(args[:2], ["git", "--no-optional-locks"])

    def test_benchmark_rejects_bad_arguments(self):
        # All of these are rejected before any GPU is needed.
        for argv in (
            ["--samples", "2"],
            ["--op", "all", "--shape", "4"],
            ["--op", "add", "--shape", "4", "5"],
            ["--op", "matmul", "--dtype", "float32"],
            ["--cache", "lukewarm"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    benchmark_main(argv)

    def test_timing_summary(self):
        stats = summarize([3.0, 1.0, 2.0])
        self.assertEqual(stats["median_ms"], 2.0)
        self.assertEqual(stats["samples_ms"], [3.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            summarize([0.0])


if __name__ == "__main__":
    unittest.main()
