"""Regression tests for the learning tools; runnable with python -S (no third parties)."""

import contextlib
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from learning import check
from learning.checks import CPU_EXERCISES, close, equal


class CheckerTests(unittest.TestCase):
    def test_reference_cpu_solutions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check.main(["all", "--solution"]), 0)

    def test_student_starters_are_reported_as_incomplete(self):
        # Use a temporary unfinished exercise so learners may freely edit their real files.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "exercises").mkdir()
            (root / "exercises" / "offsets.py").write_text(
                "def ceil_div(n, block):\n    raise NotImplementedError('finish me')\n"
                "def block_offsets(n, block, pid):\n    raise NotImplementedError('finish me')\n"
            )
            output = io.StringIO()
            with patch.object(check, "ROOT", root), contextlib.redirect_stdout(output):
                self.assertEqual(check.main(["offsets"]), 2)
            self.assertIn("incomplete", output.getvalue())
            self.assertNotIn("REFERENCE SOLUTIONS", output.getvalue())

    def test_unfinished_function_does_not_hide_later_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "exercises").mkdir()
            (root / "exercises" / "offsets.py").write_text(
                "def ceil_div(n, block):\n    raise NotImplementedError('finish me')\n"
                "def block_offsets(n, block, pid):\n"
                "    offsets = [pid * block + i for i in range(block)]\n"
                "    return offsets, [i < n for i in offsets]\n"
            )
            output = io.StringIO()
            with patch.object(check, "ROOT", root), contextlib.redirect_stdout(output):
                self.assertEqual(check.main(["offsets"]), 2)
            self.assertIn("PASS tail mask", output.getvalue())

    def test_wrong_answer_fails(self):
        module = types.SimpleNamespace(answer=lambda: 7)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                check.run_checks(module, [("answer", lambda m: equal(m.answer(), 8))]),
                (0, 1, False),
            )

    def test_numeric_comparison_rejects_nan(self):
        with self.assertRaises(AssertionError):
            close(float("nan"), 1)
        with self.assertRaisesRegex(AssertionError, r"result\[1\]"):
            close([1, 2], [1, 3])

    def test_hints_never_load_student_code(self):
        with patch.object(check, "load_exercise", side_effect=AssertionError("should not load")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check.main(["all", "--hint"]), 0)

    def test_syntax_error_is_readable_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "exercises").mkdir()
            (root / "exercises" / "offsets.py").write_text("def broken syntax:\n")
            output = io.StringIO()
            with patch.object(check, "ROOT", root), contextlib.redirect_stdout(output):
                self.assertEqual(check.main(["offsets"]), 1)
            self.assertIn("LOAD ERROR: SyntaxError", output.getvalue())

    def test_unavailable_gpu_is_not_a_pass(self):
        with patch("learning.gpu_checks.gpu_available", return_value="test has no GPU"):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check.main(["triton_add", "--solution"]), 3)

    def test_reference_files_are_separate(self):
        for name in CPU_EXERCISES:
            self.assertNotEqual(
                (check.ROOT / "exercises" / f"{name}.py").resolve(),
                (check.ROOT / "solutions" / f"{name}.py").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
